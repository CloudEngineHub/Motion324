# Inference using GT mesh from pcds directory + rendered images.
# Loads mesh data (vertices, faces, UV textures) from pcds/<name>_pointclouds/
# instead of loading from a GLB file.

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.append(project_root)

import importlib
import torch
import numpy as np
import imageio
import trimesh
from pathlib import Path
from natsort import natsorted
from PIL import Image
from scipy.spatial import cKDTree

from setup import init_config, init_distributed
from utils.inference_utils import (
    seed_everything, load_checkpoint, smooth_trajectories,
    load_u2net_model, segment_foreground_with_u2net
)
from utils.render import clear_scene, import_glb, drive_mesh_with_trajs_frames_gt
from utils.visualization import visualize_input_data
from dataset.dataset_utils import load_uv_preprocessing_data, track_with_normal_rgb

os.environ['SPCONV_ALGO'] = 'native'
config = init_config()


def load_video_from_path(video_path):
    """
    Load video from file or image directory.

    Args:
        video_path: Path to video file (.mp4, .avi, .mov) or directory containing images

    Returns:
        video_np: numpy array of shape (T, H, W, C), range [0, 255]
    """
    if video_path.lower().endswith(('.mp4', '.avi', '.mov')):
        print(f"Loading video file: {video_path}")
        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        reader.close()
        video_np = np.stack(frames, axis=0)
        print(f"Loaded {len(frames)} frames from video, shape: {video_np.shape}")

    elif os.path.isdir(video_path):
        print(f"Loading images from directory: {video_path}")
        image_paths = natsorted([
            os.path.join(video_path, f)
            for f in os.listdir(video_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        frames = [np.array(Image.open(img_path).convert("RGB")) for img_path in image_paths]
        video_np = np.stack(frames, axis=0)
        print(f"Loaded {len(frames)} images from directory, shape: {video_np.shape}")
    else:
        raise ValueError(f"video_path must be a video file or image directory: {video_path}")

    return video_np


def prepare_mesh_data_from_pcds(config, pcd_dir, device):
    """
    Load and prepare mesh data from a pcds/<name>_pointclouds/ directory,
    following the same data loading pattern as dataset/dyscene.py.

    Expects:
      - pcd_dir/faces.npy        (F, 3) int
      - pcd_dir/frame_0000.npy   (V, 3) float  (reference frame vertices)
      - pcd_dir/uv_face_texture.npz  (face_uvs, texture_array)

    Args:
        config: Configuration object
        pcd_dir: Path to pcds/<name>_pointclouds/ directory
        device: torch device

    Returns:
        input_data: dict ready for the model
        faces_np:   (F, 3) int64 numpy array
        vertices:   (V, 3) float32 numpy array (reference frame)
    """
    pcd_dir = Path(pcd_dir)

    faces_np = np.load(pcd_dir / 'faces.npy').astype(np.int64)
    vertex_np_0 = np.load(pcd_dir / 'frame_0000.npy').astype(np.float32)
    print(f"Loaded pcds data: {vertex_np_0.shape[0]} vertices, {faces_np.shape[0]} faces")

    uv_data_path = pcd_dir / 'uv_face_texture.npz'
    if not uv_data_path.exists():
        raise FileNotFoundError(f"UV texture data not found: {uv_data_path}")
    uv_data = load_uv_preprocessing_data(str(uv_data_path))
    face_uvs = uv_data['face_uvs']
    texture_array = uv_data['texture_array']

    vertices = vertex_np_0.copy()
    ref_mesh = trimesh.Trimesh(vertices=vertices, faces=faces_np, process=False)
    vertex_normals_np = ref_mesh.vertex_normals.astype(np.float32)

    num_shape_samples = getattr(config.training, "num_shape_samples", 16384)
    print(f"Sampling {num_shape_samples} surface points with UV texturing...")
    ref_shape_pcd, ref_shape_normals, ref_shape_rgbs, _ = track_with_normal_rgb(
        init_mesh=ref_mesh,
        vertex_frames=vertices[None, ...],
        faces=faces_np,
        num_samples=num_shape_samples,
        face_uvs=face_uvs,
        texture_array=texture_array,
    )
    ref_shape_pcd = ref_shape_pcd[0]
    ref_shape_normals = ref_shape_normals[0]
    ref_shape_rgbs = ref_shape_rgbs[0]
    print(f"Surface sampling complete: pcd {ref_shape_pcd.shape}, "
          f"normals {ref_shape_normals.shape}, rgbs {ref_shape_rgbs.shape}")

    tree = cKDTree(ref_shape_pcd.numpy())
    _, nearest_indices = tree.query(vertices, k=1)
    vert_rgb = ref_shape_rgbs[nearest_indices].numpy()

    input_data = {
        'ref_shape_pcd': ref_shape_pcd[None].float().to(device),
        'ref_shape_normals': ref_shape_normals[None].float().to(device),
        'ref_shape_rgbs': ref_shape_rgbs[None].float().to(device),
        'ref_pcd': torch.from_numpy(vertices)[None].float().to(device),
        'ref_normal': torch.from_numpy(vertex_normals_np)[None].float().to(device),
        'ref_rgb': torch.from_numpy(vert_rgb)[None].float().to(device),
        'faces': torch.from_numpy(faces_np)[None].long().to(device),
    }

    return input_data, faces_np, vertices


def run_model_inference(model, input_data, video_tensor, config, device):
    """
    Run model inference on video with chunking support.

    Returns:
        trajs: Predicted trajectories (1, T, N, 3)
    """
    chunk_size = config.training.get('frames', 12)
    total_T = video_tensor.shape[0]
    print(f"Total frames: {total_T}, chunk size: {chunk_size}")

    amp_dtype_mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
        'tf32': torch.float32,
    }

    if total_T <= chunk_size:
        print(f"Total frames ({total_T}) <= chunk size ({chunk_size}), processing all at once")
        input_data_chunk = input_data.copy()
        input_data_chunk['rgb_video'] = video_tensor[None].float().to(device)

        with torch.autocast(
            enabled=config.training.get('use_amp', False),
            device_type="cuda",
            dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'bf16')],
        ):
            output_chunk = model(input_data_chunk)

        if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
            trajs = output_chunk['pcd_moved'].float()
            print(f"Trajectories shape: {trajs.shape}")
        else:
            trajs = None
            print("Warning: No pcd_moved in output")
    else:
        slide_size = chunk_size - 1
        chunk_start_indices = list(range(0, total_T - chunk_size + 1, slide_size))
        if chunk_start_indices and (chunk_start_indices[-1] + chunk_size < total_T):
            chunk_start_indices.append(total_T - chunk_size)
        print(f"Chunk start indices: {chunk_start_indices}")

        out_trajs_lst = []

        for i, start_idx in enumerate(chunk_start_indices):
            end_idx = start_idx + chunk_size

            if i == 0:
                rgb_window = video_tensor[0:chunk_size]
            else:
                rgb_window = torch.cat([
                    video_tensor[0:1],
                    video_tensor[start_idx + 1:end_idx],
                ], dim=0)

            input_data_chunk = input_data.copy()
            input_data_chunk['rgb_video'] = rgb_window[None].float().to(device)

            if rgb_window.shape[0] != chunk_size:
                print(f"Warning: Chunk rgb_video frames != chunk_size, skipping")
                continue

            print(f"Processing chunk {i + 1}/{len(chunk_start_indices)}: "
                  f"frames {start_idx}-{end_idx}")

            with torch.autocast(
                enabled=config.training.get('use_amp', False),
                device_type="cuda",
                dtype=amp_dtype_mapping[config.training.get('amp_dtype', 'fp16')],
            ):
                output_chunk = model(input_data_chunk)

            if isinstance(output_chunk, dict) and 'pcd_moved' in output_chunk:
                trajs_chunk = output_chunk['pcd_moved'].float()
                out_trajs_lst.append(trajs_chunk)
            else:
                print("Warning: No pcd_moved in chunk output, skipping")

        if len(out_trajs_lst) > 0:
            if len(chunk_start_indices) >= 2:
                merged_trajs_lst = []
                for i in range(len(out_trajs_lst)):
                    if i == 0 and i != len(out_trajs_lst) - 2:
                        chunk_trajs = out_trajs_lst[i].clone()
                        chunk_trajs[:, 0, :, :] = input_data['ref_pcd']
                        merged_trajs_lst.append(chunk_trajs)
                    elif i < len(out_trajs_lst) - 2:
                        merged_trajs_lst.append(out_trajs_lst[i][:, 1:, ...])
                    elif i == len(out_trajs_lst) - 2:
                        start_a = chunk_start_indices[-2]
                        start_b = chunk_start_indices[-1]
                        keep_len = max(start_b - start_a, 0)
                        if keep_len > 0 and len(out_trajs_lst) != 2:
                            merged_trajs_lst.append(
                                out_trajs_lst[i][:, 1:1 + keep_len, ...])
                        elif keep_len > 0 and i == 0 and len(out_trajs_lst) == 2:
                            chunk_trajs = out_trajs_lst[i].clone()
                            chunk_trajs[:, 0, :, :] = input_data['ref_pcd']
                            merged_trajs_lst.append(
                                chunk_trajs[:, :1 + keep_len, ...])
                    elif i == len(out_trajs_lst) - 1:
                        merged_trajs_lst.append(out_trajs_lst[i][:, 1:, ...])

                if len(merged_trajs_lst) > 0:
                    trajs = torch.cat(merged_trajs_lst, dim=1)
                    print(f"Merged trajectories shape: {trajs.shape}")
                else:
                    trajs = None
                    print("No trajectories obtained")
            else:
                trajs = out_trajs_lst[0].clone()
                trajs[:, 0, :, :] = input_data['ref_pcd']
                print(f"Trajectories shape: {trajs.shape}")
        else:
            trajs = None
            print("No trajectories obtained")

    return trajs


def save_segmented_videos(video_tensor, mask_tensor, output_dir, fps=12):
    """Save segmented videos (black bg, white bg) and mask video."""
    os.makedirs(output_dir, exist_ok=True)

    output_black_path = os.path.join(output_dir, "input_rgb_video_segmented_black.mp4")
    with imageio.get_writer(output_black_path, fps=fps) as writer:
        for i in range(video_tensor.shape[0]):
            frame_uint8 = (video_tensor[i].numpy() * 255).astype(np.uint8)
            writer.append_data(frame_uint8)
    print(f"Saved segmented video (black bg): {output_black_path}")

    output_white_path = os.path.join(output_dir, "input_rgb_video_segmented_white.mp4")
    with imageio.get_writer(output_white_path, fps=fps) as writer:
        for i in range(video_tensor.shape[0]):
            frame = video_tensor[i].numpy()
            mask = mask_tensor[i].numpy()
            frame_white = frame * mask + (1 - mask)
            frame_uint8 = (frame_white * 255).astype(np.uint8)
            writer.append_data(frame_uint8)
    print(f"Saved segmented video (white bg): {output_white_path}")

    output_mask_path = os.path.join(output_dir, "input_foreground_mask.mp4")
    with imageio.get_writer(output_mask_path, fps=fps) as writer:
        for i in range(mask_tensor.shape[0]):
            mask = mask_tensor[i].numpy()
            mask_gray = (mask.squeeze(-1) * 255).astype(np.uint8)
            mask_rgb = np.stack([mask_gray] * 3, axis=-1)
            writer.append_data(mask_rgb)
    print(f"Saved foreground mask video: {output_mask_path}")


def run_inference_on_video(config):
    """Main inference function using GT mesh from pcds directory."""
    seed_everything(777)
    ddp_info = init_distributed(seed=777)
    torch.backends.cuda.matmul.allow_tf32 = config.training.get('use_tf32', True)
    torch.backends.cudnn.allow_tf32 = config.training.get('use_tf32', True)

    module, class_name = config.model.class_name.rsplit(".", 1)
    Motion324 = importlib.import_module(module).__dict__[class_name]
    model = Motion324(config).to(ddp_info.device)

    ckpt_path = config.training.get("resume_ckpt", "")
    if not ckpt_path:
        ckpt_path = os.path.join(config.training.checkpoint_dir, "latest.pt")

    step_info = load_checkpoint(ckpt_path, model, ddp_info.device)
    print(f"Loaded checkpoint from step {step_info['param_update_step']}")
    model.eval()

    use_segmentation = getattr(config, 'use_segmentation', False)
    u2net_model = None
    if use_segmentation:
        print("Loading U2Net model for foreground segmentation...")
        u2net_model = load_u2net_model(device=ddp_info.device)
        if u2net_model is None:
            print("Warning: U2Net model loading failed")
    else:
        print("Foreground segmentation disabled")

    pcd_dir = config.data_dir

    with torch.no_grad():
        input_data, faces_np, vertices = prepare_mesh_data_from_pcds(
            config, pcd_dir, ddp_info.device)

        if not hasattr(config, 'video_path') or config.video_path is None:
            raise ValueError("config.video_path not specified")

        video_np = load_video_from_path(config.video_path)

        if u2net_model is not None:
            masked_frames, masks = segment_foreground_with_u2net(
                video_np, u2net_model, device=ddp_info.device)
            print("Applied foreground segmentation")
        else:
            masked_frames = video_np
            masks = np.ones(video_np.shape[:3] + (1,), dtype=np.float32)

        video_tensor = torch.from_numpy(
            masked_frames.astype(np.float32)).float() / 255.0
        mask_tensor = torch.from_numpy(
            masks.astype(np.float32)).float()

        T = config.training.frames
        start_frame = getattr(config, 'start_frame', 0)

        if start_frame + T <= video_tensor.shape[0]:
            video_tensor = video_tensor[start_frame:start_frame + T]
            mask_tensor = mask_tensor[start_frame:start_frame + T]
        else:
            print(f"Warning: Insufficient frames, using "
                  f"{video_tensor.shape[0] - start_frame} frames from {start_frame}")
            video_tensor = video_tensor[start_frame:]
            mask_tensor = mask_tensor[start_frame:]

        output_video_dir = config.output_dir if hasattr(config, 'output_dir') else "./"
        save_segmented_videos(video_tensor, mask_tensor, output_video_dir, fps=12)

        total_T = video_tensor.shape[0]
        input_data['rgb_video'] = video_tensor

        visualize_input_data(
            input_data,
            save_path=os.path.join(output_video_dir, "input_data_vis.png"))

        trajs = run_model_inference(
            model, input_data, video_tensor, config, ddp_info.device)

        if trajs is not None:
            trajs = smooth_trajectories(
                trajs,
                method='combined',
                motion_threshold=0.002,
                window_size=3,
                sigma=1.0,
                savgol_polyorder=2,
                oneeuro_mincutoff=1.0,
                oneeuro_beta=0.007,
                visualization_dir=None,
            )
            print("Trajectory smoothing complete")

        if trajs is not None:
            trajs_to_use = trajs[:, :total_T, :, :]
            trajs_b = trajs_to_use.clone()
            trajs_b[..., 0] = trajs_to_use[..., 0]      # x -> x
            trajs_b[..., 1] = -trajs_to_use[..., 2]     # y -> -z
            trajs_b[..., 2] = trajs_to_use[..., 1]      # z -> y
            trajs_to_use = trajs_b

            # Resolve GLB path: explicit config or auto-derive from pcds dir
            glb_path = getattr(config, 'glb_path', None)
            if not glb_path:
                pcd_dir_path = Path(pcd_dir)
                name = pcd_dir_path.name.replace('_pointclouds', '')
                glb_path = str(pcd_dir_path.parent.parent / 'glbs' / f'{name}.glb')

            print(f"Loading GLB for animation: {glb_path}")
            clear_scene()
            mesh_objects = import_glb(glb_path)

            drive_mesh_with_trajs_frames_gt(
                mesh_objects,
                trajs_to_use.cpu(),
                os.path.join(output_video_dir, 'output_animation'),
                azi=0,
                ele=0,
                export_format='glb'
            )
            print(f"Animation saved to: {os.path.join(output_video_dir, 'output_animation.glb')}")
        else:
            print("Warning: No trajectories produced, skipping animation")


if __name__ == "__main__":
    run_inference_on_video(config)
