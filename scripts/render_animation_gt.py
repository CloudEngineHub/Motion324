"""Single-file GT animation renderer.

This file contains the simplified workflow of ``render_animation_gt.py`` in one place.
It keeps the same rendering logic (import, normalize, sample, render, and pack output)
while avoiding multi-module indirection.

./path/to/blender-4.0.0-linux-x64/blender -b -P \
    ./scripts/render_animation_gt.py \
    -- --input_dir ./path/to/glb \
    --output_dir ./path/to/output_dir --camera_idx 0 --frames 33

"""

import argparse
import json
import math
import os
from glob import glob
from typing import List, Optional, Tuple

import bpy
import imageio.v2 as imageio
import numpy as np
import mathutils


def _gather_keyframes(obj):
    """Collect all animation keyframes from object and its shape keys."""
    frames = set()
    ad = obj.animation_data
    if ad and ad.action:
        for fc in ad.action.fcurves:
            for kp in fc.keyframe_points:
                frames.add(int(round(kp.co.x)))

    sk = getattr(obj.data, "shape_keys", None)
    if sk and sk.animation_data and sk.animation_data.action:
        for fc in sk.shape_keys.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                frames.add(int(round(kp.co.x)))

    return sorted(frames)


def _clear_scene():
    """Clear all objects and reset scene state."""
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)

    bpy.context.scene.use_nodes = True
    node_tree = bpy.context.scene.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0
    for action in list(bpy.data.actions):
        bpy.data.actions.remove(action)


def _import_animation_file(filepath):
    """Import animation file in FBX/GLB/GLTF format."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=filepath)
    elif ext == ".ply":
        bpy.ops.wm.ply_import(filepath=filepath)
    else:
        raise RuntimeError(f"Unsupported input file: {filepath}")


def _load_armature(filepath: str, ignore_components=None):
    """Load model and return the first imported ARMATURE/MESH object.

    Args:
        filepath: Input animation file path.
        ignore_components: Optional list of object-name substrings to remove after import.
    """
    ignore_components = ignore_components or []
    before = set(bpy.data.objects)
    _import_animation_file(filepath)

    # Remove non-target imported components (e.g. helper meshes) by name.
    for obj in [x for x in set(bpy.data.objects) - before]:
        if any(key in obj.name for key in ignore_components):
            bpy.data.objects.remove(obj, do_unlink=True)

    armature_candidates = [
        obj
        for obj in bpy.data.objects
        if obj not in before and obj.type in {"ARMATURE", "MESH"}
    ]
    return armature_candidates[0] if armature_candidates else None


def _scene_bbox(ignore_matrix=False, single_obj=None):
    """Compute a bbox from all scene meshes or one mesh."""
    meshes = (
        [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        if single_obj is None
        else [single_obj]
    )
    if not meshes:
        raise RuntimeError("No mesh objects found")

    bbox_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bbox_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    for obj in meshes:
        for v in obj.bound_box:
            p = np.array(v, dtype=np.float64)
            if not ignore_matrix:
                p = np.array((obj.matrix_world @ mathutils.Vector(v))) 
            p = np.array(p[:3], dtype=np.float64)
            bbox_min = np.minimum(bbox_min, p)
            bbox_max = np.maximum(bbox_max, p)
    return bbox_min, bbox_max


def _smooth_scene():
    """Enable auto-smooth for mesh objects."""
    for obj in [x for x in bpy.context.scene.objects if x.type == "MESH"]:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = np.deg2rad(30)


def _normalize_scene(normalize_range=1.0, use_parent_node=True):
    """Normalize the scene to fit into a fixed range around origin."""
    bbox_min, bbox_max = _scene_bbox(ignore_matrix=False, single_obj=None)
    span = bbox_max - bbox_min
    scale = normalize_range / np.max(span)

    offset = -(bbox_min + bbox_max) / 2.0

    if use_parent_node:
        parent = bpy.data.objects.new("NormalizationNode", None)
        bpy.context.scene.collection.objects.link(parent)
        for obj in [o for o in bpy.context.scene.objects if o.parent is None]:
            if obj is not parent:
                obj.parent = parent
                obj.matrix_parent_inverse = parent.matrix_world.inverted()
        parent.scale = (scale, scale, scale)
        parent.location = tuple(offset * scale)
    else:
        for obj in [o for o in bpy.context.scene.objects if o.parent is None]:
            obj.matrix_world.translation += mathutils.Vector(offset)
            obj.matrix_world.translation = obj.matrix_world.translation * scale
            obj.scale = obj.scale * scale

    bpy.ops.object.select_all(action="DESELECT")


def _set_env_map(env_map: str):
    """Set the scene environment texture."""
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True

    node_tree = world.node_tree
    for node in list(node_tree.nodes):
        node_tree.nodes.remove(node)

    env_texture_node = node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    bg_node = node_tree.nodes.new(type="ShaderNodeBackground")
    output_node = node_tree.nodes.new(type="ShaderNodeOutputWorld")
    links = node_tree.links
    links.new(env_texture_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    if env_map:
        env_texture_node.image = bpy.data.images.load(env_map, check_existing=True)


def _get_camera_positions_on_sphere(
    center: Tuple[float, float, float],
    radius: float,
    elevations: List[float],
    azimuths: Optional[List[float]] = None,
):
    """Copy the original sphere-camera math from bpyrenderer."""
    points, mats, elevation_t, azimuth_t = [], [], [], []
    for elev_deg in elevations:
        elev_rad = math.radians(elev_deg)
        for az_deg in azimuths:
            az_rad = math.radians(az_deg)
            phi = 0.5 * math.pi - elev_rad
            elevation_t.append(elev_deg)
            azimuth_t.append(az_rad)

            x = center[0] + radius * math.sin(phi) * math.cos(az_rad)
            y = center[1] + radius * math.sin(phi) * math.sin(az_rad)
            z = center[2] + radius * math.cos(phi)
            cam_pos = mathutils.Vector((x, y, z))

            center_vec = mathutils.Vector(center)
            rotation_euler = (center_vec - cam_pos).to_track_quat("-Z", "Y").to_euler()
            mats.append(
                mathutils.Matrix.Translation(cam_pos) @ rotation_euler.to_matrix().to_4x4()
            )
            points.append(cam_pos)
    return points, mats, elevation_t, azimuth_t


def _add_camera(cam2world_matrix: mathutils.Matrix):
    """Create or reuse scene camera and set its transform."""
    if bpy.context.scene.camera is None:
        bpy.ops.object.camera_add(location=(0, 0, 0))
        for obj in bpy.data.objects:
            if obj.type == "CAMERA":
                bpy.context.scene.camera = obj
                break

    camera = bpy.context.scene.camera
    camera.data.type = "PERSP"
    camera.data.sensor_width = 32
    camera.data.lens = 35
    camera.matrix_world = cam2world_matrix

    frame = bpy.context.scene.frame_end
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    camera.data.keyframe_insert(data_path="type", frame=frame)
    camera.data.keyframe_insert(data_path="sensor_width", frame=frame)
    camera.data.keyframe_insert(data_path="lens", frame=frame)
    return camera


def _set_material_flags(process_materials=False):
    if not process_materials:
        return
    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        try:
            bsdf = material.node_tree.nodes["Principled BSDF"]
            if bsdf.inputs["Normal"].is_linked:
                for link in list(bsdf.inputs["Normal"].links):
                    material.node_tree.links.remove(link)
        except Exception:
            pass
        if material.blend_method == "BLEND":
            material.show_transparent_back = False
        material.blend_method = "OPAQUE"


def _sample_indices(scene_num_frames: int, num_frames: int):
    """Evenly sample animation frames at 1.5x speed, matching original behavior."""
    return np.linspace(0, scene_num_frames - 1, num_frames, dtype=int)


def _write_video(output_dir: str, camera_idx: int, width: int, height: int, fps: int):
    render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
    if not render_files:
        return None

    video_path = os.path.join(output_dir, f"cam{camera_idx}_w.mp4")
    with imageio.get_writer(video_path, fps=fps) as writer:
        white_bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        for file in render_files:
            image = imageio.imread(file)
            rgb = image[:, :, :3]
            if image.shape[-1] == 4:
                alpha = image[:, :, 3:4] / 255.0
                rgb = rgb * alpha + white_bg * (1 - alpha)
            writer.append_data(rgb.astype(np.uint8))
    return video_path


def process(
    model_path: str,
    output_dir: str,
    camera_idx: int = 0,
    num_frames: Optional[int] = None,
    process_materials: bool = False,
    env_map: str = "./examples/brown_photostudio_02_1k.exr",
):
    print(f"Render request: model={model_path}, output={output_dir}, camera={camera_idx}")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Init render engine
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    bpy.context.scene.render.use_high_quality_normals = True

    _clear_scene()

    # 2) Import model
    armature = _load_armature(
        model_path, ignore_components=["Icosphere", "polySurface"]
    )
    _smooth_scene()

    if armature is None:
        raise RuntimeError(f"No imported armature/mesh found in {model_path}")

    # 3) Determine rendering frame count from .npy if available, else fallback
    basename = os.path.basename(model_path).replace(".glb", "")[:20]
    parent_dir = os.path.dirname(model_path).replace("/glbs", "/pcds")
    pcd_dir = None
    if os.path.exists(parent_dir):
        for item in os.listdir(parent_dir):
            if item.startswith(basename) and os.path.isdir(os.path.join(parent_dir, item)):
                pcd_dir = os.path.join(parent_dir, item)
                break

    if pcd_dir is None:
        pcd_dir = os.path.join(parent_dir, basename)

    if os.path.exists(pcd_dir):
        frame_candidates = glob(os.path.join(pcd_dir, "frame_*.npy"))
        resolved_frames = len(frame_candidates)
        print(f"Found {resolved_frames} point-cloud frames in {pcd_dir}")
    elif num_frames is not None:
        resolved_frames = num_frames
        print(f"Point-cloud directory not found: {pcd_dir}, using num_frames={resolved_frames}")
    else:
        resolved_frames = 32
        print(
            f"Point-cloud directory not found: {pcd_dir}, using default num_frames={resolved_frames}"
        )

    # 4) Scene normalization
    _normalize_scene(1.0, use_parent_node=True)
    _set_material_flags(process_materials)
    _set_env_map(env_map)

    # 5) Set frame range (1.5x speed sampling)
    target_frames = int(resolved_frames * 1.5)
    bpy.context.scene.frame_end = max(target_frames - 1, 0)
    scene_frame_start = bpy.context.scene.frame_start
    scene_frame_end = bpy.context.scene.frame_end
    scene_num_frames = scene_frame_end - scene_frame_start + 1

    print(f"Scene frame range: [{scene_frame_start}, {scene_frame_end}]")
    print(f"Scene total frames: {scene_num_frames}")
    print(f"Target render frames: {resolved_frames}")

    azimuth_map = {0: 270, 1: 180, 2: 90, 3: 0}
    print(f"Camera index: {camera_idx}, Azimuth: {azimuth_map[camera_idx]}")
    cam_pos, cam_mats, elevations, azimuths = _get_camera_positions_on_sphere(
        center=(0, 0, 0),
        radius=1.5,
        elevations=[0],
        azimuths=[azimuth_map[camera_idx]],
    )
    cameras = [_add_camera(cam_mats[0])]

    width, height, fps = 512, 512, 16
    scene = bpy.context.scene
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.fps = fps
    scene.render.film_transparent = True

    # 6) Render sampled frames
    sample_idx = _sample_indices(scene_num_frames, resolved_frames)
    print(f"Rendering with sampled frame indices: {sample_idx}")
    for frame_no, src_idx in enumerate(sample_idx):
        scene.frame_set(scene_frame_start + src_idx)
        scene.render.use_compositing = True
        scene.use_nodes = True
        if "Render Layers" not in scene.node_tree.nodes:
            scene.node_tree.nodes.new("CompositorNodeRLayers")
        scene.render.filepath = os.path.join(output_dir, f"render_{frame_no:04d}.png")
        bpy.ops.render.render(write_still=True)

    # 7) Build mp4 and metadata
    video_path = _write_video(output_dir, camera_idx, width, height, fps)
    if video_path:
        print(f"Generated video: {video_path}")

    meta_info = {"width": width, "height": height, "locations": []}
    for i in range(len(cam_pos)):
        meta_info["locations"].append(
            {
                "index": f"{i:04d}",
                "projection_type": cameras[i].data.type,
                "ortho_scale": cameras[i].data.ortho_scale,
                "camera_angle_x": cameras[i].data.angle_x,
                "elevation": elevations[i],
                "azimuth": azimuths[i],
                "transform_matrix": [list(row) for row in cameras[i].matrix_world],
            }
        )
    with open(os.path.join(output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=4)


if __name__ == "__main__":
    import sys

    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render GT animation from one GLB file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Input GLB path.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory root or camera-specific directory.",
    )
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera index 0-3.")
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Optional frame count override.",
    )
    parser.add_argument("--process_materials", action="store_true")
    parser.add_argument("--render_all_cameras", action="store_true")

    args = parser.parse_args(argv)

    try:
        if args.render_all_cameras:
            for idx in range(4):
                process(
                    args.input_dir,
                    os.path.join(args.output_dir, f"camera_{idx}"),
                    camera_idx=idx,
                    num_frames=args.frames,
                    process_materials=args.process_materials,
                )
        else:
            process(
                args.input_dir,
                args.output_dir,
                camera_idx=args.camera_idx,
                num_frames=args.frames,
                process_materials=args.process_materials,
            )
    except SystemExit:
        raise
