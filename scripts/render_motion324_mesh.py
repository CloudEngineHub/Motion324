"""Render Motion324 mesh animations from GT-style views.

Example:
  ./path/to/blender-4.0.0-linux-x64/blender -b -P \
      ./scripts/render_motion324_mesh.py \
      -- --input_dir ./path/to/output_animation.glb \
      --output_dir ./path/to/output_dir --camera_idx 0
"""

import argparse
import json
import math
import os
import re
import subprocess
from glob import glob
from typing import List, Optional, Tuple

import bpy
import mathutils
import numpy as np


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
        for fc in sk.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                frames.add(int(round(kp.co.x)))
    return sorted(frames)


def _get_animation_frame_range():
    """Collect animation frame range from all objects in scene."""
    all_frames = set()
    for obj in bpy.data.objects:
        all_frames.update(_gather_keyframes(obj))

    if not all_frames:
        return 0, 31

    return min(all_frames), max(all_frames)


def _clear_scene():
    """Clear all imported objects and reset scene state."""
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


def _import_model(filepath):
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


def _load_animation_target(filepath: str, ignore_components=None):
    """Load model and return first imported ARMATURE/MESH object."""
    ignore_components = ignore_components or []
    before = set(bpy.data.objects)
    _import_model(filepath)

    for obj in [x for x in set(bpy.data.objects) - before]:
        if any(key in obj.name for key in ignore_components):
            bpy.data.objects.remove(obj, do_unlink=True)

    armature_candidates = [
        obj
        for obj in bpy.data.objects
        if obj not in before and obj.type in {"ARMATURE", "MESH"}
    ]
    if not armature_candidates:
        return None
    return armature_candidates[0]


def _scene_bbox(ignore_matrix=False, single_obj=None):
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"] if single_obj is None else [single_obj]
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
    for obj in [x for x in bpy.context.scene.objects if x.type == "MESH"]:
        obj.data.use_auto_smooth = True
        obj.data.auto_smooth_angle = np.deg2rad(30)


def _normalize_scene(normalize_range=1.0, use_parent_node=True):
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


def _write_video(output_dir: str, camera_idx: int, width: int, height: int, fps: int):
    render_files = sorted(glob(os.path.join(output_dir, "render_*.png")))
    if not render_files:
        return None

    first_frame_match = re.search(r"render_(\d+)\.png$", render_files[0])
    if first_frame_match is None:
        raise RuntimeError(f"Unexpected rendered frame name: {render_files[0]}")

    video_path = os.path.join(output_dir, f"cam{camera_idx}_w.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=white:s={width}x{height}:r={fps}",
            "-framerate",
            str(fps),
            "-start_number",
            first_frame_match.group(1),
            "-i",
            os.path.join(output_dir, "render_%04d.png"),
            "-filter_complex",
            "[0:v][1:v]overlay=shortest=1,format=yuv420p",
            "-c:v",
            "libx264",
            "-r",
            str(fps),
            video_path,
        ],
        check=True,
    )
    return video_path


def process(
    model_path: str,
    output_dir: str,
    camera_idx: int = 0,
    process_materials: bool = False,
    env_map: str = "./examples/brown_photostudio_02_1k.exr",
):
    print(f"Render request: model={model_path}, output={output_dir}, camera={camera_idx}")
    os.makedirs(output_dir, exist_ok=True)

    # 1) Render engine
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.scene.eevee.taa_render_samples = 64
    bpy.context.scene.eevee.use_gtao = True
    bpy.context.scene.eevee.use_ssr = True
    bpy.context.scene.eevee.use_bloom = True
    if hasattr(bpy.context.scene.render, "use_high_quality_normals"):
        bpy.context.scene.render.use_high_quality_normals = True

    _clear_scene()

    # 2) Import model
    armature = _load_animation_target(
        model_path,
        ignore_components=["Icosphere", "polySurface"],
    )
    _smooth_scene()

    if armature is None:
        raise RuntimeError(f"No imported armature/mesh found in {model_path}")

    # 3) Resolve animation frame range from keyframes
    scene_frame_start, scene_frame_end = _get_animation_frame_range()
    scene = bpy.context.scene
    scene.frame_start = scene_frame_start
    scene.frame_end = scene_frame_end
    print(
        f"Scene frame range: [{scene_frame_start}, {scene_frame_end}]"
    )
    print(
        f"Scene total frames: {scene.frame_end - scene.frame_start + 1}"
    )

    # 4) Scene normalization and materials
    _normalize_scene(1.0, use_parent_node=True)
    _set_material_flags(process_materials)
    _set_env_map(env_map)

    azimuth_map = {0: 270, 1: 180, 2: 90, 3: 0}
    if camera_idx not in azimuth_map:
        raise ValueError(f"camera_idx must be 0-3, got {camera_idx}")
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

    # 6) Render full keyframe range (original speed)
    if scene_frame_start > scene_frame_end:
        raise RuntimeError("Invalid animation frame range")
    scene.render.use_compositing = True
    scene.use_nodes = True
    if "Render Layers" not in scene.node_tree.nodes:
        scene.node_tree.nodes.new("CompositorNodeRLayers")
    scene.render.filepath = os.path.join(output_dir, "render_")
    bpy.ops.render.render(animation=True)

    # 7) Write video and metadata
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


def main():
    argv = []
    argv = os.sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]

    parser = argparse.ArgumentParser(
        description="Render Motion324 mesh animation from one file and GT-view camera angles."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input model path (.glb/.fbx/.gltf/.obj/.ply)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory root or camera-specific directory.",
    )
    parser.add_argument("--camera_idx", type=int, default=0, help="Camera index 0-3.")
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
                    process_materials=args.process_materials,
                )
        else:
            process(
                args.input_dir,
                args.output_dir,
                camera_idx=args.camera_idx,
                process_materials=args.process_materials,
            )
    except SystemExit:
        raise


if __name__ == "__main__":
    main()
