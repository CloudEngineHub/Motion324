import bpy
import os
import math
import imageio
import glob

def _gather_keyframes(obj):
    """Collect all keyframe numbers from an object's action and shape key actions.
    
    Args:
        obj: A Blender object.
        
    Returns:
        list[int]: Sorted list of unique keyframe numbers.
    """
    frames_set = set()
    if obj.animation_data and obj.animation_data.action:
        for fc in obj.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                frames_set.add(int(round(kp.co.x)))
    sk = getattr(obj.data, "shape_keys", None)
    if sk and sk.animation_data and sk.animation_data.action:
        for fc in sk.animation_data.action.fcurves:
            for kp in fc.keyframe_points:
                frames_set.add(int(round(kp.co.x)))
    return sorted(frames_set)


def get_animation_frame_range():
    """Returns the start and end frames of the animation by scanning all scene objects.
    
    Returns:
        Tuple[int, int]: The start and end frames of the animation.
    """
    all_frames = set()
    for obj in bpy.data.objects:
        all_frames.update(_gather_keyframes(obj))

    if not all_frames:
        print("No valid animation keyframes found")
        return 0, 32

    start_frame = min(all_frames)
    end_frame = max(all_frames)
    print(f"Found animation range: {start_frame} - {end_frame} ({len(all_frames)} keyframes)")
    return start_frame, end_frame


def set_cam():
    cam_data = bpy.data.cameras.new(name="Camera")
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = (0.0, -5, 0.0)
    cam_obj.rotation_euler = (math.pi/2, 0, 0)
    bpy.context.scene.camera = cam_obj


def set_light():
    light_data = bpy.data.lights.new(name="PointLight", type='POINT')
    light_data.energy = 1000
    light_obj = bpy.data.objects.new("Point", light_data)
    bpy.context.scene.collection.objects.link(light_obj)
    light_obj.location = (0.0, -6, 0.0)


def import_animation_file(filepath):
    """Import an animation file. Supports .fbx and .glb/.gltf formats.
    
    Args:
        filepath (str): Path to the animation file.
        
    Returns:
        bool: True if import succeeded, False otherwise.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    else:
        print(f"Unsupported file format: {ext}")
        return False
    return True


def find_animation_file(item_anim_dir):
    """Find the animation file (.fbx or .glb) in the given directory or its 'animation' subdirectory.
    
    Args:
        item_anim_dir (str): Path to the animation directory for an item.
        
    Returns:
        str or None: Path to the found animation file, or None if not found.
    """
    search_dirs = [
        item_anim_dir,
        os.path.join(item_anim_dir, 'animation')
    ]
    for search_dir in search_dirs:
        for name in ('output_animation.fbx', 'output_animation.glb', 'output_animation.gltf'):
            candidate = os.path.join(search_dir, name)
            if os.path.exists(candidate):
                return candidate
    return None


def render(base_dir, item):
    anim_dir = os.path.join(base_dir, item)
    output_video = os.path.join(anim_dir, "animation.mp4")

    if os.path.exists(output_video):
        print(f"Skipping {item} — output already exists")
        return

    anim_file = find_animation_file(anim_dir)
    if anim_file is None:
        print(f"No animation file found for {item}, skipping")
        return

    # Clear the scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import the animation file 4 times, each rotated 90 degrees around Z-axis
    # and placed at the 4 corners of the scene
    positions = [(-1, 0, 1), (1, 0, 1), (-1, 0, -1), (1, 0, -1)]
    rotations = [0, math.pi/2, math.pi, 3*math.pi/2]

    imported_obj = None
    for pos, rot in zip(positions, rotations):
        if not import_animation_file(anim_file):
            return

        bpy.context.view_layer.update()
        imported_obj = bpy.context.selected_objects[-1]
        bpy.context.view_layer.objects.active = imported_obj
        imported_obj.select_set(True)
        imported_obj.delta_location = pos
        imported_obj.delta_rotation_euler = (0, 0, rot)
    if imported_obj is not None:
        imported_obj.select_set(False)

    set_cam()
    set_light()

    output_dir = os.path.join(anim_dir, "output_animation")
    frame_start, frame_end = get_animation_frame_range()
    fps = 16

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.resolution_x = 960
    scene.render.resolution_y = 960
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    scene.render.filepath = os.path.join(output_dir, "") 
    scene.render.use_file_extension = True

    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences
    prefs.addons['cycles'].preferences.compute_device_type = 'CUDA'

    bpy.ops.render.render(animation=True)

    print(f"Render complete. Frames saved to: {output_dir}")

    # Collect all JPEG frames in output_dir and sort them
    frame_files = sorted(glob.glob(os.path.join(output_dir, "*.jpg")))

    # Write video using imageio
    with imageio.get_writer(output_video, fps=fps) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Video saved to: {output_video}")
    


if __name__ == "__main__":
    import sys
    import argparse

    # Blender passes script-level args after the '--' separator
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Render animation results")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Path to a single output directory to render "
            "(e.g. ./examples/output or ./examples/tiger_processed/animation). "
            "When omitted, all subdirectories under examples/ are rendered in batch."
        ),
    )
    args = parser.parse_args(argv)

    if args.output_dir:
        # Single-directory mode — compatible with 4D_from_existing.sh / 4D_from_video.sh output
        output_dir = os.path.abspath(args.output_dir)
        base_dir = os.path.dirname(output_dir)
        item = os.path.basename(output_dir)
        print(f"Rendering single directory: {output_dir} ...")
        render(base_dir, item)
    else:
        # Batch mode: scan ../examples/ and render every subdirectory
        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.normpath(os.path.join(scripts_dir, "..", "examples"))

        items = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

        for item in items:
            print(f"Rendering {item} ...")
            render(base_dir, item)
