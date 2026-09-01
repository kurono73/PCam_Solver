# Movie Clip and scene timeline conversion helpers.


def pcam_get_clip_scene_range(clip):
    frame_start = int(clip.frame_start)
    return frame_start, frame_start + int(clip.frame_duration) - 1


def pcam_scene_to_clip_frame(clip, scene_frame):
    return int(scene_frame) - int(clip.frame_start) + 1 + int(clip.frame_offset)


def pcam_clip_to_scene_frame(clip, clip_frame):
    return int(clip_frame) + int(clip.frame_start) - 1 - int(clip.frame_offset)


def pcam_get_frame_range(props):
    clip = props.target_clip
    if not clip:
        return (1, 1)
    if props.use_custom_range:
        return (min(props.bake_start, props.bake_end), max(props.bake_start, props.bake_end))
    return pcam_get_clip_scene_range(clip)


def pcam_get_reference_frame(context, props, frame_start=None, frame_end=None):
    if frame_start is None or frame_end is None:
        frame_start, frame_end = pcam_get_frame_range(props)
    if props.use_reference_frame_lock:
        return props.reference_frame
    return max(frame_start, min(frame_end, context.scene.frame_current))


def pcam_pick_valid_reference_frame(valid_frames, hint, require_exact=False):
    if not valid_frames:
        return None
    if require_exact:
        return hint if hint in valid_frames else None
    return min(valid_frames, key=lambda frame: (abs(frame - hint), frame))
