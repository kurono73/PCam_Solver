# pcam_solver common helpers
import bpy
from mathutils import Vector, Euler, Matrix, Quaternion
import math
import itertools

# --- UI and Validation Helpers ---

def get_track_objects(self, context):
    items = []
    if self.target_clip:
        for i, ob in enumerate(self.target_clip.tracking.objects):
            items.append((str(i), ob.name, ""))
    if not items:
        items.append(("0", "Camera", ""))
    return items

def pcam_required_track_count(props):
    if props.mode == 'CLIP_TRACK':
        return 0
    if props.mode == 'THREE_POINT':
        return 3
    if props.mode == 'TWO_POINT':
        return 2
    return 1

def pcam_depth_reference_required(props):
    if props.mode == 'ONE_POINT':
        return not props.tripod_mode
    if props.mode in {'TWO_POINT', 'THREE_POINT'}:
        if props.apply_to == 'OBJECT':
            return True
        if props.tripod_mode and props.scale_mode == 'NONE':
            return False
        if not props.tripod_mode and props.scale_mode == 'NONE':
            return True
        return props.scale_mode in {'Z_DEPTH', 'FOCAL_LENGTH'}
    if props.mode == 'CLIP_TRACK':
        if props.apply_to == 'OBJECT':
            return True
        if props.scale_mode == 'NONE' and props.tripod_mode:
            return False
        if props.scale_mode == 'NONE' and not props.tripod_mode:
            return True
        return props.scale_mode in {'Z_DEPTH', 'FOCAL_LENGTH'}
    return False

def pcam_get_track_pool(props):
    if not props.target_clip:
        return None
    try:
        return props.target_clip.tracking.objects[int(props.tracking_object_idx)].tracks
    except Exception:
        return None

def pcam_get_frame_range(props):
    clip = props.target_clip
    if not clip:
        return (1, 1)
    if props.use_custom_range:
        return (min(props.bake_start, props.bake_end), max(props.bake_start, props.bake_end))
    return (
        clip.frame_start + clip.frame_offset,
        clip.frame_start + clip.frame_duration - 1 + clip.frame_offset,
    )

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
    return nearest_frame(valid_frames, hint)

def pcam_get_bake_block_reason(context, props):
    if not props.target_clip:
        return "Movie Clip is required."
    cam = context.scene.camera
    if not cam:
        return "Active Camera is required."
    if props.apply_to == 'OBJECT' and not props.target_object:
        return "Target Object is required."
    if (
        props.apply_to == 'CAMERA' and
        props.mode in {'ONE_POINT', 'TWO_POINT', 'THREE_POINT'} and
        cam.location.length_squared <= 1e-12
    ):
        return "Move camera from origin."
    if pcam_depth_reference_required(props) and not props.clip_depth_object:
        return "Depth Reference is required."
    if props.use_reference_frame_lock:
        frame_start, frame_end = pcam_get_frame_range(props)
        if props.reference_frame < frame_start or props.reference_frame > frame_end:
            return "Reference Frame must be inside the bake range."

    track_pool = pcam_get_track_pool(props)
    if track_pool is None:
        return "Track Layer is invalid."

    if props.mode == 'CLIP_TRACK':
        if len(track_pool) == 0:
            return "At least one tracker is required."
        return ""

    required = pcam_required_track_count(props)
    track_names = [props.track_1, props.track_2, props.track_3][:required]
    if any((not name or name == "NONE") for name in track_names):
        return f"{required} tracker{'s are' if required > 1 else ' is'} required."
    if len(set(track_names)) != len(track_names):
        return "Trackers must be different."
    missing = [name for name in track_names if track_pool.get(name) is None]
    if missing:
        return f"Tracker not found: {missing[0]}"
    return ""

# --- Camera and Tracker Geometry Helpers ---

def get_camera_tan(cam_data, lens_value, scene):
    sensor_fit = cam_data.sensor_fit
    pixel_aspect = scene.render.pixel_aspect_x / max(scene.render.pixel_aspect_y, 1e-6)
    res_x = scene.render.resolution_x * (scene.render.resolution_percentage / 100.0) * pixel_aspect
    res_y = scene.render.resolution_y * (scene.render.resolution_percentage / 100.0)
    if sensor_fit == 'VERTICAL' or (sensor_fit == 'AUTO' and res_x < res_y):
        sy = cam_data.sensor_height
        sx = sy * (res_x / res_y)
    else:
        sx = cam_data.sensor_width
        sy = sx * (res_y / res_x)
    f_safe = max(lens_value, 1e-6)
    return (sx / 2.0) / f_safe, (sy / 2.0) / f_safe

def marker_to_camera_ray(marker_co, tan_x, tan_y, cam_data=None):
    shift_x = getattr(cam_data, "shift_x", 0.0) if cam_data is not None else 0.0
    shift_y = getattr(cam_data, "shift_y", 0.0) if cam_data is not None else 0.0
    sensor_fit = getattr(cam_data, "sensor_fit", "AUTO") if cam_data is not None else "AUTO"
    if sensor_fit == 'VERTICAL':
        shift_tan = tan_y
    elif sensor_fit == 'HORIZONTAL':
        shift_tan = tan_x
    else:
        shift_tan = max(tan_x, tan_y)
    return Vector((
        (2.0 * marker_co.x - 1.0) * tan_x + 2.0 * shift_x * shift_tan,
        (2.0 * marker_co.y - 1.0) * tan_y + 2.0 * shift_y * shift_tan,
        -1.0,
    )).normalized()

def get_track_display_co(track, marker):
    co = Vector(marker.co)
    offset = getattr(track, "offset", None)
    if offset is not None:
        co += Vector((offset[0], offset[1]))
    return co

def matrix_without_scale(matrix):
    return Matrix.Translation(matrix.translation) @ matrix.to_quaternion().to_matrix().to_4x4()

def evaluated_matrix_world(context, obj):
    depsgraph = context.evaluated_depsgraph_get()
    try:
        depsgraph.update()
    except Exception:
        pass
    return obj.evaluated_get(depsgraph).matrix_world.copy()

def wrap_pi(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

def frames_with_point_spread(track_data_list, frames, min_dist_sq=1e-12):
    spread_frames = []
    for frame in frames:
        points = [track_data.get(frame) for track_data in track_data_list]
        if any(point is None for point in points):
            continue
        for p1, p2 in itertools.combinations(points, 2):
            if (p2 - p1).length_squared > min_dist_sq:
                spread_frames.append(frame)
                break
    return spread_frames

def frames_with_triangle_area(track_data_list, frames, min_area_sq=1e-12):
    area_frames = []
    if len(track_data_list) < 3:
        return area_frames
    for frame in frames:
        points = [track_data.get(frame) for track_data in track_data_list[:3]]
        if any(point is None for point in points):
            continue
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        if v1.cross(v2).length_squared > min_area_sq:
            area_frames.append(frame)
    return area_frames

def nearest_frame(frames, hint):
    if not frames:
        return None
    return min(frames, key=lambda frame: (abs(frame - hint), frame))

def format_skip_reasons(skip_counts):
    parts = [f"{name}={count}" for name, count in skip_counts.items() if count]
    return ", ".join(parts) if parts else "none"

def max_point_spread(track_data_list, frames):
    max_dist = 0.0
    for frame in frames:
        points = [track_data.get(frame) for track_data in track_data_list]
        if any(point is None for point in points):
            continue
        for p1, p2 in itertools.combinations(points, 2):
            max_dist = max(max_dist, (p2 - p1).length)
    return max_dist

def max_triangle_area_metric(track_data_list, frames):
    max_area = 0.0
    if len(track_data_list) < 3:
        return max_area
    for frame in frames:
        points = [track_data.get(frame) for track_data in track_data_list[:3]]
        if any(point is None for point in points):
            continue
        max_area = max(max_area, (points[1] - points[0]).cross(points[2] - points[0]).length)
    return max_area

def point_cloud_avg_distance(points):
    if not points:
        return 0.0
    centroid = sum(points, Vector()) / len(points)
    return sum((point - centroid).length for point in points) / len(points)

def median_edge_scale(points_from, points_to):
    ratios = []
    count = min(len(points_from), len(points_to))
    for i1, i2 in itertools.combinations(range(count), 2):
        edge_from = (points_from[i2] - points_from[i1]).length
        edge_to = (points_to[i2] - points_to[i1]).length
        if edge_from > 1e-6:
            ratios.append(edge_to / edge_from)
    if not ratios:
        return None
    ratios.sort()
    return ratios[len(ratios) // 2]

# --- Weighting and Curve Stabilization Helpers ---

def marker_center_weight(marker_co, aspect):
    d = math.sqrt(((marker_co.x - 0.5) * aspect) ** 2 + (marker_co.y - 0.5) ** 2)
    return 1.0 + 5.0 * math.exp(-10.0 * (d ** 2))

def weighted_marker_centroid(markers, track_names, aspect, center_weight=False):
    accum = Vector((0.0, 0.0))
    sum_w = 0.0
    for track_name in track_names:
        marker_co = markers.get(track_name)
        if marker_co is None:
            continue
        w = marker_center_weight(marker_co, aspect) if center_weight else 1.0
        accum += marker_co * w
        sum_w += w
    if sum_w <= 1e-9:
        return None
    return accum / sum_w

def weighted_points_centroid(points, weights=None):
    if not points:
        return None
    if weights is None:
        weights = [1.0] * len(points)
    accum = points[0].copy()
    accum *= 0.0
    sum_w = 0.0
    for point, weight in zip(points, weights):
        if weight <= 1e-9:
            continue
        accum += point * weight
        sum_w += weight
    if sum_w <= 1e-9:
        return None
    return accum / sum_w

def select_stable_track_names(frame, frame_sets, fallback_names=None, min_count=2):
    curr_names = set(frame_sets.get(frame, set()))
    if not curr_names:
        return set(fallback_names) if fallback_names else set()

    prev_names = set(frame_sets.get(frame - 1, curr_names))
    next_names = set(frame_sets.get(frame + 1, curr_names))

    stable3 = curr_names & prev_names & next_names
    if len(stable3) >= min_count:
        return stable3

    stable_prev = curr_names & prev_names
    if len(stable_prev) >= min_count:
        return stable_prev

    stable_next = curr_names & next_names
    if len(stable_next) >= min_count:
        return stable_next

    return curr_names if not fallback_names else (curr_names & set(fallback_names) or curr_names)

def triangle_edges(points):
    return [
        points[1] - points[0],
        points[2] - points[1],
        points[0] - points[2],
    ]

def average_twist_roll_angle(ref_edges, curr_edges, view_axis, ref_align_quat=None):
    if view_axis.length_squared < 1e-9:
        return 0.0
    axis = view_axis.normalized()
    angles = []
    weights = []
    for ref_edge, curr_edge in zip(ref_edges, curr_edges):
        if ref_align_quat:
            ref_edge_cmp = ref_align_quat.inverted() @ ref_edge
        else:
            ref_edge_cmp = ref_edge
        if ref_edge_cmp.length_squared < 1e-9 or curr_edge.length_squared < 1e-9:
            continue
        roll_quat = ref_edge_cmp.rotation_difference(curr_edge)
        try:
            swing, twist = roll_quat.to_swing_twist(axis)
        except Exception:
            twist = Quaternion()
        angle = wrap_pi(twist.angle)
        twist_axis = getattr(twist, "axis", None)
        if twist_axis is not None and twist_axis.length_squared > 1e-9 and twist_axis.dot(axis) < 0.0:
            angle = -angle
        angles.append(angle)
        weights.append(min(ref_edge_cmp.length_squared, curr_edge.length_squared))
    if angles and sum(weights) > 1e-6:
        return sum(a * w for a, w in zip(angles, weights)) / sum(weights)
    return 0.0

def average_planar_roll_delta(ref_edges, curr_edges):
    angles = []
    weights = []
    for ref_edge, curr_edge in zip(ref_edges, curr_edges):
        if ref_edge.length_squared < 1e-9 or curr_edge.length_squared < 1e-9:
            continue
        delta = wrap_pi(
            math.atan2(curr_edge.y, curr_edge.x) -
            math.atan2(ref_edge.y, ref_edge.x)
        )
        angles.append(delta)
        weights.append(min(ref_edge.length_squared, curr_edge.length_squared))
    if angles and sum(weights) > 1e-6:
        return sum(a * w for a, w in zip(angles, weights)) / sum(weights)
    return 0.0

def solve_planar_roll_from_points(ref_points, curr_points, weights=None, aspect=1.0):
    if not ref_points or not curr_points or len(ref_points) != len(curr_points):
        return 0.0
    if weights is None:
        weights = [1.0] * len(ref_points)

    c_ref = weighted_points_centroid(ref_points, weights)
    c_curr = weighted_points_centroid(curr_points, weights)
    if c_ref is None or c_curr is None:
        return 0.0

    angles = []
    valid_weights = []
    for ref_point, curr_point, weight in zip(ref_points, curr_points, weights):
        if weight <= 1e-9:
            continue
        v_ref = ref_point - c_ref
        v_curr = curr_point - c_curr
        if v_ref.length_squared <= 1e-9 or v_curr.length_squared <= 1e-9:
            continue
        a_ref = math.atan2(v_ref.y, v_ref.x * aspect)
        a_curr = math.atan2(v_curr.y, v_curr.x * aspect)
        angles.append(wrap_pi(a_curr - a_ref))
        valid_weights.append(weight * min(v_ref.length_squared, v_curr.length_squared))

    if not angles or sum(valid_weights) <= 1e-6:
        return 0.0
    return sum(angle * weight for angle, weight in zip(angles, valid_weights)) / sum(valid_weights)

def stabilize_roll_curve(angle_map, frames, despike_threshold_deg=1.0, smooth_blend=0.35):
    if not angle_map or len(frames) < 3:
        return angle_map.copy()

    ordered = sorted(frames)
    values = []
    prev_val = None
    for frame in ordered:
        val = angle_map.get(frame, 0.0)
        if prev_val is not None:
            val = prev_val + wrap_pi(val - prev_val)
        values.append(val)
        prev_val = val

    threshold = math.radians(despike_threshold_deg)
    despiked = values[:]
    for i in range(len(values)):
        left = max(0, i - 2)
        right = min(len(values), i + 3)
        window = sorted(values[left:right])
        median = window[len(window) // 2]
        if abs(values[i] - median) > threshold:
            despiked[i] = values[i] * 0.15 + median * 0.85

    smoothed = despiked[:]
    for i in range(1, len(despiked) - 1):
        local_avg = (despiked[i - 1] + 2.0 * despiked[i] + despiked[i + 1]) / 4.0
        smoothed[i] = despiked[i] * (1.0 - smooth_blend) + local_avg * smooth_blend

    return {frame: wrap_pi(val) for frame, val in zip(ordered, smoothed)}

def stabilize_scalar_curve(value_map, frames, blend_map=None, max_blend=0.35):
    if not value_map or len(frames) < 3:
        return value_map.copy()

    ordered = sorted(frames)
    smoothed = value_map.copy()
    for i in range(1, len(ordered) - 1):
        frame = ordered[i]
        curr = smoothed.get(frame)
        if curr is None:
            continue
        blend = max_blend * (blend_map.get(frame, 1.0) if blend_map else 1.0)
        if blend <= 1e-6:
            continue
        prev = smoothed.get(ordered[i - 1], curr)
        nxt = smoothed.get(ordered[i + 1], curr)
        local_avg = (prev + 2.0 * curr + nxt) / 4.0
        smoothed[frame] = curr * (1.0 - blend) + local_avg * blend
    return smoothed

def stabilize_vector_curve(value_map, frames, blend_map=None, max_blend=0.35):
    if not value_map or len(frames) < 3:
        return {frame: value.copy() for frame, value in value_map.items()}

    ordered = sorted(frames)
    smoothed = {frame: value.copy() for frame, value in value_map.items()}
    for i in range(1, len(ordered) - 1):
        frame = ordered[i]
        curr = smoothed.get(frame)
        if curr is None:
            continue
        blend = max_blend * (blend_map.get(frame, 1.0) if blend_map else 1.0)
        if blend <= 1e-6:
            continue
        prev = smoothed.get(ordered[i - 1], curr)
        nxt = smoothed.get(ordered[i + 1], curr)
        local_avg = (prev + curr * 2.0 + nxt) * 0.25
        smoothed[frame] = curr.lerp(local_avg, blend)
    return smoothed

def get_transition_segments(frames, blend_map, threshold=0.32):
    ordered = sorted(frames)
    segments = []
    start = None
    for frame in ordered:
        if blend_map.get(frame, 0.0) > threshold:
            if start is None:
                start = frame
        elif start is not None:
            segments.append((start, prev_frame))
            start = None
        prev_frame = frame
    if start is not None:
        segments.append((start, ordered[-1]))
    return segments

def expand_transition_blends(blend_map, frames, radius=2, decay=0.65):
    ordered = sorted(frames)
    expanded = {frame: float(blend_map.get(frame, 0.0)) for frame in ordered}
    for i, frame in enumerate(ordered):
        base = blend_map.get(frame, 0.0)
        if base <= 1e-6:
            continue
        for step in range(1, radius + 1):
            weight = base * (decay ** step)
            if i - step >= 0:
                prev_frame = ordered[i - step]
                expanded[prev_frame] = max(expanded.get(prev_frame, 0.0), weight)
            if i + step < len(ordered):
                next_frame = ordered[i + step]
                expanded[next_frame] = max(expanded.get(next_frame, 0.0), weight)
    return expanded

def bridge_scalar_curve(value_map, frames, blend_map, threshold=0.32, max_bridge_blend=0.7):
    bridged = value_map.copy()
    ordered = sorted(frames)
    index_map = {frame: i for i, frame in enumerate(ordered)}
    for start, end in get_transition_segments(ordered, blend_map, threshold):
        start_idx = index_map[start]
        end_idx = index_map[end]
        if start_idx <= 0 or end_idx >= len(ordered) - 1:
            continue
        prev_frame = ordered[start_idx - 1]
        next_frame = ordered[end_idx + 1]
        prev_value = bridged.get(prev_frame)
        next_value = bridged.get(next_frame)
        if prev_value is None or next_value is None:
            continue
        span = max(1, next_frame - prev_frame)
        seg_blend = min(max_bridge_blend, max(blend_map.get(f, 0.0) for f in ordered[start_idx:end_idx + 1]))
        for frame in ordered[start_idx:end_idx + 1]:
            curr = bridged.get(frame, prev_value)
            t = (frame - prev_frame) / span
            interp = (1.0 - t) * prev_value + t * next_value
            bridged[frame] = curr * (1.0 - seg_blend) + interp * seg_blend
    return bridged

def bridge_vector_curve(value_map, frames, blend_map, threshold=0.32, max_bridge_blend=0.7):
    bridged = {frame: value.copy() for frame, value in value_map.items()}
    ordered = sorted(frames)
    index_map = {frame: i for i, frame in enumerate(ordered)}
    for start, end in get_transition_segments(ordered, blend_map, threshold):
        start_idx = index_map[start]
        end_idx = index_map[end]
        if start_idx <= 0 or end_idx >= len(ordered) - 1:
            continue
        prev_frame = ordered[start_idx - 1]
        next_frame = ordered[end_idx + 1]
        prev_value = bridged.get(prev_frame)
        next_value = bridged.get(next_frame)
        if prev_value is None or next_value is None:
            continue
        span = max(1, next_frame - prev_frame)
        seg_blend = min(max_bridge_blend, max(blend_map.get(f, 0.0) for f in ordered[start_idx:end_idx + 1]))
        for frame in ordered[start_idx:end_idx + 1]:
            curr = bridged.get(frame, prev_value)
            t = (frame - prev_frame) / span
            interp = prev_value.lerp(next_value, t)
            bridged[frame] = curr.lerp(interp, seg_blend)
    return bridged

def bridge_quaternion_curve(quat_map, frames, blend_map, threshold=0.32, max_bridge_blend=0.72):
    bridged = {frame: quat.copy() for frame, quat in quat_map.items()}
    ordered = sorted(frames)
    index_map = {frame: i for i, frame in enumerate(ordered)}
    for start, end in get_transition_segments(ordered, blend_map, threshold):
        start_idx = index_map[start]
        end_idx = index_map[end]
        if start_idx <= 0 or end_idx >= len(ordered) - 1:
            continue
        prev_frame = ordered[start_idx - 1]
        next_frame = ordered[end_idx + 1]
        prev_quat = bridged.get(prev_frame)
        next_quat = bridged.get(next_frame)
        if prev_quat is None or next_quat is None:
            continue
        span = max(1, next_frame - prev_frame)
        seg_blend = min(max_bridge_blend, max(blend_map.get(f, 0.0) for f in ordered[start_idx:end_idx + 1]))
        for frame in ordered[start_idx:end_idx + 1]:
            curr = bridged.get(frame, prev_quat)
            t = (frame - prev_frame) / span
            interp = prev_quat.slerp(next_quat, t)
            bridged[frame] = curr.slerp(interp, seg_blend)
    return bridged

def smooth_quaternion_curve(quat_map, frames, blend_map=None, max_blend=0.24):
    if not quat_map or len(frames) < 3:
        return {frame: quat.copy() for frame, quat in quat_map.items()}

    ordered = sorted(frames)
    smoothed = {frame: quat.copy() for frame, quat in quat_map.items()}
    for i in range(1, len(ordered) - 1):
        frame = ordered[i]
        curr = smoothed.get(frame)
        if curr is None:
            continue
        blend = max_blend * (blend_map.get(frame, 1.0) if blend_map else 1.0)
        if blend <= 1e-6:
            continue
        prev_quat = smoothed.get(ordered[i - 1], curr)
        next_quat = smoothed.get(ordered[i + 1], curr)
        interp = prev_quat.slerp(next_quat, 0.5)
        smoothed[frame] = curr.slerp(interp, blend)
    return smoothed

def smooth_scalar_curve_global(value_map, frames, strength=0.0, passes=1):
    if strength <= 1e-6:
        return value_map.copy()
    smoothed = value_map.copy()
    max_blend = min(0.94, 0.10 + 0.84 * strength)
    for _ in range(max(1, passes)):
        smoothed = stabilize_scalar_curve(smoothed, frames, None, max_blend=max_blend)
    return smoothed

def smooth_vector_curve_global(value_map, frames, strength=0.0, passes=1):
    if strength <= 1e-6:
        return {frame: value.copy() for frame, value in value_map.items()}
    smoothed = {frame: value.copy() for frame, value in value_map.items()}
    max_blend = min(0.94, 0.10 + 0.84 * strength)
    for _ in range(max(1, passes)):
        smoothed = stabilize_vector_curve(smoothed, frames, None, max_blend=max_blend)
    return smoothed

def smooth_quaternion_curve_global(quat_map, frames, strength=0.0, passes=1):
    if strength <= 1e-6:
        return {frame: quat.copy() for frame, quat in quat_map.items()}
    smoothed = {frame: quat.copy() for frame, quat in quat_map.items()}
    max_blend = min(0.94, 0.10 + 0.84 * strength)
    for _ in range(max(1, passes)):
        smoothed = smooth_quaternion_curve(smoothed, frames, None, max_blend=max_blend)
    return smoothed

def track_visibility_streak(frame, frame_sets, track_name, step):
    streak = 0
    cur = frame
    while track_name in frame_sets.get(cur, set()):
        streak += 1
        cur += step
    return streak

def track_stability_weight(frame, frame_sets, track_name):
    prev_len = track_visibility_streak(frame, frame_sets, track_name, -1)
    next_len = track_visibility_streak(frame, frame_sets, track_name, 1)
    rise = min(1.0, prev_len / 4.0)
    fall = min(1.0, next_len / 4.0)
    return 0.18 + 0.82 * min(rise, fall)

# --- Movie Clip Marker Helpers ---

def get_track_marker_co(clip, tracking_object_idx, track_name, scene_frame):
    try:
        track_obj = clip.tracking.objects[int(tracking_object_idx)]
        track = track_obj.tracks.get(track_name)
    except Exception:
        return None
    if not track:
        return None
    f_clip = scene_frame - clip.frame_start + 1 - clip.frame_offset
    marker = track.markers.find_frame(f_clip)
    if not marker or getattr(marker, 'mute', False):
        return None
    return get_track_display_co(track, marker)

def solve_focal_tripod_rotation_from_markers(context, cam_data, clip, tracking_object_idx, track_names, ref_frame, frame, ref_lens, frame_lens, lock_roll=False):
    tan_ref_x, tan_ref_y = get_camera_tan(cam_data, ref_lens, context.scene)
    tan_frame_x, tan_frame_y = get_camera_tan(cam_data, frame_lens, context.scene)
    ray_ref_list = []
    ray_curr_list = []

    for track_name in track_names:
        marker_ref = get_track_marker_co(clip, tracking_object_idx, track_name, ref_frame)
        marker_curr = get_track_marker_co(clip, tracking_object_idx, track_name, frame)
        if marker_ref is None or marker_curr is None:
            continue
        ray_ref_list.append(marker_to_camera_ray(marker_ref, tan_ref_x, tan_ref_y, cam_data))
        ray_curr_list.append(marker_to_camera_ray(marker_curr, tan_frame_x, tan_frame_y, cam_data))

    if len(ray_ref_list) < 1:
        return None
    if lock_roll or len(ray_ref_list) < 2:
        return solve_tripod_rotation_from_rays(ray_ref_list, ray_curr_list, True)

    pan_tilt_quat = solve_tripod_pan_tilt_from_rays(ray_ref_list, ray_curr_list)
    if pan_tilt_quat == Quaternion():
        return Quaternion()

    c_ref = sum(ray_ref_list, Vector((0.0, 0.0, 0.0)))
    if c_ref.length_squared < 1e-9:
        return pan_tilt_quat
    c_ref.normalize()

    curr_aligned = [pan_tilt_quat @ ray for ray in ray_curr_list]
    angles = []
    valid_weights = []
    for ray_ref, ray_curr_aligned in zip(ray_ref_list, curr_aligned):
        ref_proj = ray_ref - ray_ref.project(c_ref)
        curr_proj = ray_curr_aligned - ray_curr_aligned.project(c_ref)
        if ref_proj.length_squared > 1e-6 and curr_proj.length_squared > 1e-6:
            cross = curr_proj.cross(ref_proj)
            sign = -1.0 if cross.dot(c_ref) > 0 else 1.0
            angles.append(curr_proj.angle(ref_proj) * sign)
            valid_weights.append(ref_proj.length_squared)

    if not angles or sum(valid_weights) <= 1e-6:
        return pan_tilt_quat

    delta_roll = sum(a * w for a, w in zip(angles, valid_weights)) / sum(valid_weights)
    return Quaternion(c_ref, -delta_roll) @ pan_tilt_quat

# --- Rotation Solver Helpers ---

def average_quaternions(quaternions, weights=None):
    if not quaternions:
        return Quaternion()
    if weights is None:
        weights = [1.0] * len(quaternions)

    ref_quat = quaternions[0]
    accum = [0.0, 0.0, 0.0, 0.0]
    total_w = 0.0
    for quat, weight in zip(quaternions, weights):
        if weight <= 1e-9:
            continue
        sign = 1.0 if (
            ref_quat.w * quat.w +
            ref_quat.x * quat.x +
            ref_quat.y * quat.y +
            ref_quat.z * quat.z
        ) >= 0.0 else -1.0
        accum[0] += quat.w * weight * sign
        accum[1] += quat.x * weight * sign
        accum[2] += quat.y * weight * sign
        accum[3] += quat.z * weight * sign
        total_w += weight

    if total_w <= 1e-9:
        return ref_quat.copy()

    norm = math.sqrt(sum(v * v for v in accum))
    if norm <= 1e-9:
        return ref_quat.copy()
    return Quaternion((accum[0] / norm, accum[1] / norm, accum[2] / norm, accum[3] / norm))

def dominant_eigenvector_symmetric4(mat):
    a = [[float(mat[r][c]) for c in range(4)] for r in range(4)]
    v = [[1.0 if r == c else 0.0 for c in range(4)] for r in range(4)]
    pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))

    for _ in range(16):
        max_off = max(abs(a[i][j]) for i, j in pairs)
        if max_off <= 1e-10:
            break

        for p, q in pairs:
            apq = a[p][q]
            if abs(apq) <= 1e-10:
                continue

            app = a[p][p]
            aqq = a[q][q]
            tau = (aqq - app) / (2.0 * apq)
            t = 1.0 / (abs(tau) + math.sqrt(1.0 + tau * tau))
            if tau < 0.0:
                t = -t
            c = 1.0 / math.sqrt(1.0 + t * t)
            s = t * c

            for k in range(4):
                if k != p and k != q:
                    akp = a[k][p]
                    akq = a[k][q]
                    a[k][p] = a[p][k] = c * akp - s * akq
                    a[k][q] = a[q][k] = s * akp + c * akq

            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq
            a[p][q] = a[q][p] = 0.0

            for k in range(4):
                vkp = v[k][p]
                vkq = v[k][q]
                v[k][p] = c * vkp - s * vkq
                v[k][q] = s * vkp + c * vkq

    eig_idx = max(range(4), key=lambda idx: a[idx][idx])
    eigenvec = [v[row][eig_idx] for row in range(4)]
    norm = math.sqrt(sum(component * component for component in eigenvec))
    if norm <= 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return tuple(component / norm for component in eigenvec)

def solve_weighted_kabsch_rotation(vec_ref_list, vec_curr_list, lock_roll=False, weights=None):
    if not vec_ref_list or not vec_curr_list or len(vec_ref_list) != len(vec_curr_list):
        return Quaternion()
    if weights is None:
        weights = [1.0] * len(vec_ref_list)

    valid = []
    for vec_ref, vec_curr, weight in zip(vec_ref_list, vec_curr_list, weights):
        if (
            weight > 1e-9 and
            vec_ref.length_squared > 1e-9 and
            vec_curr.length_squared > 1e-9
        ):
            valid.append((vec_ref.normalized(), vec_curr.normalized(), weight))
    if not valid:
        return Quaternion()

    if lock_roll or len(valid) < 2:
        return solve_tripod_pan_tilt_from_rays(
            [vec_ref for vec_ref, _, _ in valid],
            [vec_curr for _, vec_curr, _ in valid],
            [weight for _, _, weight in valid],
        )

    s_xx = s_xy = s_xz = 0.0
    s_yx = s_yy = s_yz = 0.0
    s_zx = s_zy = s_zz = 0.0
    for vec_ref, vec_curr, weight in valid:
        s_xx += weight * vec_curr.x * vec_ref.x
        s_xy += weight * vec_curr.x * vec_ref.y
        s_xz += weight * vec_curr.x * vec_ref.z
        s_yx += weight * vec_curr.y * vec_ref.x
        s_yy += weight * vec_curr.y * vec_ref.y
        s_yz += weight * vec_curr.y * vec_ref.z
        s_zx += weight * vec_curr.z * vec_ref.x
        s_zy += weight * vec_curr.z * vec_ref.y
        s_zz += weight * vec_curr.z * vec_ref.z

    sigma = s_xx + s_yy + s_zz
    k_mat = (
        (sigma, s_yz - s_zy, s_zx - s_xz, s_xy - s_yx),
        (s_yz - s_zy, s_xx - s_yy - s_zz, s_xy + s_yx, s_zx + s_xz),
        (s_zx - s_xz, s_xy + s_yx, -s_xx + s_yy - s_zz, s_yz + s_zy),
        (s_xy - s_yx, s_zx + s_xz, s_yz + s_zy, -s_xx - s_yy + s_zz),
    )

    q = dominant_eigenvector_symmetric4(k_mat)
    quat = Quaternion((q[0], q[1], q[2], q[3]))
    quat.normalize()
    return quat

def solve_tripod_pan_tilt_from_rays(ray_ref_list, ray_curr_list, weights=None):
    if not ray_ref_list or not ray_curr_list or len(ray_ref_list) != len(ray_curr_list):
        return Quaternion()
    if weights is None:
        weights = [1.0] * len(ray_ref_list)

    valid = []
    for ray_ref, ray_curr, weight in zip(ray_ref_list, ray_curr_list, weights):
        if (
            weight > 1e-9 and
            ray_ref.length_squared > 1e-9 and
            ray_curr.length_squared > 1e-9
        ):
            valid.append((ray_ref.normalized(), ray_curr.normalized(), weight))
    if not valid:
        return Quaternion()

    sum_w = sum(weight for _, _, weight in valid)
    c_ref = sum((ray_ref * weight for ray_ref, _, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    c_curr = sum((ray_curr * weight for _, ray_curr, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    if c_ref.length_squared < 1e-9 or c_curr.length_squared < 1e-9:
        return Quaternion()
    c_ref.normalize()
    c_curr.normalize()

    base_quat = c_curr.rotation_difference(c_ref)
    if len(valid) < 2:
        return base_quat

    per_track_quats = [ray_curr.rotation_difference(ray_ref) for ray_ref, ray_curr, _ in valid]
    per_track_weights = [weight for _, _, weight in valid]
    avg_quat = average_quaternions(per_track_quats, per_track_weights)

    forward = Vector((0.0, 0.0, -1.0))
    spread = sum(ray_ref.angle(c_ref) * weight for ray_ref, _, weight in valid) / sum_w
    offset = c_ref.angle(forward)
    blend = min(0.65, max(0.0, (0.32 - spread) * 2.0) + max(0.0, offset - 0.25) * 0.45)
    if blend <= 1e-6:
        return base_quat
    return base_quat.slerp(avg_quat, blend)

def solve_tripod_pan_tilt_from_rays_strict(ray_ref_list, ray_curr_list, weights=None):
    if not ray_ref_list or not ray_curr_list or len(ray_ref_list) != len(ray_curr_list):
        return Quaternion()
    if weights is None:
        weights = [1.0] * len(ray_ref_list)

    valid = []
    for ray_ref, ray_curr, weight in zip(ray_ref_list, ray_curr_list, weights):
        if (
            weight > 1e-9 and
            ray_ref.length_squared > 1e-9 and
            ray_curr.length_squared > 1e-9
        ):
            valid.append((ray_ref.normalized(), ray_curr.normalized(), weight))
    if not valid:
        return Quaternion()

    sum_w = sum(weight for _, _, weight in valid)
    c_ref = sum((ray_ref * weight for ray_ref, _, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    c_curr = sum((ray_curr * weight for _, ray_curr, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    if c_ref.length_squared < 1e-9 or c_curr.length_squared < 1e-9:
        return Quaternion()
    return c_curr.normalized().rotation_difference(c_ref.normalized())

def solve_single_ray_pan_tilt(ray_ref, ray_curr):
    if ray_ref.length_squared < 1e-9 or ray_curr.length_squared < 1e-9:
        return Quaternion()

    ref = ray_ref.normalized()
    curr = ray_curr.normalized()
    cy = curr.y
    cz = curr.z
    radius = math.sqrt(cy * cy + cz * cz)
    if radius < 1e-9:
        return curr.rotation_difference(ref)

    target_y = max(-radius, min(radius, ref.y))
    phi = math.atan2(-cz, cy)
    acos_val = math.acos(max(-1.0, min(1.0, target_y / radius)))
    candidates = []
    for tilt in (phi + acos_val, phi - acos_val):
        tilt = wrap_pi(tilt)
        tilt_mat = Matrix.Rotation(tilt, 3, 'X')
        tilted = tilt_mat @ curr
        ux, uz = tilted.x, tilted.z
        denom = ux * ux + uz * uz
        if denom < 1e-9:
            continue
        sin_pan = (ref.x * uz - ref.z * ux) / denom
        cos_pan = (ref.x * ux + ref.z * uz) / denom
        pan = wrap_pi(math.atan2(sin_pan, cos_pan))
        pan_tilt = Matrix.Rotation(pan, 3, 'Y') @ tilt_mat
        solved = pan_tilt @ curr
        error = (solved - ref).length
        cost = error + 1e-4 * (abs(pan) + abs(tilt))
        candidates.append((cost, pan, tilt))

    if not candidates:
        return curr.rotation_difference(ref)
    _cost, pan, tilt = min(candidates, key=lambda item: item[0])
    return (Matrix.Rotation(pan, 3, 'Y') @ Matrix.Rotation(tilt, 3, 'X')).to_quaternion()

def solve_single_ray_euler_y_locked(reference_quat, ray_ref, ray_curr, init_euler, rotation_mode):
    if ray_ref.length_squared < 1e-9 or ray_curr.length_squared < 1e-9:
        return reference_quat.copy()
    if rotation_mode in {'QUATERNION', 'AXIS_ANGLE'}:
        return reference_quat @ solve_single_ray_pan_tilt(ray_ref, ray_curr)

    target_dir = reference_quat @ ray_ref.normalized()
    curr = ray_curr.normalized()
    x = float(init_euler.x)
    y = float(init_euler.y)
    z = float(init_euler.z)

    def quat_from_xz(xv, zv):
        return Euler((xv, y, zv), rotation_mode).to_quaternion()

    def residual(xv, zv):
        return quat_from_xz(xv, zv) @ curr - target_dir

    eps = 1e-5
    for _ in range(12):
        r = residual(x, z)
        if r.length < 1e-8:
            break
        rx = (residual(x + eps, z) - residual(x - eps, z)) / (2.0 * eps)
        rz = (residual(x, z + eps) - residual(x, z - eps)) / (2.0 * eps)
        a00 = rx.dot(rx)
        a01 = rx.dot(rz)
        a11 = rz.dot(rz)
        b0 = -rx.dot(r)
        b1 = -rz.dot(r)
        det = a00 * a11 - a01 * a01
        if abs(det) < 1e-12:
            break
        dx = (b0 * a11 - b1 * a01) / det
        dz = (a00 * b1 - a01 * b0) / det
        dx = max(-0.25, min(0.25, dx))
        dz = max(-0.25, min(0.25, dz))
        x = wrap_pi(x + dx)
        z = wrap_pi(z + dz)
        if abs(dx) + abs(dz) < 1e-9:
            break

    solved = quat_from_xz(x, z)
    old_error = (reference_quat @ (solve_single_ray_pan_tilt(ray_ref, ray_curr) @ curr) - target_dir).length
    new_error = (solved @ curr - target_dir).length
    return solved if new_error <= old_error + 1e-5 else reference_quat @ solve_single_ray_pan_tilt(ray_ref, ray_curr)

def solve_tripod_rotation_from_rays(ray_ref_list, ray_curr_list, lock_roll=False, weights=None):
    if not ray_ref_list or not ray_curr_list or len(ray_ref_list) != len(ray_curr_list):
        return Quaternion()
    if weights is None:
        weights = [1.0] * len(ray_ref_list)

    pan_tilt_quat = (
        solve_tripod_pan_tilt_from_rays_strict(ray_ref_list, ray_curr_list, weights)
        if lock_roll else
        solve_tripod_pan_tilt_from_rays(ray_ref_list, ray_curr_list, weights)
    )
    if pan_tilt_quat == Quaternion():
        return Quaternion()

    c_ref = sum((ray * weight for ray, weight in zip(ray_ref_list, weights)), Vector((0.0, 0.0, 0.0)))
    if c_ref.length_squared < 1e-9:
        return pan_tilt_quat
    c_ref.normalize()
    delta_roll = 0.0

    if not lock_roll and len(ray_ref_list) >= 2:
        curr_aligned = [pan_tilt_quat @ ray for ray in ray_curr_list]
        angles = []
        valid_weights = []
        for ray_ref, ray_curr_aligned, weight in zip(ray_ref_list, curr_aligned, weights):
            ref_proj = ray_ref - ray_ref.project(c_ref)
            curr_proj = ray_curr_aligned - ray_curr_aligned.project(c_ref)
            if ref_proj.length_squared > 1e-6 and curr_proj.length_squared > 1e-6:
                cross = curr_proj.cross(ref_proj)
                sign = -1.0 if cross.dot(c_ref) > 0 else 1.0
                angles.append(curr_proj.angle(ref_proj) * sign)
                valid_weights.append(weight * ref_proj.length_squared)
        if angles and sum(valid_weights) > 1e-6:
            delta_roll = sum(a * w for a, w in zip(angles, valid_weights)) / sum(valid_weights)

    return Quaternion(c_ref, delta_roll) @ pan_tilt_quat

def enforce_roll_sign_continuity(base_quat, ray_ref_list, ray_curr_list, view_axis, weights=None):
    if base_quat == Quaternion() or len(ray_ref_list) < 2 or len(ray_ref_list) != len(ray_curr_list):
        return base_quat
    if view_axis.length_squared < 1e-9:
        return base_quat
    if weights is None:
        weights = [1.0] * len(ray_ref_list)

    axis = view_axis.normalized()
    try:
        swing_quat, _twist_quat = base_quat.to_swing_twist(axis)
    except Exception:
        return base_quat

    curr_aligned = [swing_quat @ ray for ray in ray_curr_list]
    angles = []
    valid_weights = []
    for ray_ref, ray_curr_aligned, weight in zip(ray_ref_list, curr_aligned, weights):
        if weight <= 1e-9:
            continue
        ref_proj = ray_ref - ray_ref.project(axis)
        curr_proj = ray_curr_aligned - ray_curr_aligned.project(axis)
        if ref_proj.length_squared > 1e-6 and curr_proj.length_squared > 1e-6:
            cross = curr_proj.cross(ref_proj)
            sign = -1.0 if cross.dot(axis) > 0 else 1.0
            angles.append(curr_proj.angle(ref_proj) * sign)
            valid_weights.append(weight * ref_proj.length_squared)

    if not angles or sum(valid_weights) <= 1e-6:
        return base_quat

    delta_roll = sum(angle * weight for angle, weight in zip(angles, valid_weights)) / sum(valid_weights)
    return Quaternion(axis, delta_roll) @ swing_quat

def replace_quaternion_twist(base_quat, view_axis, twist_angle):
    if base_quat == Quaternion() or view_axis.length_squared < 1e-9:
        return base_quat
    axis = view_axis.normalized()
    try:
        swing_quat, _twist_quat = base_quat.to_swing_twist(axis)
    except Exception:
        return base_quat
    return Quaternion(axis, twist_angle) @ swing_quat

def preserve_camera_roll_from_reference(candidate_quat, reference_quat):
    if candidate_quat == Quaternion() or reference_quat == Quaternion():
        return candidate_quat

    ref_forward = reference_quat @ Vector((0.0, 0.0, -1.0))
    cand_forward = candidate_quat @ Vector((0.0, 0.0, -1.0))
    if ref_forward.length_squared < 1e-9 or cand_forward.length_squared < 1e-9:
        return candidate_quat

    return ref_forward.normalized().rotation_difference(cand_forward.normalized()) @ reference_quat

def stabilize_camera_roll_step(candidate_quat, reference_quat, max_step_rad=math.radians(35.0)):
    if candidate_quat == Quaternion() or reference_quat == Quaternion():
        return candidate_quat

    view_axis = candidate_quat @ Vector((0.0, 0.0, -1.0))
    if view_axis.length_squared < 1e-9:
        return candidate_quat
    view_axis.normalize()

    roll_locked_quat = preserve_camera_roll_from_reference(candidate_quat, reference_quat)
    roll_delta_quat = candidate_quat @ roll_locked_quat.inverted()
    roll_delta = signed_twist_angle(roll_delta_quat, view_axis)
    if abs(roll_delta) <= max_step_rad:
        return candidate_quat

    roll_delta = max(-max_step_rad, min(max_step_rad, roll_delta))
    return Quaternion(view_axis, roll_delta) @ roll_locked_quat

def signed_twist_angle(quat, axis):
    if quat == Quaternion() or axis.length_squared < 1e-9:
        return 0.0
    try:
        _swing, twist = quat.to_swing_twist(axis.normalized())
    except Exception:
        return 0.0
    angle = wrap_pi(twist.angle)
    twist_axis = getattr(twist, "axis", None)
    if twist_axis is not None and twist_axis.length_squared > 1e-9 and twist_axis.dot(axis) < 0.0:
        angle = -angle
    return angle

def soft_reanchor_rotation(current_rot_mat, desired_rot_mat, anchor_count, blend_scale=1.0):
    if anchor_count <= 0:
        return current_rot_mat

    current_quat = current_rot_mat.to_quaternion()
    desired_quat = desired_rot_mat.to_quaternion()
    delta_quat = desired_quat @ current_quat.inverted()
    delta_angle = abs(delta_quat.angle)

    blend = min(0.28, 0.10 + 0.05 * min(anchor_count - 1, 3))
    if delta_angle > math.radians(8.0):
        blend *= 0.35
    elif delta_angle > math.radians(4.0):
        blend *= 0.55
    elif delta_angle > math.radians(2.0):
        blend *= 0.75
    blend *= max(0.0, blend_scale)
    blend = min(0.28, blend)

    if blend <= 1e-6:
        return current_rot_mat

    step_quat = Quaternion().slerp(delta_quat, blend)
    twist_axis = (current_quat @ Vector((0.0, 0.0, -1.0))).normalized()
    quat_vec = Vector((step_quat.x, step_quat.y, step_quat.z))
    twist_vec = twist_axis * quat_vec.dot(twist_axis)
    if twist_vec.length_squared > 1e-12:
        twist_quat = Quaternion((step_quat.w, twist_vec.x, twist_vec.y, twist_vec.z))
        twist_quat.normalize()
        swing_quat = step_quat @ twist_quat.inverted()
        twist_sign = 1.0 if twist_vec.dot(twist_axis) >= 0.0 else -1.0
        twist_angle = twist_quat.angle * twist_sign
        boosted_twist = twist_angle * 1.35
        twist_cap = abs(twist_angle) + math.radians(0.35)
        boosted_twist = max(-twist_cap, min(twist_cap, boosted_twist))
        step_quat = swing_quat @ Quaternion(twist_axis, boosted_twist)
    return (step_quat @ current_quat).to_matrix().to_4x4()

# --- Depth Reference and Fixed-Point Solve Helpers ---

def raycast_marker_world(context, cam, depth_obj, marker_co):
    hit = raycast_marker_world_with_normal(context, cam, depth_obj, marker_co)
    return hit[0] if hit is not None else None

def raycast_marker_world_with_normal(context, cam, depth_obj, marker_co):
    if not cam or not depth_obj:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    cam_eval = cam.evaluated_get(depsgraph)
    obj_eval = depth_obj.evaluated_get(depsgraph)
    cam_mat = cam_eval.matrix_world
    origin = cam_mat.translation
    tan_x, tan_y = get_camera_tan(cam_eval.data, cam_eval.data.lens, context.scene)
    v_cam = marker_to_camera_ray(marker_co, tan_x, tan_y, cam_eval.data)
    v_world = cam_mat.to_3x3() @ v_cam

    mat_inv = obj_eval.matrix_world.inverted()
    ray_origin = mat_inv @ origin
    dir_loc = (mat_inv.to_3x3() @ v_world).normalized()
    success, loc, normal, face_index = obj_eval.ray_cast(ray_origin, dir_loc)
    if success:
        normal_world = obj_eval.matrix_world.inverted().transposed().to_3x3() @ normal
        if normal_world.length_squared > 1e-12:
            normal_world.normalize()
        return obj_eval.matrix_world @ loc, normal_world
    return None

def raycast_marker_world_from_matrix(context, cam_data, cam_matrix, depth_obj, marker_co, lens_value):
    if cam_data is None or depth_obj is None:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = depth_obj.evaluated_get(depsgraph)
    origin = cam_matrix.translation
    tan_x, tan_y = get_camera_tan(cam_data, lens_value, context.scene)
    v_cam = marker_to_camera_ray(marker_co, tan_x, tan_y, cam_data)
    v_world = cam_matrix.to_3x3() @ v_cam

    mat_inv = obj_eval.matrix_world.inverted()
    ray_origin = mat_inv @ origin
    dir_loc = (mat_inv.to_3x3() @ v_world).normalized()
    success, loc, normal, face_index = obj_eval.ray_cast(ray_origin, dir_loc)
    if success:
        return obj_eval.matrix_world @ loc
    return None

def map_anchor_by_segment_in_camera_space(p1_start, p2_start, p1_curr, p2_curr, anchor_start, cam_matrix):
    cam_inv = cam_matrix.inverted()
    s1 = cam_inv @ p1_start
    s2 = cam_inv @ p2_start
    c1 = cam_inv @ p1_curr
    c2 = cam_inv @ p2_curr
    anchor = cam_inv @ anchor_start

    start_mid = (s1 + s2) * 0.5
    curr_mid = (c1 + c2) * 0.5
    start_vec = Vector((s2.x - s1.x, s2.y - s1.y))
    curr_vec = Vector((c2.x - c1.x, c2.y - c1.y))
    if start_vec.length_squared < 1e-9 or curr_vec.length_squared < 1e-9:
        return None

    start_x = start_vec.normalized()
    start_y = Vector((-start_x.y, start_x.x))
    curr_x = curr_vec.normalized()
    curr_y = Vector((-curr_x.y, curr_x.x))
    scale = curr_vec.length / start_vec.length

    offset = Vector((anchor.x - start_mid.x, anchor.y - start_mid.y))
    u = offset.dot(start_x)
    v = offset.dot(start_y)
    curr_xy = Vector((curr_mid.x, curr_mid.y)) + (curr_x * u + curr_y * v) * scale
    curr_z = curr_mid.z + (anchor.z - start_mid.z) * scale
    return cam_matrix @ Vector((curr_xy.x, curr_xy.y, curr_z))

def refine_rotation_center_alignment(base_quat, desired_dirs, observed_rays, weights=None):
    if base_quat == Quaternion() or not desired_dirs or not observed_rays or len(desired_dirs) != len(observed_rays):
        return base_quat
    if weights is None:
        weights = [1.0] * len(desired_dirs)

    valid = []
    for desired_dir, observed_ray, weight in zip(desired_dirs, observed_rays, weights):
        if weight <= 1e-9 or desired_dir.length_squared <= 1e-9 or observed_ray.length_squared <= 1e-9:
            continue
        valid.append((desired_dir.normalized(), observed_ray.normalized(), weight))
    if not valid:
        return base_quat

    sum_w = sum(weight for _, _, weight in valid)
    desired_center = sum((desired_dir * weight for desired_dir, _, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    observed_center = sum((observed_ray * weight for _, observed_ray, weight in valid), Vector((0.0, 0.0, 0.0))) / sum_w
    if desired_center.length_squared < 1e-9 or observed_center.length_squared < 1e-9:
        return base_quat

    desired_center.normalize()
    solved_center = base_quat @ observed_center.normalized()
    if solved_center.length_squared < 1e-9:
        return base_quat
    solved_center.normalize()

    correction = solved_center.rotation_difference(desired_center)
    if correction.angle > math.radians(85.0):
        return base_quat
    return correction @ base_quat

def solve_point_ray_camera_location(points_world, rays_world, fallback_loc, weights=None):
    if not points_world or not rays_world or len(points_world) != len(rays_world):
        return fallback_loc.copy()
    if weights is None:
        weights = [1.0] * len(points_world)

    mat = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    rhs = Vector((0.0, 0.0, 0.0))
    valid_count = 0
    for point, ray, weight in zip(points_world, rays_world, weights):
        if weight <= 1e-9 or ray.length_squared <= 1e-9:
            continue
        direction = ray.normalized()
        projector = Matrix.Identity(3)
        for row in range(3):
            for col in range(3):
                projector[row][col] -= direction[row] * direction[col]
        mat += projector * weight
        rhs += (projector @ point) * weight
        valid_count += 1

    if valid_count < 2:
        return fallback_loc.copy()
    try:
        return mat.inverted() @ rhs
    except Exception:
        return fallback_loc.copy()

def solve_camera_location_on_depth_plane(points_world, rays_world, fallback_loc, view_dir, depth_anchor_loc, weights=None):
    if not points_world or not rays_world or len(points_world) != len(rays_world):
        return fallback_loc.copy()
    if view_dir.length_squared <= 1e-12:
        return fallback_loc.copy()
    axis_z = view_dir.normalized()
    axis_x = axis_z.cross(Vector((0.0, 1.0, 0.0)))
    if axis_x.length_squared <= 1e-12:
        axis_x = axis_z.cross(Vector((1.0, 0.0, 0.0)))
    if axis_x.length_squared <= 1e-12:
        return fallback_loc.copy()
    axis_x.normalize()
    axis_y = axis_z.cross(axis_x).normalized()
    if weights is None:
        weights = [1.0] * len(points_world)

    a00 = a01 = a11 = 0.0
    b0 = b1 = 0.0
    valid_count = 0
    for point, ray, weight in zip(points_world, rays_world, weights):
        if weight <= 1e-9 or ray.length_squared <= 1e-9:
            continue
        direction = ray.normalized()
        projector = Matrix.Identity(3)
        for row in range(3):
            for col in range(3):
                projector[row][col] -= direction[row] * direction[col]
        px = projector @ axis_x
        py = projector @ axis_y
        residual = projector @ (depth_anchor_loc - point)
        a00 += weight * px.dot(px)
        a01 += weight * px.dot(py)
        a11 += weight * py.dot(py)
        b0 += -weight * px.dot(residual)
        b1 += -weight * py.dot(residual)
        valid_count += 1

    if valid_count < 2:
        return fallback_loc.copy()
    det = a00 * a11 - a01 * a01
    if abs(det) <= 1e-12:
        return fallback_loc.copy()
    x = (b0 * a11 - b1 * a01) / det
    y = (a00 * b1 - a01 * b0) / det
    return depth_anchor_loc + axis_x * x + axis_y * y

def solve_point_from_rays(ray_origins, ray_directions, fallback_point, weights=None):
    if not ray_origins or not ray_directions or len(ray_origins) != len(ray_directions):
        return fallback_point.copy()
    if weights is None:
        weights = [1.0] * len(ray_origins)

    mat = Matrix(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    rhs = Vector((0.0, 0.0, 0.0))
    valid_count = 0
    for origin, direction, weight in zip(ray_origins, ray_directions, weights):
        if weight <= 1e-9 or direction.length_squared <= 1e-9:
            continue
        ray = direction.normalized()
        projector = Matrix.Identity(3)
        for row in range(3):
            for col in range(3):
                projector[row][col] -= ray[row] * ray[col]
        mat += projector * weight
        rhs += (projector @ origin) * weight
        valid_count += 1

    if valid_count < 2:
        return fallback_point.copy()
    try:
        return mat.inverted() @ rhs
    except Exception:
        return fallback_point.copy()

def solve_rotation_quat_at_location(points_world, rays_local, cam_loc, fallback_quat, lock_roll=False, weights=None, prefer_center=False):
    if not points_world or not rays_local or len(points_world) != len(rays_local):
        return fallback_quat.copy()
    if weights is None:
        weights = [1.0] * len(points_world)

    desired_dirs = []
    observed_rays = []
    valid_weights = []
    for point_world, ray_local, weight in zip(points_world, rays_local, weights):
        if weight <= 1e-9 or ray_local.length_squared <= 1e-9:
            continue
        view_vec = point_world - cam_loc
        if view_vec.length_squared <= 1e-9:
            continue
        desired_dirs.append(view_vec.normalized())
        observed_rays.append(ray_local.normalized())
        valid_weights.append(weight)

    if len(desired_dirs) < 2:
        return fallback_quat.copy()
    solved_quat = solve_weighted_kabsch_rotation(desired_dirs, observed_rays, False, valid_weights)
    if prefer_center:
        solved_quat = refine_rotation_center_alignment(solved_quat, desired_dirs, observed_rays, valid_weights)
    return solved_quat

def solve_track_rotation_from_follow_points(track_names, fixed_world_points, current_world_points, cam_loc, ray_origin_loc, ray_origin_quat, fallback_quat, lock_roll=False, prefer_center=False):
    points_world = []
    rays_local = []
    for track_name in track_names:
        point_world = fixed_world_points.get(track_name)
        current_point = current_world_points.get(track_name)
        if point_world is None or current_point is None:
            continue
        ray_world = current_point - ray_origin_loc
        if ray_world.length_squared <= 1e-9:
            continue
        points_world.append(point_world)
        rays_local.append((ray_origin_quat.inverted() @ ray_world).normalized())

    if len(points_world) < 2:
        return None
    solved_quat = solve_rotation_quat_at_location(points_world, rays_local, cam_loc, fallback_quat, lock_roll, prefer_center=prefer_center)
    if not lock_roll:
        solved_quat = stabilize_camera_roll_step(solved_quat, fallback_quat)
    return solved_quat

def build_triangle_basis(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    if v1.length_squared < 1e-9 or v2.length_squared < 1e-9 or v1.cross(v2).length_squared < 1e-9:
        return None
    z_axis = v1.cross(v2).normalized()
    x_axis = v1.normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()

def camera_axis_plane_anchor(points, cam_loc, cam_quat):
    if len(points) < 3:
        return None
    normal = (points[1] - points[0]).cross(points[2] - points[0])
    if normal.length_squared < 1e-9:
        return None
    forward = cam_quat @ Vector((0.0, 0.0, -1.0))
    denom = normal.dot(forward)
    if abs(denom) < 1e-9:
        return None
    distance = normal.dot(points[0] - cam_loc) / denom
    if distance <= 1e-6:
        return None
    return cam_loc + forward * distance

def apply_z_lock(ideal_loc, ideal_rot_mat, target_point, initial_z):
    locked_loc = ideal_loc.copy()
    locked_loc.z = initial_z
    ideal_rot = ideal_rot_mat.to_quaternion()
    vec_ideal = (target_point - ideal_loc).normalized()
    vec_locked = (target_point - locked_loc).normalized()
    if vec_ideal.length_squared > 1e-6 and vec_locked.length_squared > 1e-6:
        final_rot = (vec_ideal.rotation_difference(vec_locked) @ ideal_rot).to_matrix().to_4x4()
        return locked_loc, final_rot
    return locked_loc, ideal_rot_mat

def adjust_location_depth_along_camera_axis(camera_matrix, object_loc, scale_ratio):
    if scale_ratio is None or scale_ratio <= 1e-6:
        return object_loc.copy()
    cam_loc = camera_matrix.translation
    view_dir = camera_matrix.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    if view_dir.length_squared <= 1e-12:
        return object_loc.copy()
    view_dir.normalize()
    depth = (object_loc - cam_loc).dot(view_dir)
    if abs(depth) <= 1e-12:
        return object_loc.copy()
    target_depth = depth / scale_ratio
    return object_loc + view_dir * (target_depth - depth)

def object_location_from_local_anchor(anchor_world, anchor_local, rotation_quat, scale):
    scaled_local = Vector((
        anchor_local.x * scale.x,
        anchor_local.y * scale.y,
        anchor_local.z * scale.z,
    ))
    return anchor_world - (rotation_quat.to_matrix() @ scaled_local)

# --- Lightweight Smoothing Helpers ---

def savitzky_golay_filter(data_dict):
    if len(data_dict) < 5: return data_dict
    frames = sorted(data_dict.keys())
    smoothed = {}
    coeffs = [-3, 12, 17, 12, -3]
    norm = 35.0
    
    first_val = list(data_dict.values())[0]
    empty_vec = Vector((0.0, 0.0)) if len(first_val) == 2 else Vector((0.0, 0.0, 0.0))

    for i in range(len(frames)):
        f = frames[i]
        if i < 2 or i >= len(frames) - 2:
            smoothed[f] = data_dict[f]
        else:
            pts = [data_dict[frames[i+j]] for j in range(-2, 3)]
            smoothed[f] = sum((c * p for c, p in zip(coeffs, pts)), empty_vec) / norm
    return smoothed

