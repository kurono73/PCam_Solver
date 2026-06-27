# pcam_solver
import bpy
from mathutils import Vector, Euler, Matrix, Quaternion
from bpy_extras import anim_utils
import math
import gpu
from gpu_extras.batch import batch_for_shader
import itertools

_handle_3d = None

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
        if props.tripod_mode and props.scale_mode == 'NONE':
            return False
        if not props.tripod_mode and props.scale_mode == 'NONE':
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
    res_x = scene.render.resolution_x * (scene.render.resolution_percentage / 100.0)
    res_y = scene.render.resolution_y * (scene.render.resolution_percentage / 100.0)
    if sensor_fit == 'VERTICAL' or (sensor_fit == 'AUTO' and res_x < res_y):
        sy = cam_data.sensor_height
        sx = sy * (res_x / res_y)
    else:
        sx = cam_data.sensor_width
        sy = sx * (res_y / res_x)
    f_safe = max(lens_value, 1e-6)
    return (sx / 2.0) / f_safe, (sy / 2.0) / f_safe

def marker_to_camera_ray(marker_co, tan_x, tan_y):
    return Vector(((2.0 * marker_co.x - 1.0) * tan_x, (2.0 * marker_co.y - 1.0) * tan_y, -1.0)).normalized()

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

def median_value(values):
    if not values:
        return None
    values_sorted = sorted(values)
    return values_sorted[len(values_sorted) // 2]

def robust_filter_values(values, z_scale=3.5):
    if len(values) < 5:
        return values
    med = median_value(values)
    deviations = [abs(v - med) for v in values]
    mad = median_value(deviations)
    if mad is None or mad < 1e-9:
        return values
    sigma = 1.4826 * mad
    filtered = [v for v in values if abs(v - med) <= z_scale * sigma]
    return filtered if filtered else values

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

def solve_focal_tripod_lock_roll_from_markers(context, cam_data, clip, tracking_object_idx, track_names, ref_frame, frame, ref_lens, frame_lens):
    tan_ref_x, tan_ref_y = get_camera_tan(cam_data, ref_lens, context.scene)
    tan_frame_x, tan_frame_y = get_camera_tan(cam_data, frame_lens, context.scene)
    ray_ref_list = []
    ray_curr_list = []

    for track_name in track_names:
        marker_ref = get_track_marker_co(clip, tracking_object_idx, track_name, ref_frame)
        marker_curr = get_track_marker_co(clip, tracking_object_idx, track_name, frame)
        if marker_ref is None or marker_curr is None:
            continue
        ray_ref_list.append(marker_to_camera_ray(marker_ref, tan_ref_x, tan_ref_y))
        ray_curr_list.append(marker_to_camera_ray(marker_curr, tan_frame_x, tan_frame_y))

    if len(ray_ref_list) < 1:
        return None
    return solve_tripod_rotation_from_rays(ray_ref_list, ray_curr_list, True)

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
    if not cam or not depth_obj:
        return None

    depsgraph = context.evaluated_depsgraph_get()
    cam_eval = cam.evaluated_get(depsgraph)
    obj_eval = depth_obj.evaluated_get(depsgraph)
    cam_mat = cam_eval.matrix_world
    origin = cam_mat.translation
    tan_x, tan_y = get_camera_tan(cam_eval.data, cam_eval.data.lens, context.scene)
    v_cam = marker_to_camera_ray(marker_co, tan_x, tan_y)
    v_world = cam_mat.to_3x3() @ v_cam

    mat_inv = obj_eval.matrix_world.inverted()
    ray_origin = mat_inv @ origin
    dir_loc = (mat_inv.to_3x3() @ v_world).normalized()
    success, loc, normal, face_index = obj_eval.ray_cast(ray_origin, dir_loc)
    if success:
        return obj_eval.matrix_world @ loc
    return None

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

def solve_single_track_rotation_from_follow_point(fixed_world_point, current_world_point, cam_loc, ray_origin_loc, ray_origin_quat, fallback_quat):
    if fixed_world_point is None or current_world_point is None:
        return fallback_quat.copy()
    desired_dir = fixed_world_point - cam_loc
    observed_world = current_world_point - ray_origin_loc
    if desired_dir.length_squared <= 1e-9 or observed_world.length_squared <= 1e-9:
        return fallback_quat.copy()
    observed_local = ray_origin_quat.inverted() @ observed_world.normalized()
    current_dir = fallback_quat @ observed_local
    if current_dir.length_squared <= 1e-9:
        return fallback_quat.copy()
    correction = current_dir.normalized().rotation_difference(desired_dir.normalized())
    return correction @ fallback_quat

def build_triangle_basis(points):
    v1 = points[1] - points[0]
    v2 = points[2] - points[0]
    if v1.length_squared < 1e-9 or v2.length_squared < 1e-9 or v1.cross(v2).length_squared < 1e-9:
        return None
    z_axis = v1.cross(v2).normalized()
    x_axis = v1.normalized()
    y_axis = z_axis.cross(x_axis).normalized()
    return Matrix((x_axis, y_axis, z_axis)).transposed()

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

# --- GPU Preview Callback ---

def draw_trackers_callback():
    context = bpy.context
    props = getattr(context.scene, "pcam_solve_props", None)
    if not props or not props.track_preview:
        return
    
    cam = context.scene.camera
    clip = props.target_clip
    if not cam or not clip:
        return
    
    try:
        ob = clip.tracking.objects[int(props.tracking_object_idx)]
    except Exception:
        return
    
    frame = context.scene.frame_current
    f_clip = frame - clip.frame_start + 1 - clip.frame_offset
    points_hit, points_miss, lines = [], [], []
    depth_obj = props.clip_depth_object
    depsgraph = context.evaluated_depsgraph_get()
    
    cam_eval = cam.evaluated_get(depsgraph)
    cam_mat = cam_eval.matrix_world
    origin = cam_mat.translation
    tan_x, tan_y = get_camera_tan(cam_eval.data, cam_eval.data.lens, context.scene)
    
    obj_eval = depth_obj.evaluated_get(depsgraph) if depth_obj else None
    if obj_eval:
        mat_inv = obj_eval.matrix_world.inverted()
        ray_origin = mat_inv @ origin
    else:
        mat_inv = None
        ray_origin = None

    selected_names = []
    if props.mode == 'ONE_POINT':
        selected_names = [props.track_1]
    elif props.mode == 'TWO_POINT':
        selected_names = [props.track_1, props.track_2]
    elif props.mode == 'THREE_POINT':
        selected_names = [props.track_1, props.track_2, props.track_3]

    for t in ob.tracks:
        if props.mode != 'CLIP_TRACK' and t.name not in selected_names:
            continue
            
        m = t.markers.find_frame(f_clip)
        if m and not getattr(m, 'mute', False):
            marker_co = get_track_display_co(t, m)
            v_cam = marker_to_camera_ray(marker_co, tan_x, tan_y)
            v_world = cam_mat.to_3x3() @ v_cam
            hit_loc = None
            if obj_eval:
                dir_loc = (mat_inv.to_3x3() @ v_world).normalized()
                success, loc, normal, face_index = obj_eval.ray_cast(ray_origin, dir_loc)
                if success:
                    hit_loc = obj_eval.matrix_world @ loc
                    points_hit.append(hit_loc)
            if hit_loc is None:
                hit_loc = origin + v_world * 5.0
                points_miss.append(hit_loc)
            lines.extend([origin, hit_loc])
            
    if not lines:
        return
    
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    try:
        gpu.state.point_size_set(props.preview_point_size)
    except Exception:
        pass
    
    if lines:
        batch = batch_for_shader(shader, 'LINES', {"pos": lines})
        shader.bind()
        shader.uniform_float("color", props.preview_color_line)
        batch.draw(shader)
    if points_hit:
        batch = batch_for_shader(shader, 'POINTS', {"pos": points_hit})
        shader.bind()
        shader.uniform_float("color", props.preview_color_hit)
        batch.draw(shader)
    if points_miss:
        batch = batch_for_shader(shader, 'POINTS', {"pos": points_miss})
        shader.bind()
        shader.uniform_float("color", props.preview_color_miss)
        batch.draw(shader)
    gpu.state.blend_set('NONE')

def update_track_preview(self, context):
    global _handle_3d
    if self.track_preview:
        if _handle_3d is None:
            _handle_3d = bpy.types.SpaceView3D.draw_handler_add(draw_trackers_callback, (), 'WINDOW', 'POST_VIEW')
    else:
        if _handle_3d:
            try:
                bpy.types.SpaceView3D.draw_handler_remove(_handle_3d, 'WINDOW')
            except Exception:
                pass
            _handle_3d = None

def update_custom_range_preview(self, context):
    scene = context.scene
    if not self.use_custom_range or not self.custom_range_use_preview:
        scene.use_preview_range = False
        return
    scene.use_preview_range = True
    scene.frame_preview_start = min(self.bake_start, self.bake_end)
    scene.frame_preview_end = max(self.bake_start, self.bake_end)

def update_existing_position_lock(self, context):
    if (
        self.apply_to == 'CAMERA' and
        self.mode in {'TWO_POINT', 'THREE_POINT'} and
        self.scale_mode == 'FOCAL_LENGTH' and
        not self.tripod_mode and
        self.clip_use_existing_position
    ):
        self.clip_use_existing_focal = True

def update_reference_frame_lock(self, context):
    if self.use_reference_frame_lock:
        frame_start, frame_end = pcam_get_frame_range(self)
        self.reference_frame = max(frame_start, min(frame_end, context.scene.frame_current))

# --- Properties ---

class PCamSolveProperties(bpy.types.PropertyGroup):
    apply_to: bpy.props.EnumProperty(
        name="Apply To",
        description="Choose whether the solved motion is baked onto the active camera or onto a target object",
        items=[
            ('CAMERA', "Camera", "Bake the solved motion to the active scene camera"),
            ('OBJECT', "Object", "Bake the solved motion to the selected target object"),
        ],
        default='CAMERA',
    )
    mode: bpy.props.EnumProperty(
        name="Mode",
        description="Choose the solving method based on how many reference tracks or clip-wide tracks you want to use",
        items=[
            ('ONE_POINT', "1 Point Track", "Use one tracked point for simple pan and tilt motion; Object targets use Blender Follow Track and do not estimate depth scale"),
            ('TWO_POINT', "2 Point Track", "Use two tracked points for motion, scale, and roll estimation"),
            ('THREE_POINT', "3 Point Track", "Use three tracked points for a more stable 3D-style solve"),
            ('CLIP_TRACK', "Clip Track", "Use all available tracks in the selected tracking layer for a clip-wide solve"),
        ],
        default='TWO_POINT',
    )
    
    target_object: bpy.props.PointerProperty(
        name="Target Object",
        description="Object that receives the baked motion when Apply To is set to Object",
        type=bpy.types.Object,
    )
    target_clip: bpy.props.PointerProperty(
        name="Movie Clip",
        description="Movie clip that contains the tracking data used by the solver",
        type=bpy.types.MovieClip,
    )
    tracking_object_idx: bpy.props.EnumProperty(
        name="Track Layer",
        description="Tracking object or layer inside the movie clip to read tracks from",
        items=get_track_objects,
    )
    clip_depth_object: bpy.props.PointerProperty(
        name="Depth Reference",
        description="Object used to raycast tracker positions into 3D space; Object targets also use its rotation as the local depth basis",
        type=bpy.types.Object,
    )
    
    track_1: bpy.props.StringProperty(name="Track 1", description="Primary track used by 1-point, 2-point, or 3-point solving")
    track_2: bpy.props.StringProperty(name="Track 2", description="Second track used by 2-point and 3-point solving")
    track_3: bpy.props.StringProperty(name="Track 3", description="Third track used by 3-point solving")

    use_reference_frame_lock: bpy.props.BoolProperty(
        name="Lock Reference Frame",
        description="Use the stored reference frame for every bake instead of the current timeline frame",
        default=False,
        update=update_reference_frame_lock,
    )
    reference_frame: bpy.props.IntProperty(
        name="Reference",
        description="Frame used as the solve reference when Lock Reference Frame is enabled",
        default=1,
    )
    
    use_undistort: bpy.props.BoolProperty(
        name="Undistort",
        description="Use undistorted tracker positions when extracting track motion from the movie clip",
        default=False,
    )
    track_smoothing: bpy.props.BoolProperty(
        name="Track Smoothing",
        description="Apply smoothing to extracted track motion to reduce sub-pixel jitter between frames",
        default=False,
    )
    track_preview: bpy.props.BoolProperty(
        name="Preview Tracker Raycast",
        description="Draw tracker rays and raycast hit points in the 3D viewport for debugging depth references",
        default=False,
        update=update_track_preview,
    )
    
    preview_color_hit: bpy.props.FloatVectorProperty(
        name="Hit Color",
        description="Viewport color used for raycast hits on the depth reference object",
        subtype='COLOR',
        size=4,
        default=(0.1, 1.0, 0.2, 1.0),
    )
    preview_color_miss: bpy.props.FloatVectorProperty(
        name="Miss Color",
        description="Viewport color used for rays that do not hit the depth reference object",
        subtype='COLOR',
        size=4,
        default=(1.0, 0.1, 0.1, 1.0),
    )
    preview_color_line: bpy.props.FloatVectorProperty(
        name="Ray Color",
        description="Viewport color used for the preview rays drawn from the camera toward tracker positions",
        subtype='COLOR',
        size=4,
        default=(1.0, 1.0, 1.0, 0.25),
    )
    preview_point_size: bpy.props.FloatProperty(
        name="Point Size",
        description="Viewport size of preview points used for raycast hits and misses",
        default=14.0,
        min=1.0,
        max=50.0,
    )

    clip_lock_roll: bpy.props.BoolProperty(
        name="Lock Roll",
        description="Prevent roll from being solved so the result only uses pan, tilt, and optional depth motion",
        default=False,
    )
    clip_center_weight: bpy.props.BoolProperty(
        name="Center Weighting",
        description="Give more influence to tracks closer to the image center when estimating motion",
        default=True,
    )
    clip_position_smooth: bpy.props.FloatProperty(
        name="Position Smooth",
        description="Smooth the solved Clip Track position curves before rotation refine",
        default=0.35,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    clip_focal_smooth: bpy.props.FloatProperty(
        name="Focal Smooth",
        description="Smooth the solved focal length curve for Clip Track",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    clip_pan_tilt_smooth: bpy.props.FloatProperty(
        name="Pan/Tilt Smooth",
        description="Smooth only the pan and tilt portion of the solved Clip Track rotation",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    clip_roll_smooth: bpy.props.FloatProperty(
        name="Roll Smooth",
        description="Smooth only the roll portion of the solved Clip Track rotation",
        default=0.0,
        min=0.0,
        max=1.0,
        subtype='FACTOR',
    )
    clip_use_existing_position: bpy.props.BoolProperty(
        name="Use Existing Position",
        description="Reuse the existing location curve and recompute only the remaining solve channels where supported",
        default=False,
        update=update_existing_position_lock,
    )
    clip_use_existing_focal: bpy.props.BoolProperty(
        name="Use Existing Focal",
        description="Reuse the existing lens curve and recompute only the remaining solve channels where supported",
        default=False,
    )
    lock_camera_z: bpy.props.BoolProperty(
        name="Lock Height",
        description="Keep the solved camera height fixed while still solving horizontal motion and rotation",
        default=False,
    )
    tripod_mode: bpy.props.BoolProperty(
        name="Tripod Motion",
        description="For Z-Depth, solve apparent scale as depth-direction dolly motion; otherwise solve as tripod-style rotational motion",
        default=False,
        update=update_existing_position_lock,
    )
    scale_mode: bpy.props.EnumProperty(
        name="Scale Method",
        description="Choose how apparent size changes in the tracked image are interpreted",
        items=[
            ('FOCAL_LENGTH', "Focal Length", "Interpret size change as a zoom or focal length change"),
            ('Z_DEPTH', "Z-Depth", "Interpret size change as forward or backward movement in depth"),
            ('NONE', "None", "Ignore scale change and only solve the remaining motion components"),
        ],
        default='Z_DEPTH',
        update=update_existing_position_lock,
    )

    use_custom_range: bpy.props.BoolProperty(
        name="Custom Range",
        description="Bake only within a manually specified frame range instead of the clip range",
        default=False,
        update=update_custom_range_preview,
    )
    custom_range_use_preview: bpy.props.BoolProperty(
        name="Use Preview Range",
        description="Show the Custom Range on Blender's timeline by syncing it to the Preview Range",
        default=True,
        update=update_custom_range_preview,
    )
    bake_start: bpy.props.IntProperty(
        name="Start",
        description="First frame of the bake range when Custom Range is enabled",
        default=1,
        update=update_custom_range_preview,
    )
    bake_end: bpy.props.IntProperty(
        name="End",
        description="Last frame of the bake range when Custom Range is enabled",
        default=250,
        update=update_custom_range_preview,
    )

# --- Operators ---

class OBJECT_OT_set_pcam_solve_bake_start(bpy.types.Operator):
    bl_idname = "view3d.set_pcam_solve_bake_start"
    bl_label = "Set Start"
    def execute(self, context):
        props = context.scene.pcam_solve_props
        props.bake_start = context.scene.frame_current
        if props.use_custom_range:
            update_custom_range_preview(props, context)
        return {'FINISHED'}

class OBJECT_OT_set_pcam_solve_bake_end(bpy.types.Operator):
    bl_idname = "view3d.set_pcam_solve_bake_end"
    bl_label = "Set End"
    def execute(self, context):
        props = context.scene.pcam_solve_props
        props.bake_end = context.scene.frame_current
        if props.use_custom_range:
            update_custom_range_preview(props, context)
        return {'FINISHED'}

class OBJECT_OT_get_pcam_solve_selected_tracks(bpy.types.Operator):
    bl_idname = "view3d.get_pcam_solve_selected_tracks"
    bl_label = "Get Selected Tracks"
    bl_options = {'REGISTER', 'UNDO'}
    
    def execute(self, context):
        props = context.scene.pcam_solve_props
        clip = next((area.spaces.active.clip for area in context.screen.areas if area.type == 'CLIP_EDITOR' and area.spaces.active.clip), None)
        if not clip:
            self.report({'WARNING'}, "No Movie Clip found.")
            return {'CANCELLED'}
            
        props.target_clip = clip
        active_idx = clip.tracking.active_object_index
        props.tracking_object_idx = str(active_idx)
        
        sel = [t.name for t in clip.tracking.objects[active_idx].tracks if t.select]
        if sel:
            if props.mode == 'ONE_POINT':
                props.track_1 = sel[0]
            elif props.mode == 'TWO_POINT':
                props.track_1 = sel[0] if len(sel) > 0 else ""
                props.track_2 = sel[1] if len(sel) > 1 else ""
            elif props.mode == 'THREE_POINT':
                props.track_1 = sel[0] if len(sel) > 0 else ""
                props.track_2 = sel[1] if len(sel) > 1 else ""
                props.track_3 = sel[2] if len(sel) > 2 else ""
            self.report({'INFO'}, f"Loaded {len(sel)} tracks.")
            return {'FINISHED'}
            
        self.report({'WARNING'}, "No tracks selected.")
        return {'FINISHED'}

class OBJECT_OT_add_pcam_solve_depth_plane(bpy.types.Operator):
    bl_idname = "view3d.add_pcam_solve_depth_plane"
    bl_label = "Add Depth Reference Plane"
    bl_description = "Add a camera-facing plane in front of the active camera and assign it as the Depth Reference"
    bl_options = {'REGISTER', 'UNDO'}

    depth: bpy.props.FloatProperty(
        name="Depth",
        description="Distance from the active camera along its view direction; plane size follows the camera field of view",
        default=0.0,
        min=0.001,
        soft_min=0.1,
        soft_max=1000.0,
        unit='LENGTH',
    )
    has_camera_reference: bpy.props.BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    camera_reference_location: bpy.props.FloatVectorProperty(size=3, options={'HIDDEN', 'SKIP_SAVE'})
    camera_reference_rotation: bpy.props.FloatVectorProperty(size=4, options={'HIDDEN', 'SKIP_SAVE'})
    camera_reference_tan: bpy.props.FloatVectorProperty(size=2, options={'HIDDEN', 'SKIP_SAVE'})

    def get_default_depth(self, context, cam, cam_loc, view_dir):
        props = context.scene.pcam_solve_props
        if props.clip_depth_object:
            vec = props.clip_depth_object.matrix_world.translation - cam_loc
            projected = vec.dot(view_dir)
            if projected > 1e-4:
                return projected
        clip_end = getattr(cam.data, "clip_end", 1000.0)
        return min(max(10.0, getattr(cam.data, "clip_start", 0.1) * 20.0), max(10.0, clip_end * 0.1))

    def capture_camera_reference(self, context, cam):
        cam_mat = matrix_without_scale(cam.matrix_world)
        cam_quat = cam_mat.to_quaternion()
        tan_x, tan_y = get_camera_tan(cam.data, cam.data.lens, context.scene)
        self.camera_reference_location = cam_mat.translation
        self.camera_reference_rotation = (cam_quat.w, cam_quat.x, cam_quat.y, cam_quat.z)
        self.camera_reference_tan = (tan_x, tan_y)
        self.has_camera_reference = True
        return cam_mat.translation, cam_quat, tan_x, tan_y

    def get_camera_reference(self, context, cam):
        if not self.has_camera_reference:
            return self.capture_camera_reference(context, cam)
        loc = Vector(self.camera_reference_location)
        quat_values = self.camera_reference_rotation
        quat = Quaternion((quat_values[0], quat_values[1], quat_values[2], quat_values[3]))
        tan_values = self.camera_reference_tan
        return loc, quat, tan_values[0], tan_values[1]

    def invoke(self, context, event):
        cam = context.scene.camera
        if cam:
            cam_loc, cam_quat, _tan_x, _tan_y = self.capture_camera_reference(context, cam)
            view_dir = cam_quat @ Vector((0.0, 0.0, -1.0))
            self.depth = self.get_default_depth(context, cam, cam_loc, view_dir)
        return self.execute(context)

    def execute(self, context):
        props = context.scene.pcam_solve_props
        cam = context.scene.camera
        if not cam:
            self.report({'ERROR'}, "No Active Camera.")
            return {'CANCELLED'}

        cam_loc, cam_quat, tan_x, tan_y = self.get_camera_reference(context, cam)
        view_dir = cam_quat @ Vector((0.0, 0.0, -1.0))

        depth = self.depth if self.depth > 0.0 else self.get_default_depth(context, cam, cam_loc, view_dir)

        plane_size = max(1.0, 2.0 * depth * max(tan_x, tan_y) * 1.25)
        plane_loc = cam_loc + view_dir * depth
        plane_mat = Matrix.Translation(plane_loc) @ cam_quat.to_matrix().to_4x4()

        mesh = bpy.data.meshes.new("PCam_Depth_Reference_Mesh")
        mesh.from_pydata(
            [(-0.5, -0.5, 0.0), (0.5, -0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()
        plane = bpy.data.objects.new("PCam_Depth_Reference", mesh)
        context.scene.collection.objects.link(plane)
        bpy.ops.object.select_all(action='DESELECT')
        plane.select_set(True)
        context.view_layer.objects.active = plane
        plane.name = "PCam_Depth_Reference"
        plane.data.name = "PCam_Depth_Reference_Mesh"
        plane.matrix_world = plane_mat @ Matrix.Scale(plane_size, 4)
        plane.display_type = 'WIRE'
        plane.show_in_front = True

        props.clip_depth_object = plane
        self.report({'INFO'}, f"Added Depth Reference '{plane.name}'.")
        return {'FINISHED'}

class PCamAnimationIO:
    # Animation I/O helpers. These wrap Blender 4.x legacy fcurves and Blender
    # 5.x action slots/channelbags behind the same local API.
    def clear_keyframes_in_range(self, id_data, data_paths, frame_start, frame_end):
        channelbags = self._iter_action_channelbags(id_data)
        if not channelbags:
            return
        for channelbag in channelbags:
            fcurves = getattr(channelbag, "fcurves", None)
            if fcurves is None:
                continue
            for fcurve in list(fcurves):
                if fcurve.data_path not in data_paths:
                    continue
                remove_indices = [
                    i for i, key in enumerate(fcurve.keyframe_points)
                    if frame_start <= key.co.x <= frame_end
                ]
                for i in reversed(remove_indices):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[i])
                if not fcurve.keyframe_points:
                    fcurves.remove(fcurve)

    def clear_animation_channels(self, id_data, keep_paths=None):
        keep_paths = set(keep_paths or ())
        channelbags = self._iter_action_channelbags(id_data)
        if not channelbags:
            return
        for channelbag in channelbags:
            fcurves = getattr(channelbag, "fcurves", None)
            if fcurves is None:
                continue
            for fcurve in list(fcurves):
                if fcurve.data_path in keep_paths:
                    continue
                fcurves.remove(fcurve)

    def snapshot_animation_curves(self, id_data, data_paths):
        data_paths = set(data_paths or ())
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None or not data_paths:
            return []
        snapshots = []
        for fcurve in fcurves:
            if fcurve.data_path not in data_paths:
                continue
            keys = []
            for key in fcurve.keyframe_points:
                key_data = {
                    "co": (float(key.co.x), float(key.co.y)),
                    "handle_left": (float(key.handle_left.x), float(key.handle_left.y)),
                    "handle_right": (float(key.handle_right.x), float(key.handle_right.y)),
                    "interpolation": key.interpolation,
                    "handle_left_type": key.handle_left_type,
                    "handle_right_type": key.handle_right_type,
                }
                if hasattr(key, "easing"):
                    key_data["easing"] = key.easing
                if hasattr(key, "back"):
                    key_data["back"] = float(key.back)
                if hasattr(key, "amplitude"):
                    key_data["amplitude"] = float(key.amplitude)
                if hasattr(key, "period"):
                    key_data["period"] = float(key.period)
                keys.append(key_data)
            snapshots.append({
                "data_path": fcurve.data_path,
                "array_index": fcurve.array_index,
                "extrapolation": fcurve.extrapolation,
                "keys": keys,
            })
        return snapshots

    def snapshot_animation_action(self, id_data):
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return []
        return self.snapshot_animation_curves(id_data, {fcurve.data_path for fcurve in fcurves})

    def copy_animation_action(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if not action:
            return None
        return action.copy()

    def _get_action_slot(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        if not anim_data:
            return None
        slot = getattr(anim_data, "action_slot", None)
        if slot is not None:
            return slot
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        slots = getattr(action, "slots", None)
        if slots is None:
            return None
        try:
            if getattr(slots, "active", None) is not None:
                return slots.active
        except Exception:
            pass
        try:
            return slots[0] if len(slots) else None
        except Exception:
            return None

    def _iter_action_channelbags(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return []

        bags = []
        get_channelbag = getattr(anim_utils, "action_get_channelbag_for_slot", None)
        ensure_channelbag = getattr(anim_utils, "action_ensure_channelbag_for_slot", None)
        slots = getattr(action, "slots", None)
        if slots is not None:
            try:
                for slot in slots:
                    channelbag = None
                    try:
                        if get_channelbag is not None:
                            channelbag = get_channelbag(action, slot)
                        elif ensure_channelbag is not None:
                            channelbag = ensure_channelbag(action, slot)
                    except Exception:
                        channelbag = None
                    if channelbag is not None and getattr(channelbag, "fcurves", None) is not None:
                        bags.append(channelbag)
            except Exception:
                pass

        if bags:
            return bags

        slot = self._get_action_slot(id_data)
        if slot is None:
            return []
        try:
            if get_channelbag is not None:
                channelbag = get_channelbag(action, slot)
            elif ensure_channelbag is not None:
                channelbag = ensure_channelbag(action, slot)
            else:
                channelbag = None
        except Exception:
            channelbag = None
        if channelbag is None or getattr(channelbag, "fcurves", None) is None:
            return []
        return [channelbag]

    def _iter_action_fcurves(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        legacy_fcurves = getattr(action, "fcurves", None)
        if legacy_fcurves is not None:
            return legacy_fcurves
        slot = self._get_action_slot(id_data)
        if slot is None:
            return None
        try:
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        except Exception:
            return None
        return getattr(channelbag, "fcurves", None)

    def _ensure_action_fcurve(self, id_data, data_path, index=0, group_name=""):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        try:
            return action.fcurve_ensure_for_datablock(id_data, data_path, index=index, group_name=group_name)
        except TypeError:
            try:
                return action.fcurve_ensure_for_datablock(id_data, data_path, index=index)
            except Exception:
                pass
        except Exception:
            pass

        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return None
        try:
            return fcurves.ensure(data_path, index=index, group_name=group_name)
        except Exception:
            try:
                return fcurves.new(data_path, index=index, group_name=group_name)
            except TypeError:
                return fcurves.new(data_path, index=index)

    def has_camera_focal_length_keys(self, camera_obj):
        cam_data = getattr(camera_obj, "data", None)
        if cam_data is None:
            return False
        fcurves = self._iter_action_fcurves(cam_data)
        if fcurves is None:
            return False
        for fcurve in fcurves:
            if fcurve.data_path == "lens":
                return len(fcurve.keyframe_points) > 0
        return False

    def camera_lens_varies_over_range(self, context, camera_obj, frame_start, frame_end, epsilon=1e-6):
        cam_data = getattr(camera_obj, "data", None)
        if cam_data is None:
            return False
        restore_frame = context.scene.frame_current
        try:
            context.scene.frame_set(frame_start)
            base_value = float(cam_data.lens)
            for frame in range(frame_start + 1, frame_end + 1):
                context.scene.frame_set(frame)
                if abs(float(cam_data.lens) - base_value) > epsilon:
                    return True
            return False
        finally:
            context.scene.frame_set(restore_frame)

    def restore_animation_action_copy(self, id_data, action_copy):
        if action_copy is None:
            return
        anim_data = id_data.animation_data_create()
        # Preserve the full camera-data action when reusing an existing focal curve.
        # Lens-only restoration proved brittle in Blender when the action was recreated during solve.
        anim_data.action = action_copy

    def restore_animation_curves(self, id_data, snapshots):
        if not snapshots:
            return
        anim_data = id_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{id_data.name}_Action")
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return
        for snap in snapshots:
            for fcurve in list(fcurves):
                if fcurve.data_path == snap["data_path"] and fcurve.array_index == snap["array_index"]:
                    fcurves.remove(fcurve)
            fcurve = self._ensure_action_fcurve(id_data, snap["data_path"], index=snap["array_index"])
            if fcurve is None:
                continue
            fcurve.extrapolation = snap["extrapolation"]
            fcurve.keyframe_points.add(len(snap["keys"]))
            for key, key_data in zip(fcurve.keyframe_points, snap["keys"]):
                key.co = key_data["co"]
                key.handle_left = key_data["handle_left"]
                key.handle_right = key_data["handle_right"]
                key.interpolation = key_data["interpolation"]
                key.handle_left_type = key_data["handle_left_type"]
                key.handle_right_type = key_data["handle_right_type"]
                if "easing" in key_data and hasattr(key, "easing"):
                    key.easing = key_data["easing"]
                if "back" in key_data and hasattr(key, "back"):
                    key.back = key_data["back"]
                if "amplitude" in key_data and hasattr(key, "amplitude"):
                    key.amplitude = key_data["amplitude"]
                if "period" in key_data and hasattr(key, "period"):
                    key.period = key_data["period"]
            fcurve.update()

    def restore_animation_snapshot_exact(self, id_data, snapshots):
        if id_data is None:
            return
        self.clear_animation_channels(id_data)
        self.restore_animation_curves(id_data, snapshots)

    def clear_animation_safely(self, target, frame_range=None, keep_target_paths=None, keep_data_paths=None):
        keep_target_paths = set(keep_target_paths or ())
        keep_data_paths = set(keep_data_paths or ())
        if frame_range is None:
            if target.animation_data and target.animation_data.action:
                fcurves = self._iter_action_fcurves(target)
                if fcurves is not None:
                    for fcurve in list(fcurves):
                        if fcurve.data_path in keep_target_paths:
                            continue
                        fcurves.remove(fcurve)
            if getattr(target, "data", None) and getattr(target.data, "animation_data", None):
                if target.data.animation_data.action:
                    fcurves = self._iter_action_fcurves(target.data)
                    if fcurves is not None:
                        for fcurve in list(fcurves):
                            if fcurve.data_path in keep_data_paths:
                                continue
                            fcurves.remove(fcurve)
            return

        if not keep_target_paths and not keep_data_paths:
            frame_start, frame_end = frame_range
            self.clear_keyframes_in_range(
                target,
                {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"},
                frame_start,
                frame_end,
            )
            if getattr(target, "data", None):
                self.clear_keyframes_in_range(target.data, {"lens"}, frame_start, frame_end)
            return

        frame_start, frame_end = frame_range
        self.clear_keyframes_in_range(
            target,
            {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"} - keep_target_paths,
            frame_start,
            frame_end,
        )
        if getattr(target, "data", None):
            self.clear_keyframes_in_range(target.data, {"lens"} - keep_data_paths, frame_start, frame_end)

    def pin_lens_constant_in_range(self, cam_data, frame_start, frame_end, lens_value, source_snapshots=None):
        if cam_data is None:
            return
        anim_data = cam_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{cam_data.name}_Action")
        fcurves = self._iter_action_fcurves(cam_data)
        if fcurves is None:
            return

        preserved_keys = []
        if source_snapshots:
            for snap in source_snapshots:
                if snap.get("data_path") != "lens":
                    continue
                for key_data in snap.get("keys", []):
                    frame = float(key_data["co"][0])
                    if frame_start <= frame <= frame_end:
                        continue
                    preserved_keys.append({
                        "frame": frame,
                        "value": float(key_data["co"][1]),
                        "interpolation": key_data.get("interpolation", 'BEZIER'),
                        "handle_left_type": key_data.get("handle_left_type", 'AUTO'),
                        "handle_right_type": key_data.get("handle_right_type", 'AUTO'),
                    })
        for fcurve in list(fcurves):
            if fcurve.data_path == "lens":
                fcurves.remove(fcurve)

        lens_fcurve = self._ensure_action_fcurve(cam_data, "lens")
        if lens_fcurve is None:
            return
        for key in list(lens_fcurve.keyframe_points):
            lens_fcurve.keyframe_points.remove(key)

        rebuilt_keys = preserved_keys + [
            {
                "frame": float(frame),
                "value": float(lens_value),
                "interpolation": 'CONSTANT',
                "handle_left_type": 'VECTOR',
                "handle_right_type": 'VECTOR',
            }
            for frame in range(frame_start, frame_end + 1)
        ]
        rebuilt_keys.sort(key=lambda item: item["frame"])

        for key_data in rebuilt_keys:
            cam_data.lens = key_data["value"]
            lens_fcurve.keyframe_points.add(1)
            key = lens_fcurve.keyframe_points[-1]
            key.co = (key_data["frame"], key_data["value"])
            key.interpolation = key_data["interpolation"]
            key.handle_left_type = key_data["handle_left_type"]
            key.handle_right_type = key_data["handle_right_type"]
        lens_fcurve.update()
        cam_data.lens = float(lens_value)


class PCamClipTrackSolver:
    # Clip Track solvers.
    #
    # The refined path below is the current camera solve path. Object targets use
    # the fallback path in execute_clip_track().
    # Current camera Clip Track path. It solves position/focal first, smooths those
    # curves, then refits rotation from fixed depth-reference points.
    def execute_clip_track_refined(self, context, target, clip, tracks, cam_ref, ref_f, frame_start, frame_end, depth, norm_curve, frame_markers, ref_lens, eff_scale_mode):
        props = context.scene.pcam_solve_props
        is_obj = (props.apply_to == 'OBJECT')
        full_frames = list(range(frame_start, frame_end + 1))
        pos_smooth = props.clip_position_smooth
        focal_smooth = props.clip_focal_smooth
        pt_smooth = props.clip_pan_tilt_smooth
        roll_smooth = props.clip_roll_smooth
        frame_range = (frame_start, frame_end) if props.use_custom_range else None
        keep_existing_position = props.clip_use_existing_position and not (props.tripod_mode and eff_scale_mode == 'FOCAL_LENGTH')
        lens_owner = cam_ref
        lens_curve_snapshot = self.snapshot_animation_action(lens_owner.data) if getattr(lens_owner, "data", None) is not None else []
        has_existing_focal_keys = self.has_camera_focal_length_keys(lens_owner)
        has_focal_variation_in_range = self.camera_lens_varies_over_range(context, lens_owner, frame_start, frame_end) if frame_range is not None else False
        keep_existing_focal = props.clip_use_existing_focal and eff_scale_mode == 'FOCAL_LENGTH' and has_existing_focal_keys
        suppress_focal_bake = props.clip_use_existing_focal and eff_scale_mode == 'FOCAL_LENGTH' and not has_existing_focal_keys
        pin_existing_focal_range = frame_range is not None and eff_scale_mode != 'FOCAL_LENGTH' and not keep_existing_focal and (has_existing_focal_keys or has_focal_variation_in_range)

        restore_frame = context.scene.frame_current
        context.scene.frame_set(ref_f)
        init_t_mat = target.matrix_world.copy()
        if not is_obj:
            init_t_mat = matrix_without_scale(init_t_mat)
        init_t_loc = init_t_mat.translation.copy()
        init_t_quat = self.get_target_rotation_quaternion(target)
        init_t_rot3 = init_t_quat.to_matrix()
        init_t_inv = init_t_mat.inverted()
        tan_ref_x, tan_ref_y = get_camera_tan(cam_ref.data, ref_lens, context.scene)
        aspect = tan_ref_x / max(tan_ref_y, 1e-6)

        existing_loc_curve = None
        existing_lens_curve = None
        location_curve_snapshot = self.snapshot_animation_curves(target, {"location"}) if keep_existing_position else []
        lens_action_copy = self.copy_animation_action(lens_owner.data) if keep_existing_focal and getattr(lens_owner, "data", None) else None
        if keep_existing_position:
            existing_loc_curve = {}
            for frame in full_frames:
                context.scene.frame_set(frame)
                existing_loc_curve[frame] = target.matrix_world.translation.copy()
        if keep_existing_focal:
            existing_lens_curve = {}
            for frame in full_frames:
                context.scene.frame_set(frame)
                existing_lens_curve[frame] = float(target.data.lens)
        context.scene.frame_set(ref_f)

        if frame_range is None:
            self.clear_animation_channels(target, {"location"} if keep_existing_position else set())
            if not keep_existing_focal and getattr(lens_owner, "data", None):
                self.clear_animation_channels(lens_owner.data)
        else:
            self.clear_keyframes_in_range(
                target,
                {"rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"} | (set() if keep_existing_position else {"location"}),
                frame_start,
                frame_end,
            )
            if getattr(lens_owner, "data", None) and not keep_existing_focal:
                self.clear_keyframes_in_range(lens_owner.data, {"lens"}, frame_start, frame_end)
        if pin_existing_focal_range and getattr(lens_owner, "data", None):
            self.pin_lens_constant_in_range(lens_owner.data, frame_start, frame_end, ref_lens, lens_curve_snapshot)

        if keep_existing_focal and existing_lens_curve is not None:
            lens_curve = existing_lens_curve.copy()
        elif eff_scale_mode == 'FOCAL_LENGTH':
            lens_curve = {f: ref_lens * norm_curve.get(f, 1.0) for f in full_frames}
            if focal_smooth > 1e-4:
                lens_curve = stabilize_scalar_curve(lens_curve, full_frames, None, max_blend=0.04 + 0.20 * focal_smooth)
                lens_curve = smooth_scalar_curve_global(lens_curve, full_frames, strength=0.10 + 0.90 * focal_smooth, passes=1 + int(round(3 * focal_smooth)))
        else:
            lens_curve = {f: ref_lens for f in full_frames}

        ref_markers = frame_markers.get(ref_f, {})
        if props.clip_depth_object:
            fixed_world_points = {}
            for track in tracks:
                marker_co = ref_markers.get(track.name)
                if marker_co is None:
                    continue
                hit = raycast_marker_world(context, cam_ref, props.clip_depth_object, marker_co)
                if hit is not None:
                    fixed_world_points[track.name] = hit
        else:
            fixed_world_points = {}
            planar_depth = max(depth, 1e-4)
            for name, marker_co in ref_markers.items():
                point_local = Vector((
                    (2.0 * marker_co.x - 1.0) * planar_depth * tan_ref_x,
                    (2.0 * marker_co.y - 1.0) * planar_depth * tan_ref_y,
                    -planar_depth,
                ))
                fixed_world_points[name] = init_t_loc + (init_t_rot3 @ point_local)

        if eff_scale_mode == 'Z_DEPTH' and not fixed_world_points:
            context.scene.frame_set(restore_frame)
            self.report({'ERROR'}, "Z-Depth needs valid reference points or depth object.")
            return {'CANCELLED'}

        frame_sets = {ref_f: set(ref_markers.keys())}
        dx_raw = {ref_f: 0.0}
        dy_raw = {ref_f: 0.0}
        pan_raw = {ref_f: Vector((0.0, 0.0, 0.0))}
        depth_raw = {ref_f: 0.0}

        if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
            track_world_data = {}
            for track in tracks:
                world_data = self.extract_track_data(context, cam_ref, clip, track.name, props.clip_depth_object, True, props.track_smoothing)
                if world_data and ref_f in world_data:
                    track_world_data[track.name] = world_data
            if not track_world_data:
                context.scene.frame_set(restore_frame)
                self.report({'ERROR'}, "No depth track data extracted.")
                return {'CANCELLED'}

            ref_depth_samples = []
            ref_depth_weights = []
            for track in tracks:
                point_world = fixed_world_points.get(track.name)
                marker_co = ref_markers.get(track.name)
                if point_world is None or marker_co is None:
                    continue
                point_local = init_t_inv @ point_world
                ref_depth_samples.append(max(1e-4, -point_local.z))
                ref_depth_weights.append(marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0)
            ref_depth_dist = (
                sum(d * w for d, w in zip(ref_depth_samples, ref_depth_weights)) / max(sum(ref_depth_weights), 1e-6)
                if ref_depth_samples else max(depth, 1e-4)
            )

            for frame in full_frames:
                if frame == ref_f:
                    continue
                scale = norm_curve.get(frame, 1.0)
                if scale <= 1e-6:
                    continue
                points_ref = []
                points_curr = []
                weights = []
                names = []
                for track in tracks:
                    world_data = track_world_data.get(track.name)
                    if world_data is None or frame not in world_data:
                        continue
                    p_ref = init_t_inv @ world_data[ref_f]
                    p_cur = init_t_inv @ world_data[frame]
                    p_cur_unzoom = Vector((p_cur.x / scale, p_cur.y / scale, p_cur.z))
                    points_ref.append(p_ref)
                    points_curr.append(p_cur_unzoom)
                    names.append(track.name)
                    marker_co = frame_markers.get(frame, {}).get(track.name)
                    weights.append(marker_center_weight(marker_co, aspect) if props.clip_center_weight and marker_co is not None else 1.0)
                if len(points_ref) < 2:
                    continue
                frame_sets[frame] = set(names)
                sum_w = max(sum(weights), 1e-6)
                c_ref = sum((p * w for p, w in zip(points_ref, weights)), Vector((0.0, 0.0, 0.0))) / sum_w
                c_cur = sum((p * w for p, w in zip(points_curr, weights)), Vector((0.0, 0.0, 0.0))) / sum_w
                pan_raw[frame] = Vector((c_cur.x - c_ref.x, c_cur.y - c_ref.y, 0.0))
                depth_raw[frame] = ref_depth_dist * (1.0 - (1.0 / scale))
        else:
            for frame in full_frames:
                if frame == ref_f:
                    continue
                markers_curr = frame_markers.get(frame, {})
                shared = list(set(ref_markers.keys()) & set(markers_curr.keys()))
                if len(shared) < 2:
                    continue
                frame_sets[frame] = set(shared)
                scale = norm_curve.get(frame, 1.0) if eff_scale_mode == 'FOCAL_LENGTH' else 1.0
                center_uv = Vector((0.5, 0.5))
                curr_unscaled = {
                    name: center_uv + (markers_curr[name] - center_uv) / max(scale, 1e-6)
                    for name in shared
                }
                c_ref = weighted_marker_centroid(ref_markers, shared, aspect, props.clip_center_weight)
                c_curr = weighted_marker_centroid(curr_unscaled, shared, aspect, props.clip_center_weight)
                if c_ref is None or c_curr is None:
                    continue
                dx_raw[frame] = -(c_curr.x - c_ref.x) * (2.0 * depth * tan_ref_x)
                dy_raw[frame] = -(c_curr.y - c_ref.y) * (2.0 * depth * tan_ref_y)

        for frame in full_frames:
            frame_sets.setdefault(frame, frame_sets.get(frame - 1, frame_sets.get(frame + 1, set())))

        transition = {}
        for i, frame in enumerate(full_frames):
            curr_set = frame_sets.get(frame, set())
            prev_set = frame_sets.get(full_frames[i - 1], curr_set) if i > 0 else curr_set
            next_set = frame_sets.get(full_frames[i + 1], curr_set) if i < len(full_frames) - 1 else curr_set
            union_a = len(curr_set | prev_set)
            union_b = len(curr_set | next_set)
            coh_a = len(curr_set & prev_set) / max(1, union_a)
            coh_b = len(curr_set & next_set) / max(1, union_b)
            transition[frame] = max(0.0, 1.0 - min(coh_a, coh_b))
        expanded = expand_transition_blends(transition, full_frames, radius=2, decay=0.7)

        if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
            pan_curve = stabilize_vector_curve(pan_raw, full_frames, expanded, max_blend=0.08 + 0.28 * pos_smooth)
            pan_curve = bridge_vector_curve(pan_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.36 + 0.28 * pos_smooth)
            depth_curve = stabilize_scalar_curve(depth_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            depth_curve = bridge_scalar_curve(depth_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.34 + 0.26 * pos_smooth)
            pan_curve = smooth_vector_curve_global(pan_curve, full_frames, strength=0.10 + 0.90 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
            depth_curve = smooth_scalar_curve_global(depth_curve, full_frames, strength=0.08 + 0.82 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
        else:
            dx_curve = stabilize_scalar_curve(dx_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            dy_curve = stabilize_scalar_curve(dy_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            dx_curve = bridge_scalar_curve(dx_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.34 + 0.28 * pos_smooth)
            dy_curve = bridge_scalar_curve(dy_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.34 + 0.28 * pos_smooth)
            dx_curve = smooth_scalar_curve_global(dx_curve, full_frames, strength=0.08 + 0.82 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
            dy_curve = smooth_scalar_curve_global(dy_curve, full_frames, strength=0.08 + 0.82 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))

        def get_loc_for_frame(frame):
            if props.tripod_mode:
                if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
                    return init_t_loc + (init_t_quat @ Vector((0.0, 0.0, -1.0))) * depth_curve.get(frame, 0.0)
                return init_t_loc.copy()
            if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
                loc = init_t_loc - (init_t_rot3 @ pan_curve.get(frame, Vector((0.0, 0.0, 0.0))))
                loc += (init_t_quat @ Vector((0.0, 0.0, -1.0))) * depth_curve.get(frame, 0.0)
                return loc
            return init_t_loc + (init_t_rot3 @ Vector((dx_curve.get(frame, 0.0), dy_curve.get(frame, 0.0), 0.0)))

        if keep_existing_position and existing_loc_curve is not None:
            loc_curve = {frame: existing_loc_curve.get(frame, init_t_loc.copy()).copy() for frame in full_frames}
        else:
            loc_curve = {frame: get_loc_for_frame(frame) for frame in full_frames}
            if not props.tripod_mode and pos_smooth > 1e-4:
                loc_curve = stabilize_vector_curve(loc_curve, full_frames, expanded, max_blend=0.05 + 0.18 * pos_smooth)
                loc_curve = bridge_vector_curve(loc_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.22 + 0.30 * pos_smooth)
                loc_curve = smooth_vector_curve_global(loc_curve, full_frames, strength=0.10 + 0.90 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
        if props.lock_camera_z:
            loc_curve = {frame: Vector((loc.x, loc.y, init_t_loc.z)) for frame, loc in loc_curve.items()}

        def build_rotation_inputs(frame):
            marker_map = frame_markers.get(frame, {})
            if eff_scale_mode == 'Z_DEPTH':
                tan_x, tan_y = tan_ref_x, tan_ref_y
            else:
                tan_x, tan_y = get_camera_tan(cam_ref.data, lens_curve[frame], context.scene)
            stable_names = select_stable_track_names(frame, frame_sets, fixed_world_points.keys())
            if len(stable_names) < 3:
                stable_names = set(frame_sets.get(frame, set())) & set(fixed_world_points.keys())
            points_world = []
            rays_local = []
            weights = []
            for name in stable_names:
                point_world = fixed_world_points.get(name)
                marker_co = marker_map.get(name)
                if point_world is None or marker_co is None:
                    continue
                stability_w = track_stability_weight(frame, frame_sets, name)
                base_w = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0
                points_world.append(point_world)
                rays_local.append(marker_to_camera_ray(marker_co, tan_x, tan_y))
                weights.append(base_w * stability_w)
            return points_world, rays_local, weights, len(stable_names), sum(weights) / max(1, len(weights))

        rot_quats = {ref_f: init_t_quat.copy()}
        for frame in range(ref_f + 1, frame_end + 1):
            points_world, rays_local, weights, stable_count, avg_weight = build_rotation_inputs(frame)
            prev_quat = rot_quats.get(frame - 1, init_t_quat)
            if len(points_world) < 2:
                rot_quats[frame] = prev_quat.copy()
                continue
            raw_quat = solve_rotation_quat_at_location(points_world, rays_local, loc_curve.get(frame, init_t_loc), prev_quat, False, weights)
            stability = min(1.0, max(0.28, (stable_count / 5.0) * avg_weight))
            blend = (0.22 + 0.58 * (1.0 - expanded.get(frame, 0.0))) * stability
            rot_quats[frame] = prev_quat.slerp(raw_quat, min(1.0, max(0.0, blend)))
        for frame in range(ref_f - 1, frame_start - 1, -1):
            points_world, rays_local, weights, stable_count, avg_weight = build_rotation_inputs(frame)
            next_quat = rot_quats.get(frame + 1, init_t_quat)
            if len(points_world) < 2:
                rot_quats[frame] = next_quat.copy()
                continue
            raw_quat = solve_rotation_quat_at_location(points_world, rays_local, loc_curve.get(frame, init_t_loc), next_quat, False, weights)
            stability = min(1.0, max(0.28, (stable_count / 5.0) * avg_weight))
            blend = (0.22 + 0.58 * (1.0 - expanded.get(frame, 0.0))) * stability
            rot_quats[frame] = next_quat.slerp(raw_quat, min(1.0, max(0.0, blend)))

        rot_quats = smooth_quaternion_curve(rot_quats, full_frames, expanded, max_blend=0.34)
        rot_quats = bridge_quaternion_curve(rot_quats, full_frames, expanded, threshold=0.24, max_bridge_blend=0.82)
        if pt_smooth > 1e-4 or roll_smooth > 1e-4:
            view_axis = Vector((0.0, 0.0, -1.0))
            pan_tilt_quats = {}
            roll_raw = {}
            for frame in full_frames:
                base_quat = rot_quats.get(frame, init_t_quat.copy())
                roll_angle = signed_twist_angle(base_quat, view_axis)
                pan_tilt_quats[frame] = replace_quaternion_twist(base_quat, view_axis, 0.0)
                roll_raw[frame] = roll_angle
            if pt_smooth > 1e-4:
                pan_tilt_quats = smooth_quaternion_curve(pan_tilt_quats, full_frames, expanded, max_blend=0.30 * pt_smooth)
                pan_tilt_quats = bridge_quaternion_curve(pan_tilt_quats, full_frames, expanded, threshold=0.24, max_bridge_blend=0.52 * pt_smooth)
                pan_tilt_quats = smooth_quaternion_curve_global(pan_tilt_quats, full_frames, strength=0.10 + 0.90 * pt_smooth, passes=1 + int(round(3 * pt_smooth)))
            if roll_smooth > 1e-4:
                roll_raw = stabilize_roll_curve(roll_raw, full_frames, despike_threshold_deg=0.8 + 1.6 * (1.0 - roll_smooth), smooth_blend=0.26 * roll_smooth)
                roll_raw = bridge_scalar_curve(roll_raw, full_frames, expanded, threshold=0.24, max_bridge_blend=0.40 * roll_smooth)
                roll_raw = smooth_scalar_curve_global(roll_raw, full_frames, strength=0.10 + 0.90 * roll_smooth, passes=1 + int(round(3 * roll_smooth)))
            for frame in full_frames:
                rot_quats[frame] = replace_quaternion_twist(
                    pan_tilt_quats.get(frame, rot_quats.get(frame, init_t_quat.copy())),
                    view_axis,
                    roll_raw.get(frame, 0.0),
                )

        for frame in full_frames:
            context.scene.frame_set(frame)
            self.set_target_rotation(target, rot_quats.get(frame, init_t_quat.copy()))
            self.keyframe_target_rotation(target, frame)
            if not keep_existing_position:
                target.location = loc_curve.get(frame, init_t_loc.copy())
                target.keyframe_insert(data_path="location", frame=frame)
            if eff_scale_mode == 'FOCAL_LENGTH' and not keep_existing_focal and not suppress_focal_bake:
                target.data.lens = lens_curve[frame]
                target.data.keyframe_insert(data_path="lens", frame=frame)

        if keep_existing_position:
            self.restore_animation_curves(target, location_curve_snapshot)
        if keep_existing_focal and getattr(lens_owner, "data", None):
            self.restore_animation_action_copy(lens_owner.data, lens_action_copy)
        elif pin_existing_focal_range and getattr(lens_owner, "data", None):
            self.pin_lens_constant_in_range(lens_owner.data, frame_start, frame_end, ref_lens, lens_curve_snapshot)

        context.scene.frame_set(ref_f)
        self.report({'INFO'}, f"Applied Clip Track motion to '{target.name}'.")
        return {'FINISHED'}

    def execute_clip_track_object_refined(self, context, target, clip, tracks, cam_ref, ref_f, frame_start, frame_end, frame_markers):
        props = context.scene.pcam_solve_props
        depth_obj = props.clip_depth_object
        if not depth_obj:
            self.report({'ERROR'}, "Depth Reference is required.")
            return {'CANCELLED'}

        restore_frame = context.scene.frame_current
        context.scene.frame_set(ref_f)
        context.view_layer.update()

        init_t_mat = target.matrix_world.copy()
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        init_t_scale = target.scale.copy()
        target_curve_snapshot = self.snapshot_animation_action(target)

        ref_markers = frame_markers.get(ref_f, {})
        if not ref_markers:
            self.report({'ERROR'}, "Reference Frame has no visible Clip Track markers.")
            return {'CANCELLED'}

        ref_depth_mat = evaluated_matrix_world(context, depth_obj)
        ref_depth_inv = ref_depth_mat.inverted()
        ref_depth_quat_inv = ref_depth_mat.to_quaternion().inverted()
        tan_ref_x, tan_ref_y = get_camera_tan(cam_ref.data, cam_ref.data.lens, context.scene)
        aspect = tan_ref_x / tan_ref_y if tan_ref_y > 1e-6 else 1.0

        ref_world_points = {}
        ref_local_points = {}
        ref_weights = {}
        for track in tracks:
            marker_co = ref_markers.get(track.name)
            if marker_co is None:
                continue
            hit = raycast_marker_world(context, cam_ref, depth_obj, marker_co)
            if hit is None:
                continue
            ref_world_points[track.name] = hit
            ref_local_points[track.name] = ref_depth_inv @ hit
            ref_weights[track.name] = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0

        if len(ref_local_points) < 2:
            self.report({'ERROR'}, "Clip Track Object needs at least two valid reference hits.")
            return {'CANCELLED'}

        self.clear_animation_safely(target, (frame_start, frame_end) if props.use_custom_range else None)

        prev_obj_quat = init_t_rot.copy()
        prev_obj_euler = init_t_rot.to_euler(target.rotation_mode) if target.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'} else None
        baked_frames = 0

        for frame in range(frame_start, frame_end + 1):
            context.scene.frame_set(frame)
            context.view_layer.update()
            marker_map = frame_markers.get(frame, {})
            curr_depth_mat = evaluated_matrix_world(context, depth_obj)
            curr_depth_inv = curr_depth_mat.inverted()
            cam_mat_curr = evaluated_matrix_world(context, cam_ref)

            names = []
            curr_world_points = []
            ref_worlds = []
            curr_locals = []
            ref_locals = []
            weights = []

            for track in tracks:
                name = track.name
                if name not in ref_local_points:
                    continue
                marker_co = marker_map.get(name)
                if marker_co is None:
                    continue
                hit = raycast_marker_world(context, cam_ref, depth_obj, marker_co)
                if hit is None:
                    continue
                weight = marker_center_weight(marker_co, aspect) if props.clip_center_weight else ref_weights.get(name, 1.0)
                names.append(name)
                curr_world_points.append(hit)
                ref_worlds.append(ref_world_points[name])
                curr_locals.append(curr_depth_inv @ hit)
                ref_locals.append(ref_local_points[name])
                weights.append(weight)

            if len(curr_locals) < 2:
                continue

            ref_centroid_world = weighted_points_centroid(ref_worlds, weights)
            curr_centroid_world = weighted_points_centroid(curr_world_points, weights)
            target.location = init_t_loc + (curr_centroid_world - ref_centroid_world)

            scale_ratio = median_edge_scale(ref_locals, curr_locals)
            if scale_ratio is None:
                ref_dist = point_cloud_avg_distance(ref_locals)
                curr_dist = point_cloud_avg_distance(curr_locals)
                scale_ratio = curr_dist / ref_dist if ref_dist > 1e-6 else 1.0
            if scale_ratio is None or scale_ratio <= 1e-6:
                scale_ratio = 1.0

            cam_loc_curr = cam_mat_curr.translation
            vec_cam_to_obj = target.location - cam_loc_curr
            current_obj_depth = vec_cam_to_obj.length
            target_depth = current_obj_depth / scale_ratio if scale_ratio > 1e-6 else current_obj_depth
            if vec_cam_to_obj.length_squared > 1e-12:
                target.location = cam_loc_curr + vec_cam_to_obj.normalized() * target_depth

            ref_centroid_local = weighted_points_centroid(ref_locals, weights)
            curr_centroid_local = weighted_points_centroid(curr_locals, weights)
            ref_vecs = [point - ref_centroid_local for point in ref_locals]
            curr_vecs = [point - curr_centroid_local for point in curr_locals]
            curr_to_ref_quat = solve_weighted_kabsch_rotation(ref_vecs, curr_vecs, False, weights)
            local_delta_quat = curr_to_ref_quat.inverted()
            solved_quat = curr_depth_mat.to_quaternion() @ local_delta_quat @ ref_depth_quat_inv @ init_t_rot
            solved_quat = self.set_target_rotation_continuous(target, solved_quat, prev_obj_quat, prev_obj_euler)
            prev_obj_quat = solved_quat.copy()
            if prev_obj_euler is not None:
                prev_obj_euler = target.rotation_euler.copy()

            target.scale = init_t_scale
            target.keyframe_insert(data_path="location", frame=frame)
            self.keyframe_target_rotation(target, frame)
            target.keyframe_insert(data_path="scale", frame=frame)
            baked_frames += 1

        if baked_frames == 0:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            context.scene.frame_set(restore_frame)
            self.report({'ERROR'}, "No frames could be baked from Clip Track Object.")
            return {'CANCELLED'}

        context.scene.frame_set(ref_f)
        total_frames = frame_end - frame_start + 1
        suffix = f" Solved {baked_frames}/{total_frames} frames." if baked_frames < total_frames else ""
        self.report({'INFO'}, f"Applied Clip Track object motion to '{target.name}'.{suffix}")
        return {'FINISHED'}


    # Clip Track dispatcher and object fallback. Camera solves currently return
    # through execute_clip_track_refined() after marker scale analysis.
    def execute_clip_track(self, context, target):
        props = context.scene.pcam_solve_props
        clip = props.target_clip
        is_obj = (props.apply_to == 'OBJECT')
        cam_ref = context.scene.camera
        
        try:
            idx = int(props.tracking_object_idx)
            tracks = clip.tracking.objects[idx].tracks
        except Exception:
            return {'CANCELLED'}
            
        if not tracks:
            return {'CANCELLED'}

        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_f = pcam_get_reference_frame(context, props, frame_start, frame_end)
        ref_lens = cam_ref.data.lens

        context.scene.frame_set(ref_f)
        ref_lens = cam_ref.data.lens

        context.scene.frame_set(frame_start)
        
        init_t_mat = target.matrix_world.copy()
        init_c_mat = cam_ref.matrix_world.copy()
        if not is_obj:
            init_t_mat = matrix_without_scale(init_t_mat)
            init_c_mat = matrix_without_scale(init_c_mat)
        init_c_rot3 = init_c_mat.to_quaternion().to_matrix()
        init_c_rot4 = init_c_rot3.to_4x4()
        
        current_lens = cam_ref.data.lens
        current_rot_mat = init_c_rot4.copy()
        current_loc = init_c_mat.translation.copy()
        eff_scale_mode = 'Z_DEPTH' if is_obj and props.scale_mode != 'NONE' else props.scale_mode
        is_tripod = props.tripod_mode if not is_obj else False
        keep_existing_position = (not is_obj) and props.clip_use_existing_position and not (is_tripod and eff_scale_mode == 'FOCAL_LENGTH')
        keep_existing_focal = (not is_obj) and eff_scale_mode == 'FOCAL_LENGTH' and props.clip_use_existing_focal
        use_refined_solver = not is_obj
        lens_curve_snapshot = self.snapshot_animation_action(cam_ref.data) if (not is_obj and getattr(cam_ref, "data", None) is not None) else []
        pin_existing_focal_range = props.use_custom_range and not is_obj and eff_scale_mode != 'FOCAL_LENGTH' and (
            self.has_camera_focal_length_keys(cam_ref) or
            self.camera_lens_varies_over_range(context, cam_ref, frame_start, frame_end)
        )
        
        if is_obj:
            depth = max(0.1, (init_t_mat.translation - init_c_mat.translation).length)
        elif props.clip_depth_object:
            origin_depth = max(0.1, (props.clip_depth_object.matrix_world.translation - init_c_mat.translation).length)
            track_depth = self.estimate_track_group_depth(context, cam_ref, clip, tracks, ref_f, props.clip_depth_object)
            if eff_scale_mode == 'Z_DEPTH' and track_depth is not None:
                depth = max(origin_depth, track_depth)
            else:
                depth = origin_depth
        else:
            depth = 1.0

        global_focus_point = init_c_mat.translation + init_c_rot3 @ Vector((0, 0, -depth))

        frame_markers = {}
        tan_ref_x, tan_ref_y = get_camera_tan(cam_ref.data, ref_lens, context.scene)
        aspect = tan_ref_x / tan_ref_y if tan_ref_y > 1e-6 else 1.0
        def phys_dist(p1, p2):
            return math.sqrt(((p1.x - p2.x) * aspect)**2 + (p1.y - p2.y)**2)

        for f in range(frame_start, frame_end + 1):
            f_clip = f - clip.frame_start + 1 - clip.frame_offset
            markers = {}
            for t in tracks:
                marker = t.markers.find_frame(f_clip)
                if marker and not getattr(marker, 'mute', False):
                    markers[t.name] = get_track_display_co(t, marker)
            frame_markers[f] = markers

        def pair_scale_ratio(markers_a, markers_b):
            shared = list(set(markers_a.keys()) & set(markers_b.keys()))
            if len(shared) < 2:
                return None
            ratios = []
            for t1_name, t2_name in itertools.combinations(shared, 2):
                dist_a = phys_dist(markers_a[t1_name], markers_a[t2_name])
                dist_b = phys_dist(markers_b[t1_name], markers_b[t2_name])
                if dist_a > 1e-4:
                    ratios.append(dist_b / dist_a)
            if not ratios:
                return None
            ratios.sort()
            return ratios[len(ratios) // 2]

        step_scale_ratios = {}
        for f in range(frame_start + 1, frame_end + 1):
            step_scale_ratios[f] = pair_scale_ratio(frame_markers[f - 1], frame_markers[f]) or 1.0

        base_curve = {ref_f: 1.0}
        for f in range(ref_f + 1, frame_end + 1):
            base_curve[f] = base_curve[f - 1] * step_scale_ratios.get(f, 1.0)
        for f in range(ref_f - 1, frame_start - 1, -1):
            step_ratio = step_scale_ratios.get(f + 1, 1.0)
            base_curve[f] = base_curve[f + 1] / step_ratio if step_ratio > 1e-6 else base_curve[f + 1]

        ref_markers = frame_markers.get(ref_f, {})
        if props.use_reference_frame_lock and not ref_markers:
            self.report({'ERROR'}, "Reference Frame has no visible Clip Track markers.")
            return {'CANCELLED'}
        if is_obj:
            return self.execute_clip_track_object_refined(
                context,
                target,
                clip,
                tracks,
                cam_ref,
                ref_f,
                frame_start,
                frame_end,
                frame_markers,
            )
        if use_refined_solver:
            abs_scale_curve = {ref_f: 1.0}
            for f in range(frame_start, frame_end + 1):
                if f == ref_f:
                    continue
                abs_ratio = pair_scale_ratio(ref_markers, frame_markers[f])
                if abs_ratio is not None:
                    abs_scale_curve[f] = abs_ratio

            correction_keys = sorted(abs_scale_curve.keys())
            correction_curve = {}
            for f in correction_keys:
                base_value = base_curve.get(f, 1.0)
                correction_curve[f] = abs_scale_curve[f] / base_value if base_value > 1e-6 else 1.0

            norm_curve = {}
            for f in range(frame_start, frame_end + 1):
                base_value = base_curve.get(f, 1.0)
                if f in correction_curve:
                    corr = correction_curve[f]
                else:
                    prev_keys = [k for k in correction_keys if k < f]
                    next_keys = [k for k in correction_keys if k > f]
                    if prev_keys and next_keys:
                        k0 = prev_keys[-1]
                        k1 = next_keys[0]
                        t = (f - k0) / max(1, (k1 - k0))
                        c0 = correction_curve[k0]
                        c1 = correction_curve[k1]
                        corr = c0 * ((c1 / c0) ** t) if c0 > 1e-6 and c1 > 1e-6 else (1.0 - t) * c0 + t * c1
                    elif prev_keys:
                        corr = correction_curve[prev_keys[-1]]
                    elif next_keys:
                        corr = correction_curve[next_keys[0]]
                    else:
                        corr = 1.0
                norm_curve[f] = base_value * corr
        else:
            norm_curve = base_curve.copy()
        target_lens = {f: ref_lens * norm_curve[f] if eff_scale_mode == 'FOCAL_LENGTH' else ref_lens for f in norm_curve}
        if (
            not is_obj and
            use_refined_solver
        ):
            return self.execute_clip_track_refined(
                context, target, clip, tracks, cam_ref, ref_f, frame_start, frame_end,
                depth, norm_curve, frame_markers, ref_lens, eff_scale_mode
            )

        target.keyframe_insert(data_path="location", frame=frame_start)
        self.keyframe_target_rotation(target, frame_start)
        if is_obj:
            target.keyframe_insert(data_path="scale", frame=frame_start)
        elif eff_scale_mode == 'FOCAL_LENGTH':
            target.data.keyframe_insert(data_path="lens", frame=frame_start)

        traj_rot = {frame_start: current_rot_mat.copy()}
        traj_loc = {frame_start: current_loc.copy()}
        traj_lens = {frame_start: current_lens}

        self.clear_animation_safely(target, (frame_start, frame_end) if props.use_custom_range else None)
        if pin_existing_focal_range:
            self.pin_lens_constant_in_range(cam_ref.data, frame_start, frame_end, ref_lens, lens_curve_snapshot)

        for f in range(frame_start + 1, frame_end + 1):
            f_clip_prev = f - 1 - clip.frame_start + 1 - clip.frame_offset
            f_clip_curr = f - clip.frame_start + 1 - clip.frame_offset
            
            context.scene.frame_set(f)
            
            valid_pairs = []
            for t in tracks:
                m1 = t.markers.find_frame(f_clip_prev)
                m2 = t.markers.find_frame(f_clip_curr)
                if m1 and m2 and not getattr(m1, 'mute', False) and not getattr(m2, 'mute', False):
                    p1 = get_track_display_co(t, m1)
                    p2 = get_track_display_co(t, m2)
                    motion = (p2 - p1).length
                    if motion < 0.2:
                        valid_pairs.append((p1, p2))

            if len(valid_pairs) < 2:
                if is_tripod and use_refined_solver and eff_scale_mode != 'NONE':
                    cam_ref.data.lens = target_lens.get(f, ref_lens)
                    context.view_layer.update()
                    tan_x2 = math.tan(cam_ref.data.angle_x / 2.0)
                    tan_y2 = math.tan(cam_ref.data.angle_y / 2.0)

                    anchor_ref_rays = []
                    anchor_curr_rays = []
                    anchor_weights = []
                    markers_curr = frame_markers.get(f, {})
                    for track_name in set(ref_markers.keys()) & set(markers_curr.keys()):
                        p_ref = ref_markers[track_name]
                        p_curr = markers_curr[track_name]
                        if props.clip_center_weight:
                            w = marker_center_weight(p_curr, aspect)
                        else:
                            w = 1.0
                        anchor_ref_rays.append(marker_to_camera_ray(p_ref, tan_ref_x, tan_ref_y))
                        anchor_curr_rays.append(marker_to_camera_ray(p_curr, tan_x2, tan_y2))
                        anchor_weights.append(w)

                    if anchor_ref_rays:
                        if eff_scale_mode == 'FOCAL_LENGTH' and not props.clip_lock_roll:
                            anchor_quat = solve_tripod_pan_tilt_from_rays(anchor_ref_rays, anchor_curr_rays, anchor_weights)
                        else:
                            anchor_quat = solve_weighted_kabsch_rotation(
                                anchor_ref_rays,
                                anchor_curr_rays,
                                props.clip_lock_roll,
                                anchor_weights,
                            )
                            if not props.clip_lock_roll:
                                anchor_axis = sum((v * w for v, w in zip(anchor_ref_rays, anchor_weights)), Vector((0.0, 0.0, 0.0)))
                                if anchor_axis.length_squared > 1e-9:
                                    anchor_quat = enforce_roll_sign_continuity(anchor_quat, anchor_ref_rays, anchor_curr_rays, anchor_axis, anchor_weights)
                        desired_rot_mat = init_c_rot4 @ anchor_quat.to_matrix().to_4x4()
                        anchor_blend = 0.72
                        current_rot_mat = soft_reanchor_rotation(current_rot_mat, desired_rot_mat, len(anchor_ref_rays), anchor_blend)
                traj_rot[f] = current_rot_mat.copy()
                traj_loc[f] = current_loc.copy()
                traj_lens[f] = target_lens.get(f, current_lens)
                continue

            cam_ref.data.lens = target_lens[f-1]
            context.view_layer.update()
            tan_x1 = math.tan(cam_ref.data.angle_x / 2.0)
            tan_y1 = math.tan(cam_ref.data.angle_y / 2.0)

            weights = []
            for p in valid_pairs:
                if props.clip_center_weight:
                    d = math.sqrt(((p[0].x - 0.5) * aspect)**2 + (p[0].y - 0.5)**2)
                    weights.append(1.0 + 5.0 * math.exp(-10.0 * (d ** 2)))
                else:
                    weights.append(1.0)
            sum_w = sum(weights)

            c1_raw = sum((p[0]*w for p,w in zip(valid_pairs, weights)), Vector((0,0))) / sum_w
            c2_raw = sum((p[1]*w for p,w in zip(valid_pairs, weights)), Vector((0,0))) / sum_w

            v1_list = [Vector(((2.0 * p[0].x - 1.0) * tan_x1, (2.0 * p[0].y - 1.0) * tan_y1, -1.0)).normalized() for p in valid_pairs]
            
            if eff_scale_mode == 'FOCAL_LENGTH':
                cam_ref.data.lens = target_lens[f]
                context.view_layer.update()
                tan_x2 = math.tan(cam_ref.data.angle_x / 2.0)
                tan_y2 = math.tan(cam_ref.data.angle_y / 2.0)
            else:
                tan_x2, tan_y2 = tan_x1, tan_y1

            step_ratio = norm_curve[f] / norm_curve[f-1] if norm_curve[f-1] > 1e-6 else 1.0
            if eff_scale_mode == 'Z_DEPTH' and step_ratio > 1e-6:
                center_uv = Vector((0.5, 0.5))
                rot_pairs = [(p1, center_uv + (p2 - center_uv) / step_ratio) for p1, p2 in valid_pairs]
            else:
                rot_pairs = valid_pairs

            v2_list_new = [Vector(((2.0 * p[1].x - 1.0) * tan_x2, (2.0 * p[1].y - 1.0) * tan_y2, -1.0)).normalized() for p in rot_pairs]

            c1_3d = sum((v*w for v,w in zip(v1_list, weights)), Vector((0,0,0))).normalized()
            c2_new_3d = sum((v*w for v,w in zip(v2_list_new, weights)), Vector((0,0,0))).normalized()

            if is_tripod:
                if eff_scale_mode == 'FOCAL_LENGTH' and use_refined_solver:
                    full_delta_quat = solve_weighted_kabsch_rotation(v1_list, v2_list_new, props.clip_lock_roll, weights)
                    if props.clip_lock_roll:
                        q_pt = full_delta_quat
                    else:
                        try:
                            q_pt, _twist_quat = full_delta_quat.to_swing_twist(c1_3d.normalized())
                        except Exception:
                            q_pt = solve_tripod_pan_tilt_from_rays(v1_list, v2_list_new, weights)
                elif eff_scale_mode == 'NONE' or not use_refined_solver:
                    q_pt = c2_new_3d.rotation_difference(c1_3d)
                    full_delta_quat = q_pt
                else:
                    q_pt = solve_tripod_pan_tilt_from_rays(v1_list, v2_list_new, weights)
                    full_delta_quat = q_pt
                if eff_scale_mode == 'Z_DEPTH':
                    e_pt = q_pt.to_euler('XYZ')
                    pan_raw = e_pt.y
                    c2_pan = Matrix.Rotation(pan_raw, 3, 'Y') @ c2_new_3d
                    tilt_angle = wrap_pi(
                        math.atan2(c1_3d.y, -c1_3d.z) -
                        math.atan2(c2_pan.y, -c2_pan.z)
                    )
                    pan_angle = pan_raw
                    q_pt_sens = Euler((tilt_angle, pan_angle, 0.0), 'XYZ').to_quaternion()
                else:
                    e_pt = q_pt.to_euler('XYZ')
                    tilt_angle = e_pt.x
                    pan_angle = e_pt.y
                    q_pt_sens = q_pt
                cur_depth = depth / norm_curve[f-1] if norm_curve[f-1] > 1e-6 else depth
                new_depth = depth / norm_curve[f] if norm_curve[f] > 1e-6 else depth
                dz = -(cur_depth - new_depth) if eff_scale_mode == 'Z_DEPTH' else 0.0
                
                delta_roll = 0.0
                if not props.clip_lock_roll:
                    v2_aligned = [q_pt_sens @ v for v in v2_list_new]
                    angles, valid_weights = [], []
                    for v1, v2_a, w in zip(v1_list, v2_aligned, weights):
                        v1_proj = v1 - v1.project(c1_3d)
                        v2_proj = v2_a - v2_a.project(c1_3d)
                        if v1_proj.length_squared > 1e-6 and v2_proj.length_squared > 1e-6:
                            cross = v2_proj.cross(v1_proj)
                            sign = -1.0 if cross.dot(c1_3d) > 0 else 1.0
                            angles.append(v2_proj.angle(v1_proj) * sign)
                            valid_weights.append(w)
                    if angles and sum(valid_weights) > 1e-6:
                        delta_roll = sum(a*w for a,w in zip(angles, valid_weights)) / sum(valid_weights)
                
                roll_angle = delta_roll
                if eff_scale_mode == 'FOCAL_LENGTH':
                    if use_refined_solver:
                        if not props.clip_lock_roll:
                            full_delta_quat = replace_quaternion_twist(full_delta_quat, c1_3d, roll_angle)
                        next_rot_mat = current_rot_mat @ full_delta_quat.to_matrix().to_4x4()
                    else:
                        roll_quat = Quaternion(c1_3d, roll_angle)
                        next_rot_mat = current_rot_mat @ (roll_quat @ q_pt).to_matrix().to_4x4()
                else:
                    mat_x = Matrix.Rotation(tilt_angle, 4, 'X')
                    mat_y = Matrix.Rotation(pan_angle, 4, 'Y')
                    mat_z = Matrix.Rotation(roll_angle, 4, 'Z')
                    next_rot_mat = current_rot_mat @ (mat_y @ mat_x @ mat_z)
                    current_loc += current_rot_mat @ Vector((0.0, 0.0, dz))
                    dz = 0.0
                current_rot_mat = next_rot_mat

                if use_refined_solver and eff_scale_mode != 'NONE':
                    anchor_ref_rays = []
                    anchor_curr_rays = []
                    anchor_weights = []
                    markers_curr = frame_markers.get(f, {})
                    for track_name in set(ref_markers.keys()) & set(markers_curr.keys()):
                        p_ref = ref_markers[track_name]
                        p_curr = markers_curr[track_name]
                        if props.clip_center_weight:
                            w = marker_center_weight(p_curr, aspect)
                        else:
                            w = 1.0
                        anchor_ref_rays.append(marker_to_camera_ray(p_ref, tan_ref_x, tan_ref_y))
                        anchor_curr_rays.append(marker_to_camera_ray(p_curr, tan_x2, tan_y2))
                        anchor_weights.append(w)

                    if anchor_ref_rays:
                        anchor_quat = solve_weighted_kabsch_rotation(
                            anchor_ref_rays,
                            anchor_curr_rays,
                            props.clip_lock_roll,
                            anchor_weights,
                        )
                        if not props.clip_lock_roll:
                            anchor_axis = sum((v * w for v, w in zip(anchor_ref_rays, anchor_weights)), Vector((0.0, 0.0, 0.0)))
                            if anchor_axis.length_squared > 1e-9:
                                anchor_quat = enforce_roll_sign_continuity(anchor_quat, anchor_ref_rays, anchor_curr_rays, anchor_axis, anchor_weights)
                        desired_rot_mat = init_c_rot4 @ anchor_quat.to_matrix().to_4x4()
                        current_rot_mat = soft_reanchor_rotation(current_rot_mat, desired_rot_mat, len(anchor_ref_rays), 0.72)

            else:
                c2_unscaled = Vector((0.5, 0.5)) + (c2_raw - Vector((0.5, 0.5))) / step_ratio if step_ratio > 1e-6 else c2_raw
                
                cur_depth = depth / norm_curve[f-1] if norm_curve[f-1] > 1e-6 else depth
                new_depth = depth / norm_curve[f] if norm_curve[f] > 1e-6 else depth
                
                eff_depth = cur_depth if eff_scale_mode == 'Z_DEPTH' else depth
                w_3d, h_3d = 2.0 * eff_depth * tan_x1, 2.0 * eff_depth * tan_y1
                dx, dy = -(c2_unscaled.x - c1_raw.x) * w_3d, -(c2_unscaled.y - c1_raw.y) * h_3d
                dz = -(cur_depth - new_depth) if eff_scale_mode == 'Z_DEPTH' else 0.0
                current_loc += current_rot_mat @ Vector((dx, dy, dz))

                delta_roll = 0.0
                if not props.clip_lock_roll:
                    angles, valid_w = [], []
                    for p1, p2, w in zip([vp[0] for vp in valid_pairs], [vp[1] for vp in valid_pairs], weights):
                        v1, v2 = p1 - c1_raw, p2 - c2_raw
                        if v1.length_squared > 1e-6 and v2.length_squared > 1e-6:
                            a1, a2 = math.atan2(v1.y, v1.x * aspect), math.atan2(v2.y, v2.x * aspect)
                            diff = a2 - a1
                            while diff > math.pi:
                                diff -= 2 * math.pi
                            while diff < -math.pi:
                                diff += 2 * math.pi
                            angles.append(diff)
                            valid_w.append(w * v1.length_squared)
                    if angles and sum(valid_w) > 1e-9:
                        raw_roll = sum(a * w for a, w in zip(angles, valid_w)) / sum(valid_w)
                        delta_roll = max(min(raw_roll, math.radians(0.5)), -math.radians(0.5))
                current_rot_mat = current_rot_mat @ Matrix.Rotation(-delta_roll, 4, 'Z')

                if use_refined_solver:
                    markers_curr = frame_markers.get(f, {})
                    shared_names = list(set(ref_markers.keys()) & set(markers_curr.keys()))
                    if shared_names:
                        c_ref_anchor = weighted_marker_centroid(ref_markers, shared_names, aspect, props.clip_center_weight)
                        c_curr_anchor = weighted_marker_centroid(markers_curr, shared_names, aspect, props.clip_center_weight)
                        if c_ref_anchor is not None and c_curr_anchor is not None:
                            if eff_scale_mode == 'Z_DEPTH' and norm_curve[f] > 1e-6:
                                center_uv = Vector((0.5, 0.5))
                                c_curr_anchor = center_uv + (c_curr_anchor - center_uv) / norm_curve[f]
                            eff_depth_abs = new_depth if eff_scale_mode == 'Z_DEPTH' else depth
                            dx_abs = -(c_curr_anchor.x - c_ref_anchor.x) * (2.0 * eff_depth_abs * tan_x2)
                            dy_abs = -(c_curr_anchor.y - c_ref_anchor.y) * (2.0 * eff_depth_abs * tan_y2)
                            dz_abs = -(depth - new_depth) if eff_scale_mode == 'Z_DEPTH' else 0.0
                            desired_loc = init_c_mat.translation + current_rot_mat.to_3x3() @ Vector((dx_abs, dy_abs, dz_abs))
                            loc_blend = 0.10 + 0.05 * min(len(shared_names) - 1, 3)
                            current_loc = current_loc.lerp(desired_loc, min(0.25, loc_blend))

            traj_rot[f] = current_rot_mat.copy()
            traj_loc[f] = current_loc.copy()
            traj_lens[f] = target_lens[f]

        cam_ref.data.lens = ref_lens

        m_align = init_c_rot4 @ traj_rot[ref_f].inverted()
        loc_offset = init_c_mat.translation - m_align @ traj_loc[ref_f]

        for f in range(frame_start, frame_end + 1):
            context.scene.frame_set(f)
            
            f_rot = m_align @ traj_rot[f]
            f_loc = loc_offset + m_align @ traj_loc[f]
            f_lens = traj_lens[f]

            if is_obj:
                o_m = init_c_mat @ (f_rot.inverted() @ Matrix.Translation(-f_loc)) @ init_t_mat
                self.set_target_rotation(target, o_m)
                target.location = o_m.translation
                if props.scale_mode == 'FOCAL_LENGTH':
                    target.scale = init_t_mat.to_scale() * (f_lens / ref_lens)
                target.keyframe_insert(data_path="scale", frame=f)
            else:
                if props.lock_camera_z and not is_tripod:
                    f_loc, rot_mat = apply_z_lock(f_loc, f_rot, global_focus_point, init_c_mat.translation.z)
                    f_rot = rot_mat
                self.set_target_rotation(target, f_rot)
                target.location = f_loc
                if eff_scale_mode == 'FOCAL_LENGTH':
                    target.data.lens = f_lens
                    target.data.keyframe_insert(data_path="lens", frame=f)

            self.keyframe_target_rotation(target, f)
            target.keyframe_insert(data_path="location", frame=f)

        context.scene.frame_set(ref_f)
        if pin_existing_focal_range:
            self.pin_lens_constant_in_range(cam_ref.data, frame_start, frame_end, ref_lens, lens_curve_snapshot)
        self.report({'INFO'}, f"Applied Clip Track motion to '{target.name}'.")
        return {'FINISHED'}


class OBJECT_OT_apply_tracking_data(PCamAnimationIO, PCamClipTrackSolver, bpy.types.Operator):
    bl_idname = "view3d.pcam_solve_apply_tracking_data"
    bl_label = "Bake Tracking to Target"
    bl_options = {'REGISTER', 'UNDO'}

    # Target transform helpers. Solvers can work in quaternions internally while
    # final baking respects the object's current rotation mode.
    def get_target_rotation_quaternion(self, target):
        if target.rotation_mode == 'QUATERNION':
            return target.rotation_quaternion.copy()
        if target.rotation_mode == 'AXIS_ANGLE':
            axis_angle = target.rotation_axis_angle
            axis = Vector((axis_angle[1], axis_angle[2], axis_angle[3]))
            if axis.length_squared < 1e-12:
                return Quaternion()
            axis.normalize()
            return Quaternion(axis, axis_angle[0])
        return target.rotation_euler.to_quaternion()

    def get_target_rotation_matrix(self, target):
        return self.get_target_rotation_quaternion(target).to_matrix().to_4x4()

    def set_target_rotation(self, target, rotation):
        self.set_target_rotation_continuous(target, rotation)

    def set_target_rotation_continuous(self, target, rotation, prev_quat_hint=None, prev_euler_hint=None):
        if isinstance(rotation, Matrix):
            rot_quat = rotation.to_quaternion()
        elif isinstance(rotation, Euler):
            rot_quat = rotation.to_quaternion()
        else:
            rot_quat = rotation.copy()

        prev_quat = prev_quat_hint.copy() if prev_quat_hint is not None else self.get_target_rotation_quaternion(target)
        if prev_quat.dot(prev_quat) > 1e-12 and rot_quat.dot(prev_quat) < 0.0:
            rot_quat.negate()

        if target.rotation_mode == 'QUATERNION':
            target.rotation_quaternion = rot_quat
        elif target.rotation_mode == 'AXIS_ANGLE':
            axis, angle = rot_quat.to_axis_angle()
            target.rotation_axis_angle = (angle, axis.x, axis.y, axis.z)
        else:
            prev_euler = prev_euler_hint.copy() if prev_euler_hint is not None else target.rotation_euler.copy()
            target.rotation_euler = rot_quat.to_euler(target.rotation_mode, prev_euler)
        return rot_quat.copy()

    def keyframe_target_rotation(self, target, frame):
        if target.rotation_mode == 'QUATERNION':
            target.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        elif target.rotation_mode == 'AXIS_ANGLE':
            target.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
        else:
            target.keyframe_insert(data_path="rotation_euler", frame=frame)

    def create_static_follow_camera(self, context, source_cam, matrix_world):
        if source_cam is None or getattr(source_cam, "data", None) is None:
            return None
        cam_data = source_cam.data.copy()
        cam_data.animation_data_clear()
        temp_cam = bpy.data.objects.new("PCam_FollowTrack_Camera", cam_data)
        temp_cam.animation_data_clear()
        context.scene.collection.objects.link(temp_cam)
        temp_cam.matrix_world = matrix_without_scale(matrix_world)
        temp_cam.hide_render = True
        temp_cam.hide_select = True
        context.view_layer.update()
        return temp_cam

    def remove_static_follow_camera(self, temp_cam):
        if temp_cam is None:
            return
        cam_data = getattr(temp_cam, "data", None)
        bpy.data.objects.remove(temp_cam, do_unlink=True)
        if cam_data is not None and cam_data.users == 0:
            bpy.data.cameras.remove(cam_data)

    # Track extraction helpers. Follow Track is still the source of truth for
    # point modes because it preserves Blender's own undistort/depth behavior.
    def estimate_track_group_depth(self, context, cam, clip, tracks, frame, depth_obj):
        if not cam or not depth_obj or not tracks:
            return None

        f_clip = frame - clip.frame_start + 1 - clip.frame_offset
        cam_loc = cam.matrix_world.translation
        depths = []

        for track in tracks:
            marker = track.markers.find_frame(f_clip)
            if not marker or getattr(marker, 'mute', False):
                continue
            hit = raycast_marker_world(context, cam, depth_obj, get_track_display_co(track, marker))
            if hit is not None:
                depths.append((hit - cam_loc).length)

        if not depths:
            return None
        return max(depths)

    def extract_tracks_data(self, context, cam, clip, track_names, depth_obj, use_undistort, smoothing):
        track_names = [name for name in track_names if name and name != "NONE"]
        if not track_names:
            return []

        orig_active = context.view_layer.objects.active
        orig_selected = context.selected_objects[:]
        orig_frame = context.scene.frame_current 
        orig_scene_camera = context.scene.camera
        props = context.scene.pcam_solve_props

        context.scene.camera = cam
        context.view_layer.update()
        
        t_obj_idx = int(props.tracking_object_idx)
        track_object = clip.tracking.objects[t_obj_idx]
        empties = []

        bpy.ops.object.select_all(action='DESELECT')
        for track_name in track_names:
            empty = bpy.data.objects.new(f"PCam_FollowTrack_{track_name}", None)
            empty.empty_display_type = 'PLAIN_AXES'
            context.scene.collection.objects.link(empty)
            empty.select_set(True)
            context.view_layer.objects.active = empty

            cons = empty.constraints.new(type='FOLLOW_TRACK')
            cons.use_active_clip = False
            cons.clip = clip
            cons.object = track_object.name
            cons.track = track_name
            cons.camera = cam
            if depth_obj:
                cons.depth_object = depth_obj
            cons.use_3d_position = False
            cons.use_undistorted_position = use_undistort
            empties.append((track_name, empty))

        if empties:
            context.view_layer.objects.active = empties[0][1]
        context.view_layer.update()

        f_s = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        f_e = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        track_data_list = [{} for _ in empties]

        def evaluated_empty_location(empty):
            depsgraph = context.evaluated_depsgraph_get()
            try:
                depsgraph.update()
            except Exception:
                pass
            return empty.evaluated_get(depsgraph).matrix_world.translation.copy()

        try:
            if empties:
                context.scene.frame_set(f_s)
                context.view_layer.update()
                bpy.ops.nla.bake(frame_start=f_s, frame_end=f_e, step=1, only_selected=True, visual_keying=True, clear_constraints=True, use_current_action=False, bake_types={'OBJECT'})
            for f in range(f_s, f_e + 1):
                f_clip = f - clip.frame_start + 1 - clip.frame_offset
                context.scene.frame_set(f)
                context.view_layer.update()
                for index, (track_name, empty) in enumerate(empties):
                    track = track_object.tracks.get(track_name)
                    marker = track.markers.find_frame(f_clip) if track else None
                    if not marker or getattr(marker, 'mute', False):
                        continue
                    if empty.animation_data and empty.animation_data.action:
                        track_data_list[index][f] = evaluated_empty_location(empty)
        finally:
            for _, empty in empties:
                action = empty.animation_data.action if empty.animation_data else None
                bpy.data.objects.remove(empty, do_unlink=True)
                if action is not None and action.users == 0:
                    bpy.data.actions.remove(action)
            context.scene.camera = orig_scene_camera
            context.scene.frame_set(orig_frame)
            context.view_layer.update()
            for o in orig_selected:
                try:
                    o.select_set(True)
                except Exception:
                    pass
            context.view_layer.objects.active = orig_active

        if smoothing:
            track_data_list = [savitzky_golay_filter(track_data) for track_data in track_data_list]
        return track_data_list

    def extract_track_data(self, context, cam, clip, track_name, depth_obj, use_undistort, smoothing):
        if not track_name or track_name == "NONE":
            return {}
        track_data_list = self.extract_tracks_data(
            context,
            cam,
            clip,
            [track_name],
            depth_obj,
            use_undistort,
            smoothing,
        )
        return track_data_list[0] if track_data_list else {}

    # Main dispatcher.
    def execute(self, context):
        props = context.scene.pcam_solve_props
        props.track_preview = False 
        
        if not props.target_clip:
            self.report({'ERROR'}, "No Clip.")
            return {'CANCELLED'}
        cam = context.scene.camera
        if not cam:
            self.report({'ERROR'}, "No Camera.")
            return {'CANCELLED'}
            
        target = cam if props.apply_to == 'CAMERA' else props.target_object
        if not target:
            self.report({'ERROR'}, "No Target.")
            return {'CANCELLED'}

        block_reason = pcam_get_bake_block_reason(context, props)
        if block_reason:
            self.report({'ERROR'}, block_reason)
            return {'CANCELLED'}
            
        if props.mode == 'CLIP_TRACK':
            return self.execute_clip_track(context, target)
        elif props.mode == 'ONE_POINT':
            return self.execute_one_point(context, target)
        elif props.mode == 'TWO_POINT':
            return self.execute_two_point(context, target)
        elif props.mode == 'THREE_POINT':
            return self.execute_three_point(context, target)
        
        return {'FINISHED'}

    # 1/2/3 point solvers.
    def execute_one_point_object_follow_track(self, context, target):
        props = context.scene.pcam_solve_props
        clip = props.target_clip
        cam_ref = context.scene.camera
        if not clip or not cam_ref:
            return {'CANCELLED'}

        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        frame_range = (frame_start, frame_end) if props.use_custom_range else None

        target_curve_snapshot = self.snapshot_animation_action(target)
        orig_frame = context.scene.frame_current
        orig_scene_camera = context.scene.camera
        orig_active = context.view_layer.objects.active
        orig_selected = context.selected_objects[:]
        cons = None

        try:
            context.scene.camera = cam_ref
            self.clear_animation_safely(target, frame_range)
            bpy.ops.object.select_all(action='DESELECT')
            target.select_set(True)
            context.view_layer.objects.active = target

            cons = target.constraints.new(type='FOLLOW_TRACK')
            cons.use_active_clip = False
            cons.clip = clip
            try:
                cons.object = clip.tracking.objects[int(props.tracking_object_idx)].name
            except Exception:
                pass
            cons.track = props.track_1
            cons.camera = cam_ref
            if props.clip_depth_object:
                cons.depth_object = props.clip_depth_object
            cons.use_3d_position = False
            cons.use_undistorted_position = props.use_undistort

            context.scene.frame_set(frame_start)
            context.view_layer.update()
            bpy.ops.nla.bake(
                frame_start=frame_start,
                frame_end=frame_end,
                step=1,
                only_selected=True,
                visual_keying=True,
                clear_constraints=True,
                use_current_action=False,
                bake_types={'OBJECT'},
            )
        finally:
            if cons is not None:
                try:
                    target.constraints.remove(cons)
                except Exception:
                    pass
            context.scene.camera = orig_scene_camera
            context.scene.frame_set(orig_frame)
            context.view_layer.update()
            for obj in orig_selected:
                try:
                    obj.select_set(True)
                except Exception:
                    pass
            context.view_layer.objects.active = orig_active

        if not target.animation_data or not target.animation_data.action:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            self.report({'ERROR'}, "No frames could be baked from Track 1.")
            return {'CANCELLED'}

        context.scene.frame_set(pcam_get_reference_frame(context, props, frame_start, frame_end))
        self.report({'INFO'}, f"Applied 1-point Follow Track to '{target.name}'.")
        return {'FINISHED'}

    def execute_one_point(self, context, target):
        props = context.scene.pcam_solve_props
        if not props.track_1 or props.track_1 == "NONE":
            self.report({'ERROR'}, "Track 1 missing.")
            return {'CANCELLED'}
            
        is_obj = (props.apply_to == 'OBJECT')
        if is_obj:
            return self.execute_one_point_object_follow_track(context, target)
        cam_ref = context.scene.camera
        eff_depth_obj = props.clip_depth_object

        clip = props.target_clip
        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        follow_cam = self.create_static_follow_camera(context, cam_ref, evaluated_matrix_world(context, cam_ref)) if cam_ref else None

        frame_range = (props.bake_start, props.bake_end) if props.use_custom_range else None
        keep_existing_position = props.clip_use_existing_position
        existing_loc_curve = {}
        location_curve_snapshot = self.snapshot_animation_curves(target, {"location"}) if keep_existing_position else []
        if keep_existing_position:
            restore_frame = context.scene.frame_current
            for f in range(frame_start, frame_end + 1):
                context.scene.frame_set(f)
                existing_loc_curve[f] = target.location.copy()
            context.scene.frame_set(restore_frame)

        target_curve_snapshot = self.snapshot_animation_action(target)
        lens_curve_snapshot = self.snapshot_animation_action(target.data) if getattr(target, "data", None) is not None else []
        has_existing_focal_keys = self.has_camera_focal_length_keys(cam_ref)
        has_focal_variation_in_range = frame_range is not None and self.camera_lens_varies_over_range(context, cam_ref, frame_range[0], frame_range[1])
        pin_existing_focal_range = frame_range is not None and props.scale_mode != 'FOCAL_LENGTH' and (has_existing_focal_keys or has_focal_variation_in_range)
        pinned_lens_value = float(target.data.lens) if getattr(target, "data", None) is not None else None
        self.clear_animation_safely(target, frame_range)
        if pin_existing_focal_range:
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)
        try:
            t1 = self.extract_track_data(context, follow_cam or cam_ref, props.target_clip, props.track_1, eff_depth_obj, props.use_undistort, props.track_smoothing)
        finally:
            self.remove_static_follow_camera(follow_cam)
        if not t1:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            self.report({'ERROR'}, "No valid frames found for Track 1 in the bake range.")
            return {'CANCELLED'}
            
        valid_f = sorted(t1.keys())
        ref_f = pcam_pick_valid_reference_frame(valid_f, ref_hint, props.use_reference_frame_lock)
        if ref_f is None:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            self.report({'ERROR'}, "Reference Frame has no valid Track 1 data.")
            return {'CANCELLED'}
        context.scene.frame_set(ref_f)
        if keep_existing_position:
            target.location = existing_loc_curve.get(ref_f, target.location).copy()
        
        init_t_mat = target.matrix_world.copy()
        init_t_mat = matrix_without_scale(init_t_mat)
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        p_start = t1.get(ref_f, t1[valid_f[0]])
        baked_frames = 0

        for f in valid_f:
            context.scene.frame_set(f)
            p_cur = t1[f]
            
            if keep_existing_position:
                target.location = existing_loc_curve.get(f, init_t_loc.copy()).copy()
                solved_quat = solve_single_track_rotation_from_follow_point(
                    p_start,
                    p_cur,
                    target.location.copy(),
                    init_t_loc,
                    init_t_rot,
                    init_t_rot,
                )
                self.set_target_rotation(target, solved_quat)
            elif props.tripod_mode:
                vec_start = p_start - init_t_mat.translation
                vec_current = p_cur - init_t_mat.translation
                if vec_start.length_squared > 1e-9 and vec_current.length_squared > 1e-9:
                    delta_rot = vec_start.rotation_difference(vec_current)
                    self.set_target_rotation(target, delta_rot.inverted() @ init_t_rot)
            else:
                init_c_rot = init_t_mat.to_3x3()
                delta_pos = p_cur - p_start
                move_local = init_c_rot.inverted() @ delta_pos
                move_scaled = Vector((move_local.x, move_local.y, 0.0))
                target.location = init_t_loc - (init_c_rot @ move_scaled)
                
                if props.lock_camera_z:
                    loc, rot_mat = apply_z_lock(target.location, self.get_target_rotation_matrix(target), p_cur, init_t_loc.z)
                    target.location = loc
                    self.set_target_rotation(target, rot_mat)
                        
            if not keep_existing_position:
                target.keyframe_insert("location", frame=f)
            self.keyframe_target_rotation(target, f)
            baked_frames += 1

        if baked_frames == 0:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            self.report({'ERROR'}, "No frames could be baked from Track 1.")
            return {'CANCELLED'}

        if keep_existing_position:
            self.restore_animation_curves(target, location_curve_snapshot)
        context.scene.frame_set(ref_f)
        if pin_existing_focal_range:
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)
        total_frames = frame_end - frame_start + 1
        suffix = f" Solved {baked_frames}/{total_frames} frames." if baked_frames < total_frames else ""
        self.report({'INFO'}, f"Applied 1-point motion to '{target.name}'.{suffix}")
        return {'FINISHED'}

    def execute_marker_tripod_none(self, context, target, track_names, label):
        props = context.scene.pcam_solve_props
        clip = props.target_clip
        cam_ref = context.scene.camera
        if not clip or not cam_ref:
            return {'CANCELLED'}

        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_f = pcam_get_reference_frame(context, props, frame_start, frame_end)

        init_t_mat = target.matrix_world.copy()
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        init_rot4 = init_t_rot.to_matrix().to_4x4()
        tan_x, tan_y = get_camera_tan(cam_ref.data, cam_ref.data.lens, context.scene)

        current_rot_mat = init_rot4.copy()
        traj_rot = {frame_start: current_rot_mat.copy()}
        extracted_list = self.extract_tracks_data(
            context,
            cam_ref,
            clip,
            track_names,
            None,
            props.use_undistort,
            props.track_smoothing,
        )
        extracted_tracks = {
            track_name: track_data
            for track_name, track_data in zip(track_names, extracted_list)
        }
        cam_loc = init_t_mat.to_translation()

        for f in range(frame_start + 1, frame_end + 1):
            valid_pairs = []
            for track_name in track_names:
                track_data = extracted_tracks.get(track_name, {})
                p_prev = track_data.get(f - 1)
                p_curr = track_data.get(f)
                if p_prev is None or p_curr is None:
                    continue
                m_prev = get_track_marker_co(clip, props.tracking_object_idx, track_name, f - 1)
                m_curr = get_track_marker_co(clip, props.tracking_object_idx, track_name, f)
                valid_pairs.append((p_prev, p_curr, m_prev, m_curr))

            if len(valid_pairs) >= 2:
                if len(valid_pairs) >= 3:
                    motions = [(p_curr - p_prev).length for p_prev, p_curr, _, _ in valid_pairs]
                    filtered_motions = robust_filter_values(motions)
                    if len(filtered_motions) != len(motions):
                        motion_set = list(filtered_motions)
                        kept_pairs = []
                        for pair, motion in zip(valid_pairs, motions):
                            matched = False
                            for i, filtered_motion in enumerate(motion_set):
                                if abs(motion - filtered_motion) < 1e-9:
                                    kept_pairs.append(pair)
                                    motion_set.pop(i)
                                    matched = True
                                    break
                            if matched:
                                continue
                        if len(kept_pairs) >= 2:
                            valid_pairs = kept_pairs

                if props.clip_center_weight:
                    aspect = tan_x / tan_y if tan_y > 1e-6 else 1.0
                    weights = [
                        marker_center_weight(marker_prev, aspect) if marker_prev is not None else 1.0
                        for _, _, marker_prev, _ in valid_pairs
                    ]
                else:
                    weights = [1.0] * len(valid_pairs)
                filtered = []
                filtered_weights = []
                for pair, weight in zip(valid_pairs, weights):
                    p_prev, p_curr, marker_prev, marker_curr = pair
                    v1 = p_prev - cam_loc
                    v2 = p_curr - cam_loc
                    if v1.length_squared > 1e-9 and v2.length_squared > 1e-9:
                        filtered.append((v1.normalized(), v2.normalized(), marker_prev, marker_curr))
                        filtered_weights.append(weight)
                v1_list = [v1 for v1, _, _, _ in filtered]
                v2_list = [v2 for _, v2, _, _ in filtered]
                valid_pairs = [(None, None, mp, mc) for _, _, mp, mc in filtered]
                weights = filtered_weights

                c1_3d = sum((v * w for v, w in zip(v1_list, weights)), Vector((0.0, 0.0, 0.0)))
                c2_3d = sum((v * w for v, w in zip(v2_list, weights)), Vector((0.0, 0.0, 0.0)))
                if c1_3d.length_squared > 1e-9 and c2_3d.length_squared > 1e-9:
                    c1_3d.normalize()
                    c2_3d.normalize()

                    q_pt = c2_3d.rotation_difference(c1_3d)
                    e_pt = q_pt.to_euler('XYZ')
                    tilt_angle = e_pt.x
                    pan_angle = e_pt.y

                    delta_roll = 0.0
                    if not props.clip_lock_roll:
                        v2_aligned = [q_pt @ v for v in v2_list]
                        angles = []
                        valid_weights = []
                        for v1, v2_a, w in zip(v1_list, v2_aligned, weights):
                            v1_proj = v1 - v1.project(c1_3d)
                            v2_proj = v2_a - v2_a.project(c1_3d)
                            if v1_proj.length_squared > 1e-6 and v2_proj.length_squared > 1e-6:
                                cross = v2_proj.cross(v1_proj)
                                sign = -1.0 if cross.dot(c1_3d) > 0 else 1.0
                                angles.append(v2_proj.angle(v1_proj) * sign)
                                valid_weights.append(w)
                        if angles and sum(valid_weights) > 1e-6:
                            delta_roll = sum(a * w for a, w in zip(angles, valid_weights)) / sum(valid_weights)

                    roll_angle = delta_roll
                    mat_x = Matrix.Rotation(tilt_angle, 4, 'X')
                    mat_y = Matrix.Rotation(pan_angle, 4, 'Y')
                    mat_z = Matrix.Rotation(roll_angle, 4, 'Z')
                    current_rot_mat = current_rot_mat @ (mat_y @ mat_x @ mat_z)

            traj_rot[f] = current_rot_mat.copy()

        m_align = init_rot4 @ traj_rot[ref_f].inverted()
        self.clear_animation_safely(target, (frame_start, frame_end) if props.use_custom_range else None)

        for f in range(frame_start, frame_end + 1):
            context.scene.frame_set(f)
            f_rot = m_align @ traj_rot[f]
            target.location = init_t_loc
            self.set_target_rotation(target, f_rot)
            target.keyframe_insert("location", frame=f)
            self.keyframe_target_rotation(target, f)

        context.scene.frame_set(ref_f)
        self.report({'INFO'}, f"Applied {label} tripod motion to '{target.name}'.")
        return {'FINISHED'}

    def execute_two_point(self, context, target):
        props = context.scene.pcam_solve_props
        if props.track_1 == "NONE" or props.track_2 == "NONE":
            self.report({'ERROR'}, "Tracks missing.")
            return {'CANCELLED'}
        if props.track_1 == props.track_2:
            self.report({'ERROR'}, "Track 1 and Track 2 must be different.")
            return {'CANCELLED'}
            
        is_obj = (props.apply_to == 'OBJECT')
        cam_ref = context.scene.camera
        if not is_obj and props.tripod_mode and props.scale_mode == 'NONE':
            return self.execute_marker_tripod_none(context, target, [props.track_1, props.track_2], "2-point")
        eff_depth_obj = props.clip_depth_object if props.clip_depth_object else (target if is_obj else None)

        clip = props.target_clip
        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        follow_cam = self.create_static_follow_camera(context, cam_ref, evaluated_matrix_world(context, cam_ref)) if cam_ref and not is_obj else None

        frame_range = (props.bake_start, props.bake_end) if props.use_custom_range else None
        keep_existing_position = (not is_obj) and props.clip_use_existing_position and not (props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH')
        lens_curve_snapshot = self.snapshot_animation_action(target.data) if (not is_obj and getattr(target, "data", None) is not None) else []
        has_existing_focal_keys = (not is_obj) and self.has_camera_focal_length_keys(cam_ref)
        use_existing_focal = props.clip_use_existing_focal or (keep_existing_position and props.scale_mode == 'FOCAL_LENGTH')
        keep_existing_focal = (not is_obj) and props.scale_mode == 'FOCAL_LENGTH' and use_existing_focal and has_existing_focal_keys
        suppress_focal_bake = (not is_obj) and props.scale_mode == 'FOCAL_LENGTH' and use_existing_focal and not has_existing_focal_keys
        has_focal_variation_in_range = (not is_obj) and frame_range is not None and self.camera_lens_varies_over_range(context, cam_ref, frame_range[0], frame_range[1])
        pin_existing_focal_range = frame_range is not None and not is_obj and props.scale_mode != 'FOCAL_LENGTH' and not keep_existing_focal and (has_existing_focal_keys or has_focal_variation_in_range)
        target_curve_snapshot = self.snapshot_animation_action(target)
        existing_loc_curve = {}
        existing_lens_curve = {}
        location_curve_snapshot = self.snapshot_animation_curves(target, {"location"}) if keep_existing_position else []
        lens_action_copy = self.copy_animation_action(target.data) if keep_existing_focal and getattr(target, "data", None) else None
        if keep_existing_position or keep_existing_focal:
            restore_frame = context.scene.frame_current
            for f in range(frame_start, frame_end + 1):
                context.scene.frame_set(f)
                if keep_existing_position:
                    existing_loc_curve[f] = target.location.copy()
                if keep_existing_focal:
                    existing_lens_curve[f] = float(target.data.lens)
            context.scene.frame_set(restore_frame)
        pinned_lens_value = float(target.data.lens) if (not is_obj and getattr(target, "data", None) is not None) else None
        if frame_range is None:
            self.clear_animation_safely(
                target,
                None,
                keep_target_paths=None,
                keep_data_paths={"lens"} if keep_existing_focal else None,
            )
        else:
            self.clear_animation_safely(
                target,
                frame_range,
                keep_target_paths=None,
                keep_data_paths={"lens"} if keep_existing_focal else None,
            )
        if pin_existing_focal_range and getattr(target, "data", None):
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)

        extract_cam = follow_cam or cam_ref
        try:
            t_d = self.extract_tracks_data(
                context,
                extract_cam,
                props.target_clip,
                [props.track_1, props.track_2],
                eff_depth_obj,
                props.use_undistort,
                props.track_smoothing,
            )
            valid_f = sorted(set(t_d[0].keys()) & set(t_d[1].keys()))
            spread_f = frames_with_point_spread(t_d, valid_f)
        finally:
            self.remove_static_follow_camera(follow_cam)
        if not valid_f:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, "No frames with both selected trackers were found in the bake range.")
            return {'CANCELLED'}
        if not spread_f:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, f"Follow Track evaluation produced degenerate 2-point positions in the bake range. Max spread: {max_point_spread(t_d, valid_f):.6g}.")
            return {'CANCELLED'}
            
        ref_f = pcam_pick_valid_reference_frame(spread_f, ref_hint, props.use_reference_frame_lock)
        if ref_f is None:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, "Reference Frame has no valid 2-point tracker spread.")
            return {'CANCELLED'}
        context.scene.frame_set(ref_f)
        init_t_mat = target.matrix_world.copy()
        if not is_obj:
            init_t_mat = matrix_without_scale(init_t_mat)
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        init_t_scale = target.scale.copy()
        
        init_f_len = cam_ref.data.lens if cam_ref else 35.0
        p1_start, p2_start = t_d[0][ref_f], t_d[1][ref_f]
        fixed_world_points = {
            props.track_1: p1_start.copy(),
            props.track_2: p2_start.copy(),
        }
        depth_ref_mat = evaluated_matrix_world(context, props.clip_depth_object) if is_obj and props.clip_depth_object else None
        depth_ref_inv = depth_ref_mat.inverted() if depth_ref_mat is not None else None
        depth_ref_quat_inv = depth_ref_mat.to_quaternion().inverted() if depth_ref_mat is not None else None
        baked_frames = 0
        skip_counts = {
            "zero_pair": 0,
            "missing_marker": 0,
            "zero_local_pair": 0,
        }

        for f in valid_f:
            context.scene.frame_set(f)
            p1_curr, p2_curr = t_d[0][f], t_d[1][f]
            
            vec_start = p2_start - p1_start
            vec_curr = p2_curr - p1_curr
            if vec_curr.length_squared == 0:
                skip_counts["zero_pair"] += 1
                continue
                
            if is_obj:
                center_start = (p1_start + p2_start) / 2.0
                center_curr = (p1_curr + p2_curr) / 2.0
                target.location = init_t_loc + (center_curr - center_start)
                
                delta_rot_quat = vec_start.rotation_difference(vec_curr)
                solved_obj_quat = delta_rot_quat @ init_t_rot
                depth_local_scale_ratio = None
                if depth_ref_mat is not None:
                    depth_curr_mat = evaluated_matrix_world(context, props.clip_depth_object)
                    depth_curr_inv = depth_curr_mat.inverted()
                    vec_start_local = (depth_ref_inv @ p2_start) - (depth_ref_inv @ p1_start)
                    vec_curr_local = (depth_curr_inv @ p2_curr) - (depth_curr_inv @ p1_curr)
                    if vec_start_local.length_squared > 1e-9 and vec_curr_local.length_squared > 1e-9:
                        local_delta_quat = vec_start_local.rotation_difference(vec_curr_local)
                        solved_obj_quat = depth_curr_mat.to_quaternion() @ local_delta_quat @ depth_ref_quat_inv @ init_t_rot
                        depth_local_scale_ratio = vec_curr_local.length / vec_start_local.length
                self.set_target_rotation(target, solved_obj_quat)
                
                scale_ratio = depth_local_scale_ratio if depth_local_scale_ratio is not None else (vec_curr.length / vec_start.length if vec_start.length > 0 else 1.0)
                cam_mat_curr = evaluated_matrix_world(context, cam_ref)
                cam_loc_curr = cam_mat_curr.translation
                vec_cam_to_obj = target.location - cam_loc_curr
                current_obj_depth = vec_cam_to_obj.length
                target_depth = current_obj_depth / scale_ratio if scale_ratio > 1e-6 else current_obj_depth
                if vec_cam_to_obj.length > 0:
                    target.location = cam_loc_curr + vec_cam_to_obj.normalized() * target_depth
                target.scale = init_t_scale
                
            else: # CAMERA
                dist_start = vec_start.length
                dist_curr = vec_curr.length
                scale_ratio = dist_curr / dist_start if dist_start > 0 else 1.0
                
                if props.tripod_mode:
                    target.location = init_t_loc
                    if props.scale_mode == 'NONE':
                        tan_x, tan_y = get_camera_tan(cam_ref.data, init_f_len, context.scene)
                        marker_ref_1 = get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_1, ref_f)
                        marker_ref_2 = get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_2, ref_f)
                        marker_cur_1 = get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_1, f)
                        marker_cur_2 = get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_2, f)
                        if any(marker is None for marker in (marker_ref_1, marker_ref_2, marker_cur_1, marker_cur_2)):
                            skip_counts["missing_marker"] += 1
                            continue
                        ray_ref_list = [
                            marker_to_camera_ray(marker_ref_1, tan_x, tan_y),
                            marker_to_camera_ray(marker_ref_2, tan_x, tan_y),
                        ]
                        ray_curr_list = [
                            marker_to_camera_ray(marker_cur_1, tan_x, tan_y),
                            marker_to_camera_ray(marker_cur_2, tan_x, tan_y),
                        ]
                        delta_quat = solve_tripod_rotation_from_rays(ray_ref_list, ray_curr_list, props.clip_lock_roll)
                        solved_quat = delta_quat @ init_t_rot
                        if props.clip_lock_roll:
                            solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                        self.set_target_rotation(target, solved_quat)
                    else:
                        solved_focal_lock_roll = False
                        if props.scale_mode == 'FOCAL_LENGTH' and props.clip_lock_roll:
                            ref_lens_for_rotation = existing_lens_curve.get(ref_f, init_f_len) if keep_existing_focal else init_f_len
                            if keep_existing_focal:
                                frame_lens_for_rotation = existing_lens_curve.get(f, ref_lens_for_rotation)
                            elif suppress_focal_bake:
                                frame_lens_for_rotation = init_f_len
                            else:
                                frame_lens_for_rotation = init_f_len * scale_ratio
                            delta_quat = solve_focal_tripod_lock_roll_from_markers(
                                context,
                                cam_ref.data,
                                props.target_clip,
                                props.tracking_object_idx,
                                [props.track_1, props.track_2],
                                ref_f,
                                f,
                                ref_lens_for_rotation,
                                frame_lens_for_rotation,
                            )
                            if delta_quat is not None:
                                target.location = init_t_loc
                                solved_quat = preserve_camera_roll_from_reference(init_t_rot @ delta_quat, init_t_rot)
                                self.set_target_rotation(target, solved_quat)
                                solved_focal_lock_roll = True

                        if not solved_focal_lock_roll:
                            center_start = (p1_start + p2_start) / 2.0
                            center_curr = (p1_curr + p2_curr) / 2.0
                            vec_pt_start = center_start - target.location
                            init_cam_matrix_inv = init_t_mat.inverted()
                            center_local_curr = init_cam_matrix_inv @ center_curr
                            center_local_curr_unzoomed = Vector((
                                center_local_curr.x / scale_ratio if scale_ratio > 1e-6 else center_local_curr.x,
                                center_local_curr.y / scale_ratio if scale_ratio > 1e-6 else center_local_curr.y,
                                center_local_curr.z
                            ))
                            center_curr_unzoomed = init_t_mat @ center_local_curr_unzoomed
                            vec_pt_curr = center_curr_unzoomed - target.location
                            pan_tilt_quat = Quaternion()
                            if vec_pt_start.length_squared > 1e-9 and vec_pt_curr.length_squared > 1e-9:
                                pan_tilt_quat = vec_pt_start.rotation_difference(vec_pt_curr)
                            
                            vec_start_panned = pan_tilt_quat.inverted() @ vec_start
                            roll_quat = vec_start_panned.rotation_difference(vec_curr)
                            
                            view_axis = vec_pt_curr.normalized()
                            try:
                                swing, twist = roll_quat.to_swing_twist(view_axis)
                            except Exception:
                                twist = Quaternion()
                            if props.clip_lock_roll:
                                twist = Quaternion()
                            
                            total_delta_quat = twist @ pan_tilt_quat
                            stabilize_quat = total_delta_quat.inverted()
                            
                            solved_quat = stabilize_quat @ init_t_mat.to_quaternion()
                            if props.clip_lock_roll:
                                solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                            self.set_target_rotation(target, solved_quat)
                    
                else: # Non-Tripod
                    center_start = (p1_start + p2_start) / 2.0
                    center_curr = (p1_curr + p2_curr) / 2.0
                    
                    init_rot_quat, init_cam_rot_mat = init_t_mat.to_quaternion(), init_t_mat.to_3x3()
                    
                    vec_start_local = init_cam_rot_mat.inverted() @ vec_start
                    vec_curr_local = init_cam_rot_mat.inverted() @ vec_curr
                    if vec_curr_local.length_squared == 0:
                        skip_counts["zero_local_pair"] += 1
                        continue
                    
                    scale_ratio_for_pan = vec_curr_local.length / vec_start_local.length if vec_start_local.length > 0 else 1.0
                    angle_start_2d = math.atan2(vec_start_local.y, vec_start_local.x)
                    angle_curr_2d = math.atan2(vec_curr_local.y, vec_curr_local.x)
                    delta_angle = angle_curr_2d - angle_start_2d
                    
                    axis = init_rot_quat @ Vector((0,0,1))
                    if props.clip_lock_roll:
                        delta_angle = 0.0
                    correction_quat = Quaternion(axis, -delta_angle)
                    self.set_target_rotation(target, correction_quat @ init_rot_quat)
                    
                    init_cam_matrix_inv = init_t_mat.inverted()
                    center_start_local = init_cam_matrix_inv @ center_start
                    center_curr_local = init_cam_matrix_inv @ center_curr
                    center_curr_local_unzoomed = Vector((
                        center_curr_local.x / scale_ratio_for_pan if scale_ratio_for_pan > 1e-6 else center_curr_local.x,
                        center_curr_local.y / scale_ratio_for_pan if scale_ratio_for_pan > 1e-6 else center_curr_local.y,
                        center_curr_local.z
                    ))
                    pan_unscaled_local = center_curr_local_unzoomed - center_start_local
                    
                    rot_inv_mat = Matrix.Rotation(-delta_angle, 3, 'Z')
                    pan_true_local = rot_inv_mat @ pan_unscaled_local
                    pan_offset_world = init_cam_rot_mat @ pan_true_local
                    
                    target.location = init_t_loc - pan_offset_world
                    
                    if props.lock_camera_z and props.scale_mode == 'Z_DEPTH':
                        c_c = (p1_start + p2_start) / 2.0
                        loc, rot_mat = apply_z_lock(target.location, self.get_target_rotation_matrix(target), c_c, init_t_loc.z)
                        target.location = loc
                        self.set_target_rotation(target, rot_mat)

                if props.scale_mode == 'Z_DEPTH':
                    center_start = (p1_start + p2_start) / 2.0
                    depth_start = (center_start - init_t_loc).length
                    depth_curr = depth_start / scale_ratio if scale_ratio > 1e-6 else depth_start
                    if props.tripod_mode:
                        view_dir = self.get_target_rotation_quaternion(target) @ Vector((0,0,-1))
                    else:
                        view_dir = init_t_mat.to_quaternion() @ Vector((0,0,-1))
                    target.location -= view_dir * (depth_curr - depth_start)
                elif props.scale_mode == 'FOCAL_LENGTH':
                    if not keep_existing_focal and not suppress_focal_bake:
                        target.data.lens = init_f_len * scale_ratio
                        target.data.keyframe_insert(data_path="lens", frame=f)

            if keep_existing_position and not is_obj:
                existing_location = existing_loc_curve.get(f, init_t_loc.copy()).copy()
                target.location = existing_location
            if keep_existing_focal and not is_obj and props.scale_mode == 'FOCAL_LENGTH':
                target.data.lens = existing_lens_curve.get(f, init_f_len)

            skip_rotation_refit = props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH' and props.clip_lock_roll
            if not is_obj and not skip_rotation_refit and (
                keep_existing_position or
                keep_existing_focal or
                props.scale_mode == 'Z_DEPTH'
            ):
                if props.scale_mode == 'Z_DEPTH' and props.lock_camera_z:
                    target.location.z = init_t_loc.z
                fallback_quat = self.get_target_rotation_quaternion(target)
                ray_origin_loc = init_t_loc
                refined_quat = solve_track_rotation_from_follow_points(
                    [props.track_1, props.track_2],
                    fixed_world_points,
                    {props.track_1: p1_curr, props.track_2: p2_curr},
                    target.location.copy(),
                    ray_origin_loc,
                    init_t_rot,
                    fallback_quat,
                    props.clip_lock_roll,
                    prefer_center=keep_existing_position,
                )
                if refined_quat is None:
                    refined_quat = fallback_quat
                if props.clip_lock_roll:
                    refined_quat = preserve_camera_roll_from_reference(refined_quat, init_t_rot)
                self.set_target_rotation(target, refined_quat)

            if not keep_existing_position:
                target.keyframe_insert("location", frame=f)
            self.keyframe_target_rotation(target, f)
            if is_obj:
                target.keyframe_insert("scale", frame=f)
            baked_frames += 1

        if baked_frames == 0:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, f"No frames could be baked from the selected 2-point trackers. Skips: {format_skip_reasons(skip_counts)}.")
            return {'CANCELLED'}

        if keep_existing_position and not is_obj:
            self.restore_animation_curves(target, location_curve_snapshot)
        if keep_existing_focal and not is_obj and getattr(target, "data", None):
            self.restore_animation_action_copy(target.data, lens_action_copy)
        elif pin_existing_focal_range and getattr(target, "data", None):
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)
            
        context.scene.frame_set(ref_f)
        total_frames = frame_end - frame_start + 1
        suffix = f" Solved {baked_frames}/{total_frames} frames." if baked_frames < total_frames else ""
        self.report({'INFO'}, f"Applied 2-point motion to '{target.name}'.{suffix}")
        return {'FINISHED'}

    def execute_three_point(self, context, target):
        props = context.scene.pcam_solve_props
        if props.track_1 == "NONE" or props.track_2 == "NONE" or props.track_3 == "NONE":
            self.report({'ERROR'}, "Tracks missing.")
            return {'CANCELLED'}
        if len({props.track_1, props.track_2, props.track_3}) < 3:
            self.report({'ERROR'}, "Track 1, Track 2, and Track 3 must all be different.")
            return {'CANCELLED'}
            
        is_obj = (props.apply_to == 'OBJECT')
        cam_ref = context.scene.camera
        if not is_obj and props.tripod_mode and props.scale_mode == 'NONE':
            return self.execute_marker_tripod_none(context, target, [props.track_1, props.track_2, props.track_3], "3-point")
        eff_scale_mode = 'Z_DEPTH' if is_obj and props.scale_mode != 'NONE' else props.scale_mode
        eff_depth_obj = props.clip_depth_object if props.clip_depth_object else (target if is_obj else None)

        clip = props.target_clip
        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        follow_cam = self.create_static_follow_camera(context, cam_ref, evaluated_matrix_world(context, cam_ref)) if cam_ref and not is_obj else None

        frame_range = (props.bake_start, props.bake_end) if props.use_custom_range else None
        keep_existing_position = (not is_obj) and props.clip_use_existing_position and not (props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH')
        lens_curve_snapshot = self.snapshot_animation_action(target.data) if (not is_obj and getattr(target, "data", None) is not None) else []
        has_existing_focal_keys = (not is_obj) and self.has_camera_focal_length_keys(cam_ref)
        use_existing_focal = props.clip_use_existing_focal or (keep_existing_position and props.scale_mode == 'FOCAL_LENGTH')
        keep_existing_focal = (not is_obj) and props.scale_mode == 'FOCAL_LENGTH' and use_existing_focal and has_existing_focal_keys
        suppress_focal_bake = (not is_obj) and props.scale_mode == 'FOCAL_LENGTH' and use_existing_focal and not has_existing_focal_keys
        has_focal_variation_in_range = (not is_obj) and frame_range is not None and self.camera_lens_varies_over_range(context, cam_ref, frame_range[0], frame_range[1])
        pin_existing_focal_range = frame_range is not None and not is_obj and props.scale_mode != 'FOCAL_LENGTH' and not keep_existing_focal and (has_existing_focal_keys or has_focal_variation_in_range)
        target_curve_snapshot = self.snapshot_animation_action(target)
        existing_loc_curve = {}
        existing_lens_curve = {}
        location_curve_snapshot = self.snapshot_animation_curves(target, {"location"}) if keep_existing_position else []
        lens_action_copy = self.copy_animation_action(target.data) if keep_existing_focal and getattr(target, "data", None) else None
        if keep_existing_position or keep_existing_focal:
            restore_frame = context.scene.frame_current
            for f in range(frame_start, frame_end + 1):
                context.scene.frame_set(f)
                if keep_existing_position:
                    existing_loc_curve[f] = target.location.copy()
                if keep_existing_focal:
                    existing_lens_curve[f] = float(target.data.lens)
            context.scene.frame_set(restore_frame)
        pinned_lens_value = float(target.data.lens) if (not is_obj and getattr(target, "data", None) is not None) else None
        if frame_range is None:
            self.clear_animation_safely(
                target,
                None,
                keep_target_paths=None,
                keep_data_paths={"lens"} if keep_existing_focal else None,
            )
        else:
            self.clear_animation_safely(
                target,
                frame_range,
                keep_target_paths=None,
                keep_data_paths={"lens"} if keep_existing_focal else None,
            )
        if pin_existing_focal_range and getattr(target, "data", None):
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)

        extract_cam = follow_cam or cam_ref
        try:
            t_d = self.extract_tracks_data(
                context,
                extract_cam,
                props.target_clip,
                [props.track_1, props.track_2, props.track_3],
                eff_depth_obj,
                props.use_undistort,
                props.track_smoothing,
            )
            valid_f = sorted(set(t_d[0].keys()) & set(t_d[1].keys()) & set(t_d[2].keys()))
            area_f = frames_with_triangle_area(t_d, valid_f)
        finally:
            self.remove_static_follow_camera(follow_cam)
        if not valid_f:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, "No frames with all selected trackers were found in the bake range.")
            return {'CANCELLED'}
        if not area_f:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, f"Follow Track evaluation produced degenerate 3-point positions in the bake range. Max area: {max_triangle_area_metric(t_d, valid_f):.6g}.")
            return {'CANCELLED'}
            
        ref_f = pcam_pick_valid_reference_frame(area_f, ref_hint, props.use_reference_frame_lock)
        if ref_f is None:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, "Reference Frame has no valid 3-point tracker area.")
            return {'CANCELLED'}
        context.scene.frame_set(ref_f)
        init_t_mat = target.matrix_world.copy()
        if not is_obj:
            init_t_mat = matrix_without_scale(init_t_mat)
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        init_t_scale = target.scale.copy()
        
        init_f_len = cam_ref.data.lens if cam_ref else 35.0
        points_start_ref = [t_d[0][ref_f], t_d[1][ref_f], t_d[2][ref_f]]
        fixed_world_points = {
            props.track_1: points_start_ref[0].copy(),
            props.track_2: points_start_ref[1].copy(),
            props.track_3: points_start_ref[2].copy(),
        }
        depth_ref_mat = evaluated_matrix_world(context, props.clip_depth_object) if is_obj and props.clip_depth_object else None
        depth_ref_inv = depth_ref_mat.inverted() if depth_ref_mat is not None else None
        depth_ref_quat_inv = depth_ref_mat.to_quaternion().inverted() if depth_ref_mat is not None else None
        points_start_ref_local = [depth_ref_inv @ point for point in points_start_ref] if depth_ref_inv is not None else []
        basis_from_depth_local = build_triangle_basis(points_start_ref_local) if points_start_ref_local else None
        prev_obj_quat = init_t_rot.copy()
        prev_obj_euler = init_t_rot.to_euler(target.rotation_mode) if target.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'} else None
        baked_frames = 0
        skip_counts = {
            "bad_ref_basis": 0,
            "bad_frame_basis": 0,
            "zero_focal_view": 0,
            "missing_marker": 0,
        }

        for f in valid_f:
            context.scene.frame_set(f)
            points_start = points_start_ref
            points_curr = [t_d[0][f], t_d[1][f], t_d[2][f]]
            
            centroid_from = sum(points_start, Vector()) / 3.0
            centroid_to = sum(points_curr, Vector()) / 3.0
            
            basis_from = build_triangle_basis(points_start)
            if basis_from is None:
                skip_counts["bad_ref_basis"] += 1
                continue
            
            basis_to = build_triangle_basis(points_curr)
            if basis_to is None:
                skip_counts["bad_frame_basis"] += 1
                continue
            rot_quat = (basis_to @ basis_from.inverted()).to_quaternion()
            
            avg_dist_from = sum([(p - centroid_from).length for p in points_start]) / 3.0
            avg_dist_to = sum([(p - centroid_to).length for p in points_curr]) / 3.0
            scale = avg_dist_to / avg_dist_from if avg_dist_from > 0 else 1.0
            edge_scale_ratios = []
            for i1, i2 in ((0, 1), (1, 2), (2, 0)):
                edge_from = (points_start[i2] - points_start[i1]).length
                edge_to = (points_curr[i2] - points_curr[i1]).length
                if edge_from > 1e-6:
                    edge_scale_ratios.append(edge_to / edge_from)
            if edge_scale_ratios:
                edge_scale_ratios.sort()
                scale_ratio_cam = edge_scale_ratios[len(edge_scale_ratios) // 2]
            else:
                scale_ratio_cam = scale

            mat_trans_to_origin = Matrix.Translation(-centroid_from)
            mat_scale = Matrix.Scale(scale, 4)
            mat_rot = rot_quat.to_matrix().to_4x4()
            mat_trans_from_origin = Matrix.Translation(centroid_to)
            
            transform_matrix = mat_trans_from_origin @ mat_rot @ mat_scale @ mat_trans_to_origin
            transform_matrix_noscale = mat_trans_from_origin @ mat_rot @ mat_trans_to_origin

            if is_obj:
                center_start = centroid_from
                center_curr = centroid_to
                target.location = init_t_loc + (center_curr - center_start)
                scale_ratio = scale
                depth_curr_mat = None
                depth_curr_inv = None
                points_curr_local = None
                if depth_ref_mat is not None:
                    depth_curr_mat = evaluated_matrix_world(context, props.clip_depth_object)
                    depth_curr_inv = depth_curr_mat.inverted()
                    points_curr_local = [depth_curr_inv @ point for point in points_curr]
                    local_scale = median_edge_scale(points_start_ref_local, points_curr_local)
                    if local_scale is None:
                        local_dist_from = point_cloud_avg_distance(points_start_ref_local)
                        local_dist_to = point_cloud_avg_distance(points_curr_local)
                        local_scale = local_dist_to / local_dist_from if local_dist_from > 1e-6 else None
                if local_scale is not None and local_scale > 1e-6:
                    scale_ratio = local_scale
                cam_mat_curr = evaluated_matrix_world(context, cam_ref)
                cam_loc_curr = cam_mat_curr.translation
                vec_cam_to_obj = target.location - cam_loc_curr
                current_obj_depth = vec_cam_to_obj.length
                target_depth = current_obj_depth / scale_ratio if scale_ratio > 1e-6 else current_obj_depth
                if vec_cam_to_obj.length > 0: 
                    target.location = cam_loc_curr + vec_cam_to_obj.normalized() * target_depth

                solved_obj_rotation = None
                if depth_ref_mat is not None and basis_from_depth_local is not None and depth_curr_mat is not None and depth_curr_inv is not None:
                    if points_curr_local is None:
                        points_curr_local = [depth_curr_inv @ point for point in points_curr]
                    basis_to_depth_local = build_triangle_basis(points_curr_local)
                    if basis_to_depth_local is not None:
                        local_delta_quat = (basis_to_depth_local @ basis_from_depth_local.inverted()).to_quaternion()
                        solved_obj_rotation = depth_curr_mat.to_quaternion() @ local_delta_quat @ depth_ref_quat_inv @ init_t_rot

                if solved_obj_rotation is None:
                    cam_inv = cam_mat_curr.inverted()
                    start_cam_points = [cam_inv @ p for p in points_start]
                    curr_cam_points = [cam_inv @ p for p in points_curr]
                    start_cam_centroid = sum(start_cam_points, Vector()) / 3.0
                    curr_cam_centroid = sum(curr_cam_points, Vector()) / 3.0
                    start_roll_points = [Vector((p.x - start_cam_centroid.x, p.y - start_cam_centroid.y)) for p in start_cam_points]
                    curr_roll_points = [Vector((p.x - curr_cam_centroid.x, p.y - curr_cam_centroid.y)) for p in curr_cam_points]
                    roll_delta = solve_planar_roll_from_points(start_roll_points, curr_roll_points)
                    if vec_cam_to_obj.length > 1e-9:
                        view_axis = vec_cam_to_obj.normalized()
                        solved_obj_rotation = Quaternion(view_axis, roll_delta) @ init_t_rot

                if solved_obj_rotation is not None:
                    solved_quat = self.set_target_rotation_continuous(
                        target,
                        solved_obj_rotation,
                        prev_obj_quat,
                        prev_obj_euler,
                    )
                    prev_obj_quat = solved_quat.copy()
                    if prev_obj_euler is not None:
                        prev_obj_euler = target.rotation_euler.copy()
                target.scale = init_t_scale
            else: # CAMERA
                scale_ratio = scale_ratio_cam if scale_ratio_cam > 1e-6 else 1.0
                init_cam_inv = init_t_mat.inverted()
                init_cam_rot_mat = init_t_mat.to_3x3()

                if props.scale_mode == 'FOCAL_LENGTH':
                    points_curr_unzoomed = []
                    for p in points_curr:
                        p_local = init_cam_inv @ p
                        points_curr_unzoomed.append(init_t_mat @ Vector((
                            p_local.x / scale_ratio if scale_ratio > 1e-6 else p_local.x,
                            p_local.y / scale_ratio if scale_ratio > 1e-6 else p_local.y,
                            p_local.z
                        )))

                    if props.tripod_mode:
                        target.location = init_t_loc
                        solved_focal_lock_roll = False
                        if props.clip_lock_roll:
                            ref_lens_for_rotation = existing_lens_curve.get(ref_f, init_f_len) if keep_existing_focal else init_f_len
                            if keep_existing_focal:
                                frame_lens_for_rotation = existing_lens_curve.get(f, ref_lens_for_rotation)
                            elif suppress_focal_bake:
                                frame_lens_for_rotation = init_f_len
                            else:
                                frame_lens_for_rotation = init_f_len * scale_ratio
                            delta_quat = solve_focal_tripod_lock_roll_from_markers(
                                context,
                                cam_ref.data,
                                props.target_clip,
                                props.tracking_object_idx,
                                [props.track_1, props.track_2, props.track_3],
                                ref_f,
                                f,
                                ref_lens_for_rotation,
                                frame_lens_for_rotation,
                            )
                            if delta_quat is not None:
                                solved_quat = preserve_camera_roll_from_reference(init_t_rot @ delta_quat, init_t_rot)
                                self.set_target_rotation(target, solved_quat)
                                solved_focal_lock_roll = True

                        if not solved_focal_lock_roll:
                            centroid_curr_unzoomed = sum(points_curr_unzoomed, Vector()) / 3.0
                            vec_pt_start = centroid_from - init_t_loc
                            vec_pt_curr_unzoomed = centroid_curr_unzoomed - init_t_loc
                            if vec_pt_start.length_squared < 1e-9 or vec_pt_curr_unzoomed.length_squared < 1e-9:
                                skip_counts["zero_focal_view"] += 1
                                continue
                            pan_tilt_quat = vec_pt_start.rotation_difference(vec_pt_curr_unzoomed)
                            view_axis = vec_pt_curr_unzoomed.normalized()
                            delta_roll = average_twist_roll_angle(
                                triangle_edges(points_start),
                                triangle_edges(points_curr_unzoomed),
                                view_axis,
                                ref_align_quat=pan_tilt_quat,
                            )
                            twist = Quaternion(view_axis, delta_roll)
                            if props.clip_lock_roll:
                                twist = Quaternion()
                            total_delta_quat = twist @ pan_tilt_quat
                            solved_quat = total_delta_quat.inverted() @ init_t_rot
                            if props.clip_lock_roll:
                                solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                            self.set_target_rotation(target, solved_quat)
                    else:
                        centroid_start_local = init_cam_inv @ centroid_from
                        centroid_curr_local = init_cam_inv @ centroid_to
                        centroid_curr_local_unzoomed = Vector((
                            centroid_curr_local.x / scale_ratio if scale_ratio > 1e-6 else centroid_curr_local.x,
                            centroid_curr_local.y / scale_ratio if scale_ratio > 1e-6 else centroid_curr_local.y,
                            centroid_curr_local.z
                        ))
                        start_local_points = [init_cam_inv @ p for p in points_start]
                        curr_local_points = [init_cam_inv @ p for p in points_curr]
                        edge_start_local = triangle_edges(start_local_points)
                        edge_curr_local_unzoomed = []
                        for edge_curr_local in triangle_edges(curr_local_points):
                            edge_curr_local_unzoomed.append(Vector((
                                edge_curr_local.x / scale_ratio if scale_ratio > 1e-6 else edge_curr_local.x,
                                edge_curr_local.y / scale_ratio if scale_ratio > 1e-6 else edge_curr_local.y,
                                edge_curr_local.z
                            )))
                        roll_delta = average_planar_roll_delta(edge_start_local, edge_curr_local_unzoomed)
                        if props.clip_lock_roll:
                            roll_delta = 0.0
                        axis = init_t_rot @ Vector((0, 0, 1))
                        correction_quat = Quaternion(axis, -roll_delta)
                        self.set_target_rotation(target, correction_quat @ init_t_rot)
                        pan_true_local = centroid_curr_local_unzoomed - centroid_start_local
                        target.location = init_t_loc - (init_cam_rot_mat @ pan_true_local)
                        if props.lock_camera_z and props.scale_mode == 'Z_DEPTH':
                            loc, rot_mat = apply_z_lock(target.location, self.get_target_rotation_matrix(target), centroid_from, init_t_loc.z)
                            target.location = loc
                            self.set_target_rotation(target, rot_mat)

                    if not keep_existing_focal and not suppress_focal_bake:
                        target.data.lens = init_f_len * scale_ratio
                        target.data.keyframe_insert(data_path="lens", frame=f)
                else:
                    if props.tripod_mode:
                        if props.scale_mode == 'NONE':
                            tan_x, tan_y = get_camera_tan(cam_ref.data, init_f_len, context.scene)
                            marker_ref_list = [
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_1, ref_f),
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_2, ref_f),
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_3, ref_f),
                            ]
                            marker_curr_list = [
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_1, f),
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_2, f),
                                get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_3, f),
                            ]
                            if any(marker is None for marker in marker_ref_list + marker_curr_list):
                                skip_counts["missing_marker"] += 1
                                continue
                            ray_ref_list = [marker_to_camera_ray(marker, tan_x, tan_y) for marker in marker_ref_list]
                            ray_curr_list = [marker_to_camera_ray(marker, tan_x, tan_y) for marker in marker_curr_list]
                            delta_quat = solve_tripod_rotation_from_rays(ray_ref_list, ray_curr_list, props.clip_lock_roll)
                        else:
                            delta_quat = rot_quat
                            direct_quat = None
                            if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
                                direct_quat = solve_track_rotation_from_follow_points(
                                    [props.track_1, props.track_2, props.track_3],
                                    fixed_world_points,
                                    {
                                        props.track_1: points_curr[0],
                                        props.track_2: points_curr[1],
                                        props.track_3: points_curr[2],
                                    },
                                    init_t_loc,
                                    init_t_loc,
                                    init_t_rot,
                                    init_t_rot,
                                    props.clip_lock_roll,
                                )
                            elif props.clip_lock_roll:
                                vec_pt_start = centroid_from - init_t_loc
                                vec_pt_curr = centroid_to - init_t_loc
                                if vec_pt_start.length_squared > 1e-9 and vec_pt_curr.length_squared > 1e-9:
                                    delta_quat = vec_pt_start.rotation_difference(vec_pt_curr)
                            target.location = init_t_loc
                            if direct_quat is not None:
                                if props.clip_lock_roll:
                                    direct_quat = preserve_camera_roll_from_reference(direct_quat, init_t_rot)
                                self.set_target_rotation(target, direct_quat)
                            elif props.scale_mode == 'NONE':
                                solved_quat = delta_quat @ init_t_rot
                                if props.clip_lock_roll:
                                    solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                                self.set_target_rotation(target, solved_quat)
                            else:
                                solved_quat = delta_quat.inverted() @ init_t_rot
                                if props.clip_lock_roll:
                                    solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                                self.set_target_rotation(target, solved_quat)
                    else:
                        loc_rot_matrix = transform_matrix_noscale.copy()
                        loc_rot_matrix.normalize()
                        
                        delta_quat = loc_rot_matrix.to_quaternion()
                        if props.clip_lock_roll:
                            vec_pt_start = centroid_from - init_t_loc
                            vec_pt_curr = centroid_to - init_t_loc
                            if vec_pt_start.length_squared > 1e-9 and vec_pt_curr.length_squared > 1e-9:
                                delta_quat = vec_pt_start.rotation_difference(vec_pt_curr)
                        loc_rot_matrix = Matrix.Translation(loc_rot_matrix.to_translation()) @ delta_quat.to_matrix().to_4x4()
                        
                        stabilize_matrix = loc_rot_matrix.inverted()
                        new_matrix = stabilize_matrix @ init_t_mat
                        
                        loc, rot, sca = new_matrix.decompose()
                        self.set_target_rotation(target, rot)
                        centroid_start_local = init_cam_inv @ centroid_from
                        centroid_curr_local = init_cam_inv @ centroid_to
                        centroid_curr_local_unzoomed = Vector((
                            centroid_curr_local.x / scale_ratio if scale_ratio > 1e-6 else centroid_curr_local.x,
                            centroid_curr_local.y / scale_ratio if scale_ratio > 1e-6 else centroid_curr_local.y,
                            centroid_curr_local.z
                        ))
                        pan_true_local = centroid_curr_local_unzoomed - centroid_start_local
                        target.location = init_t_loc - (init_cam_rot_mat @ pan_true_local)
                        
                        if props.lock_camera_z and props.scale_mode == 'Z_DEPTH':
                            loc, rot_mat = apply_z_lock(target.location, self.get_target_rotation_matrix(target), centroid_from, init_t_loc.z)
                            target.location = loc
                            self.set_target_rotation(target, rot_mat)

                    if props.lock_camera_z and props.scale_mode == 'Z_DEPTH':
                        loc, rot_mat = apply_z_lock(target.location, self.get_target_rotation_matrix(target), centroid_from, init_t_loc.z)
                        target.location = loc
                        self.set_target_rotation(target, rot_mat)

                    if props.scale_mode == 'Z_DEPTH':
                        depth_start = (centroid_from - init_t_mat.to_translation()).length
                        depth_curr = depth_start / scale_ratio if scale_ratio > 1e-6 else depth_start
                        if props.tripod_mode:
                            view_dir = self.get_target_rotation_quaternion(target) @ Vector((0,0,-1))
                        else:
                            view_dir = init_t_mat.to_quaternion() @ Vector((0,0,-1))
                        target.location -= view_dir * (depth_curr - depth_start)
                    
            if keep_existing_position and not is_obj:
                existing_location = existing_loc_curve.get(f, init_t_loc.copy()).copy()
                target.location = existing_location
            if keep_existing_focal and not is_obj and props.scale_mode == 'FOCAL_LENGTH':
                target.data.lens = existing_lens_curve.get(f, init_f_len)

            skip_rotation_refit = props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH' and props.clip_lock_roll
            if not is_obj and not skip_rotation_refit and (
                keep_existing_position or
                keep_existing_focal or
                props.scale_mode == 'Z_DEPTH'
            ):
                if props.scale_mode == 'Z_DEPTH' and props.lock_camera_z:
                    target.location.z = init_t_loc.z
                fallback_quat = self.get_target_rotation_quaternion(target)
                ray_origin_loc = init_t_loc
                refined_quat = solve_track_rotation_from_follow_points(
                    [props.track_1, props.track_2, props.track_3],
                    fixed_world_points,
                    {
                        props.track_1: points_curr[0],
                        props.track_2: points_curr[1],
                        props.track_3: points_curr[2],
                    },
                    target.location.copy(),
                    ray_origin_loc,
                    init_t_rot,
                    fallback_quat,
                    props.clip_lock_roll,
                    prefer_center=keep_existing_position,
                )
                if refined_quat is None:
                    refined_quat = fallback_quat
                if props.clip_lock_roll:
                    refined_quat = preserve_camera_roll_from_reference(refined_quat, init_t_rot)
                self.set_target_rotation(target, refined_quat)

            if not keep_existing_position:
                target.keyframe_insert("location", frame=f)
            self.keyframe_target_rotation(target, f)
            if is_obj:
                target.keyframe_insert("scale", frame=f)
            baked_frames += 1

        if baked_frames == 0:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if not is_obj and getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if lens_action_copy is not None:
                bpy.data.actions.remove(lens_action_copy)
            self.report({'ERROR'}, f"No frames could be baked from the selected 3-point trackers. Skips: {format_skip_reasons(skip_counts)}.")
            return {'CANCELLED'}

        if keep_existing_position and not is_obj:
            self.restore_animation_curves(target, location_curve_snapshot)
        if keep_existing_focal and not is_obj and getattr(target, "data", None):
            self.restore_animation_action_copy(target.data, lens_action_copy)
        elif pin_existing_focal_range and getattr(target, "data", None):
            self.pin_lens_constant_in_range(target.data, frame_range[0], frame_range[1], pinned_lens_value, lens_curve_snapshot)
            
        context.scene.frame_set(ref_f)
        total_frames = frame_end - frame_start + 1
        suffix = f" Solved {baked_frames}/{total_frames} frames." if baked_frames < total_frames else ""
        self.report({'INFO'}, f"Applied 3-point motion to '{target.name}'.{suffix}")
        return {'FINISHED'}

# --- UI Panel ---

class VIEW3D_PT_pcam_solve_panel(bpy.types.Panel):
    bl_label = "Pseudo Camera Solver"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'P-Cam' 
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.pcam_solve_props
        is_cam = props.apply_to == 'CAMERA'

        def labeled_prop(container, data, prop_name, label, factor=0.42):
            split = container.split(factor=factor, align=True)
            split.label(text=label)
            split.prop(data, prop_name, text="")

        b_target = layout.box()
        b_target.label(text="Mode & Target", icon='OUTLINER_OB_CAMERA')
        labeled_prop(b_target, props, "mode", "Mode")
        row = b_target.row(align=True)
        row.prop(props, "apply_to", expand=True)
        
        if is_cam:
            labeled_prop(b_target, context.scene, "camera", "Active Camera")
        else:
            labeled_prop(b_target, props, "target_object", "Target Object")
            labeled_prop(b_target, context.scene, "camera", "Ref Camera")
             
        if pcam_depth_reference_required(props):
            row = b_target.row(align=True)
            split = row.split(factor=0.42, align=True)
            split.label(text="Depth Reference")
            split.prop(props, "clip_depth_object", text="")
            row.operator(OBJECT_OT_add_pcam_solve_depth_plane.bl_idname, text="", icon='ADD')

        b_clip = layout.box()
        b_clip.label(text="Tracker Reference")
        b_track = b_clip.box()
        b_track.label(text="Tracker Setup", icon='CON_FOLLOWTRACK')
        row = b_track.row(align=True)
        split = row.split(factor=0.42, align=True)
        split.label(text="Movie Clip")
        split.prop(props, "target_clip", text="")
        row.operator(OBJECT_OT_get_pcam_solve_selected_tracks.bl_idname, text="", icon='FILE_REFRESH')
        
        if props.target_clip: 
            labeled_prop(b_track, props, "tracking_object_idx", "Track Layer")
             
            if props.mode in {'ONE_POINT', 'TWO_POINT', 'THREE_POINT'}:
                b_track.prop(props, "use_undistort")
                tr = b_track.box()
                try: track_pool = props.target_clip.tracking.objects[int(props.tracking_object_idx)]
                except Exception: track_pool = None
                
                if track_pool:
                    tr.prop_search(props, "track_1", track_pool, "tracks")
                    if props.mode in {'TWO_POINT', 'THREE_POINT'}:
                        tr.prop_search(props, "track_2", track_pool, "tracks")
                    if props.mode == 'THREE_POINT':
                        tr.prop_search(props, "track_3", track_pool, "tracks")
                else:
                    tr.prop(props, "track_1")
                    if props.mode in {'TWO_POINT', 'THREE_POINT'}: tr.prop(props, "track_2")
                    if props.mode == 'THREE_POINT': tr.prop(props, "track_3")

        b_clip.prop(props, "track_preview", text="Preview Tracker Raycast", icon='RESTRICT_VIEW_OFF')
        if props.track_preview:
            b_p_settings = b_clip.box()
            b_p_settings.prop(props, "preview_point_size")
            row_colors = b_p_settings.row()
            row_colors.prop(props, "preview_color_hit", text="")
            row_colors.prop(props, "preview_color_miss", text="")
            row_colors.prop(props, "preview_color_line", text="")

        b_opt = layout.box()
        b_opt.label(text="Solve Settings", icon='TOOL_SETTINGS')
        tripod_label = "Dolly Motion" if props.mode != 'ONE_POINT' and props.scale_mode == 'Z_DEPTH' else "Tripod"
        if props.mode != 'ONE_POINT':
            if is_cam:
                labeled_prop(b_opt, props, "scale_mode", "Scale Method")
            else:
                b_opt.label(text="Scale mapped to Z-Depth.")

        if props.mode == 'CLIP_TRACK':
            if is_cam:
                row = b_opt.row(align=True)
                row.prop(props, "tripod_mode", text=tripod_label)
                if not props.tripod_mode and props.scale_mode == 'Z_DEPTH':
                    row.prop(props, "lock_camera_z", text="Lock Height")

                row = b_opt.row(align=True)
                row.prop(props, "track_smoothing", text="Smooth Jitter")
                row.prop(props, "clip_center_weight", text="Center Weighting")
            else:
                row = b_opt.row(align=True)
                row.prop(props, "track_smoothing", text="Smooth Jitter")
                row = b_opt.row(align=True)
                row.prop(props, "clip_center_weight", text="Center Weighting")
             
        elif props.mode == 'ONE_POINT':
            if is_cam:
                row = b_opt.row(align=True)
                row.prop(props, "tripod_mode", text="Tripod")
                row.prop(props, "track_smoothing", text="Smooth Jitter")
            else:
                b_opt.prop(props, "track_smoothing", text="Smooth Jitter")
                 
        elif props.mode in ('TWO_POINT', 'THREE_POINT'):
            if is_cam:
                row = b_opt.row(align=True)
                row.prop(props, "tripod_mode", text=tripod_label)
                if not props.tripod_mode and props.scale_mode == 'Z_DEPTH':
                    row.prop(props, "lock_camera_z", text="Lock Height")
                row = b_opt.row(align=True)
                row.prop(props, "track_smoothing", text="Smooth Jitter")
                row.prop(props, "clip_lock_roll", text="Lock Roll")
            else:
                b_opt.prop(props, "track_smoothing", text="Smooth Jitter")

        if props.mode == 'CLIP_TRACK' and is_cam:
            b_opt.prop(props, "clip_position_smooth", text="Position Smooth")
            if props.scale_mode == 'FOCAL_LENGTH':
                b_opt.prop(props, "clip_focal_smooth", text="Focal Smooth")
            row = b_opt.row(align=True)
            row.prop(props, "clip_pan_tilt_smooth", text="Pan/Tilt Smooth")
            row.prop(props, "clip_roll_smooth", text="Roll Smooth")

        b_bake = layout.box()
        b_bake.label(text="Bake", icon='ACTION')
        row = b_bake.row(align=True)
        row.prop(props, "use_custom_range")
        if props.use_custom_range:
            preview_row = row.row(align=True)
            preview_row.prop(props, "custom_range_use_preview", text="", icon='PREVIEW_RANGE', toggle=True)
        if props.use_custom_range:
            row = b_bake.row(align=True)
            row.operator(OBJECT_OT_set_pcam_solve_bake_start.bl_idname, text="", icon='TRIA_LEFT_BAR')
            row.prop(props, "bake_start", text="In")
            row.prop(props, "bake_end", text="Out")
            row.operator(OBJECT_OT_set_pcam_solve_bake_end.bl_idname, text="", icon='TRIA_RIGHT_BAR')

        if props.mode == 'ONE_POINT' and is_cam:
            row = b_bake.row(align=True)
            row.prop(props, "clip_use_existing_position", text="Use Existing Position")
        elif props.mode == 'CLIP_TRACK' and is_cam:
            row = b_bake.row(align=True)
            pos_row = row.row(align=True)
            pos_row.enabled = not (props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH')
            pos_row.prop(props, "clip_use_existing_position", text="Use Existing Position")
            if props.scale_mode == 'FOCAL_LENGTH':
                row.prop(props, "clip_use_existing_focal", text="Use Existing Focal")
        elif props.mode in {'TWO_POINT', 'THREE_POINT'} and is_cam:
            row = b_bake.row(align=True)
            pos_row = row.row(align=True)
            use_existing_position_enabled = not (props.tripod_mode and props.scale_mode == 'FOCAL_LENGTH')
            pos_row.enabled = use_existing_position_enabled
            pos_row.prop(props, "clip_use_existing_position", text="Use Existing Position")
            if props.scale_mode == 'FOCAL_LENGTH':
                if use_existing_position_enabled and props.clip_use_existing_position:
                    focal_row = row.row(align=True)
                    focal_row.enabled = False
                    focal_row.label(text="Use Existing Focal", icon='CHECKBOX_HLT')
                else:
                    row.prop(props, "clip_use_existing_focal", text="Use Existing Focal")

        block_reason = pcam_get_bake_block_reason(context, props)
        row_bake = b_bake.row()
        row_bake.enabled = not block_reason
        row_bake.scale_y = 2.0 
        row_bake.operator(OBJECT_OT_apply_tracking_data.bl_idname, text="Bake Tracking to Target", icon='TRACKING')
        if block_reason:
            b_bake.label(text=block_reason, icon='ERROR')
        frame_start, frame_end = pcam_get_frame_range(props)
        ref_frame = pcam_get_reference_frame(context, props, frame_start, frame_end)
        ref_row = b_bake.row(align=True)
        split = ref_row.split(factor=0.88, align=True)
        left = split.row(align=True)
        if props.use_reference_frame_lock:
            left.prop(props, "reference_frame", text="Reference Frame")
        else:
            left.label(text=f"Reference Frame: {ref_frame}", icon='TIME')
        lock_row = split.row(align=True)
        lock_row.alignment = 'RIGHT'
        lock_row.prop(
            props,
            "use_reference_frame_lock",
            text="",
            icon='LOCKED' if props.use_reference_frame_lock else 'UNLOCKED',
            emboss=False,
        )

# --- Registration ---

classes = (
    PCamSolveProperties,
    OBJECT_OT_set_pcam_solve_bake_start,
    OBJECT_OT_set_pcam_solve_bake_end,
    OBJECT_OT_get_pcam_solve_selected_tracks,
    OBJECT_OT_add_pcam_solve_depth_plane,
    OBJECT_OT_apply_tracking_data,
    VIEW3D_PT_pcam_solve_panel
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pcam_solve_props = bpy.props.PointerProperty(type=PCamSolveProperties)

def unregister():
    global _handle_3d
    if _handle_3d:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle_3d, 'WINDOW')
        except Exception:
            pass
        _handle_3d = None
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    if hasattr(bpy.types.Scene, "pcam_solve_props"):
        del bpy.types.Scene.pcam_solve_props
