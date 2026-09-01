# pcam_solver Clip Track solver
from dataclasses import dataclass

from .common import *


@dataclass
class _ClipRawPositionResult:
    frame_sets: dict
    dx_raw: dict
    dy_raw: dict
    pan_raw: dict
    depth_raw: dict
    zdepth_loc_raw: object
    zdepth_roll_curve: object
    zdepth_refined_world_points: object
    track_world_data: dict

class PCamClipTrackSolver:
    # Clip Track solvers.
    #
    # The refined path below is the current camera solve path. Object targets use
    # the fallback path in execute_clip_track().
    def _solve_clip_track_raw_position(
        self,
        context,
        cam_ref,
        clip,
        tracks,
        ref_f,
        frame_start,
        frame_end,
        full_frames,
        depth,
        norm_curve,
        frame_markers,
        eff_scale_mode,
        ref_markers,
        init_t_inv,
        init_t_loc,
        init_t_quat,
        depth_constraint_normal,
        tan_ref_x,
        tan_ref_y,
        aspect,
        fixed_world_points,
        restore_frame,
        rollback_animation,
    ):
        props = context.scene.pcam_solve_props
        frame_sets = {ref_f: set(ref_markers.keys())}
        dx_raw = {ref_f: 0.0}
        dy_raw = {ref_f: 0.0}
        pan_raw = {ref_f: Vector((0.0, 0.0, 0.0))}
        depth_raw = {ref_f: 0.0}
        zdepth_loc_raw = None
        zdepth_roll_curve = None
        zdepth_refined_world_points = None

        track_world_data = {}
        if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
            for track in tracks:
                world_data = self.extract_track_data(context, cam_ref, clip, track.name, props.clip_depth_object, True, props.track_smoothing)
                if world_data:
                    track_world_data[track.name] = world_data
            if not track_world_data:
                rollback_animation()
                context.scene.frame_set(restore_frame)
                self.report({'ERROR'}, "No depth track data extracted.")
                return None

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
                scale = norm_curve.get(frame, 1.0)
                depth_raw[frame] = ref_depth_dist * (1.0 - (1.0 / scale)) if scale > 1e-6 else depth_raw.get(frame, 0.0)

            if not props.tripod_mode:
                def adjacent_roll_delta(frame_a, frame_b):
                    markers_a = frame_markers.get(frame_a, {})
                    markers_b = frame_markers.get(frame_b, {})
                    shared = list(set(markers_a.keys()) & set(markers_b.keys()))
                    if len(shared) < 2:
                        return None
                    points_a = []
                    points_b = []
                    weights = []
                    for name in shared:
                        p_a = markers_a.get(name)
                        p_b = markers_b.get(name)
                        if p_a is None or p_b is None:
                            continue
                        points_a.append(p_a)
                        points_b.append(p_b)
                        weights.append(marker_center_weight(p_b, aspect) if props.clip_center_weight else 1.0)
                    if len(points_a) < 2:
                        return None
                    return solve_planar_roll_from_points(points_a, points_b, weights, aspect)

                zdepth_roll_curve = {ref_f: 0.0}
                current_roll = 0.0
                for frame in range(ref_f + 1, frame_end + 1):
                    delta_roll = adjacent_roll_delta(frame - 1, frame)
                    if delta_roll is not None:
                        current_roll += delta_roll
                    zdepth_roll_curve[frame] = current_roll
                current_roll = 0.0
                for frame in range(ref_f - 1, frame_start - 1, -1):
                    delta_roll = adjacent_roll_delta(frame, frame + 1)
                    if delta_roll is not None:
                        current_roll -= delta_roll
                    zdepth_roll_curve[frame] = current_roll
                zdepth_roll_curve = stabilize_roll_curve(zdepth_roll_curve, full_frames, despike_threshold_deg=1.2, smooth_blend=0.18)

                def zdepth_base_rotation(frame):
                    return init_t_quat @ Quaternion(Vector((0.0, 0.0, 1.0)), -zdepth_roll_curve.get(frame, 0.0))

                def solve_zdepth_location_from_seeded_points(frame, seeded_world_points, seeded_depths, seed_frames, last_seen_frames, trusted_tracks, trust_counts, fallback_loc):
                    scale = norm_curve.get(frame, 1.0)
                    if scale <= 1e-6:
                        return None, set()
                    rot3 = zdepth_base_rotation(frame).to_matrix()
                    depth_anchor_loc = init_t_loc + depth_constraint_normal * depth_raw.get(frame, 0.0)
                    def seed_track_at_location(track_name, marker_co, loc):
                        world_data = track_world_data.get(track_name)
                        seed_depth = ref_depth_dist
                        if world_data is not None and frame in world_data:
                            seed_depth = max(1e-4, -(init_t_inv @ world_data[frame]).z)
                        ray = marker_to_camera_ray(marker_co, tan_ref_x, tan_ref_y, cam_ref.data)
                        if abs(ray.z) <= 1e-9:
                            return
                        ray_world = rot3 @ ray
                        denom = depth_constraint_normal.dot(ray_world)
                        if abs(denom) > 1e-9:
                            t = depth_constraint_normal.dot(depth_anchor_loc - loc) / denom
                            if t > 1e-6:
                                seeded_world_points[track_name] = loc + ray_world * t
                            else:
                                current_depth = seed_depth / scale
                                point_local = ray * (current_depth / -ray.z)
                                seeded_world_points[track_name] = loc + (rot3 @ point_local)
                        else:
                            current_depth = seed_depth / scale
                            point_local = ray * (current_depth / -ray.z)
                            seeded_world_points[track_name] = loc + (rot3 @ point_local)
                        seeded_depths[track_name] = seed_depth
                        seed_frames[track_name] = frame
                        last_seen_frames[track_name] = frame
                        trust_counts.setdefault(track_name, 0)

                    points_world = []
                    rays_world = []
                    weights = []
                    names = set()
                    pending_seeds = []
                    used_trusted = set()
                    for track in tracks:
                        marker_co = frame_markers.get(frame, {}).get(track.name)
                        if marker_co is None:
                            continue
                        last_seen = last_seen_frames.get(track.name)
                        if last_seen is not None and abs(frame - last_seen) > 1:
                            seeded_world_points.pop(track.name, None)
                            seeded_depths.pop(track.name, None)
                            seed_frames.pop(track.name, None)
                            trusted_tracks.discard(track.name)
                            trust_counts.pop(track.name, None)
                        if track.name not in seeded_world_points:
                            pending_seeds.append((track.name, marker_co.copy()))
                            names.add(track.name)
                            continue
                        seed_world = seeded_world_points.get(track.name)
                        seed_depth = seeded_depths.get(track.name)
                        if seed_world is None or seed_depth is None:
                            continue
                        ray = marker_to_camera_ray(marker_co, tan_ref_x, tan_ref_y, cam_ref.data)
                        if ray.length_squared <= 1e-9:
                            continue
                        seed_frame = seed_frames.get(track.name)
                        if seed_frame is None:
                            blend_weight = 1.0
                        else:
                            age = abs(frame - seed_frame)
                            blend_weight = min(1.0, max(0.15, age / 5.0))
                        if track.name not in trusted_tracks:
                            trust_counts[track.name] = trust_counts.get(track.name, 0) + 1
                            if trust_counts[track.name] < 3:
                                blend_weight *= 0.35
                            else:
                                trusted_tracks.add(track.name)
                        points_world.append(seed_world)
                        rays_world.append(rot3 @ ray)
                        base_weight = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0
                        weights.append(base_weight * blend_weight)
                        names.add(track.name)
                        last_seen_frames[track.name] = frame
                        used_trusted.add(track.name)
                    if len(points_world) < 2:
                        for track_name, marker_co in pending_seeds:
                            seed_track_at_location(track_name, marker_co, fallback_loc)
                        return fallback_loc.copy(), names
                    solved_loc = solve_camera_location_on_depth_plane(points_world, rays_world, fallback_loc, depth_constraint_normal, depth_anchor_loc, weights)
                    for track_name, marker_co in pending_seeds:
                        seed_track_at_location(track_name, marker_co, solved_loc)
                        if len(used_trusted) >= 2:
                            trust_counts[track_name] = trust_counts.get(track_name, 0) + 1
                            if trust_counts[track_name] >= 5:
                                trusted_tracks.add(track_name)
                    return solved_loc, names

                zdepth_loc_raw = {ref_f: init_t_loc.copy()}
                seeded_forward = {}
                seeded_forward_depths = {}
                seeded_forward_frames = {}
                seeded_forward_last_seen = {}
                trusted_forward = set(fixed_world_points.keys())
                trust_counts_forward = {}
                for name, point in fixed_world_points.items():
                    seeded_forward[name] = point.copy()
                    seeded_forward_depths[name] = max(1e-4, -(init_t_inv @ point).z)
                    seeded_forward_frames[name] = None
                    seeded_forward_last_seen[name] = ref_f
                cur_loc = init_t_loc.copy()
                cur_set = frame_sets.get(ref_f, set()).copy()
                for frame in range(ref_f + 1, frame_end + 1):
                    solved_loc, shared = solve_zdepth_location_from_seeded_points(frame, seeded_forward, seeded_forward_depths, seeded_forward_frames, seeded_forward_last_seen, trusted_forward, trust_counts_forward, cur_loc)
                    if solved_loc is not None:
                        cur_loc = solved_loc
                        cur_set = shared
                    frame_sets[frame] = cur_set.copy()
                    zdepth_loc_raw[frame] = cur_loc.copy()

                seeded_backward = {}
                seeded_backward_depths = {}
                seeded_backward_frames = {}
                seeded_backward_last_seen = {}
                trusted_backward = set(fixed_world_points.keys())
                trust_counts_backward = {}
                for name, point in fixed_world_points.items():
                    seeded_backward[name] = point.copy()
                    seeded_backward_depths[name] = max(1e-4, -(init_t_inv @ point).z)
                    seeded_backward_frames[name] = None
                    seeded_backward_last_seen[name] = ref_f
                cur_loc = init_t_loc.copy()
                cur_set = frame_sets.get(ref_f, set()).copy()
                for frame in range(ref_f - 1, frame_start - 1, -1):
                    solved_loc, shared = solve_zdepth_location_from_seeded_points(frame, seeded_backward, seeded_backward_depths, seeded_backward_frames, seeded_backward_last_seen, trusted_backward, trust_counts_backward, cur_loc)
                    if solved_loc is not None:
                        cur_loc = solved_loc
                        cur_set = shared
                    frame_sets[frame] = cur_set.copy()
                    zdepth_loc_raw[frame] = cur_loc.copy()

                trusted_refined_tracks = set(fixed_world_points.keys()) | trusted_forward | trusted_backward
                refined_world_points = {name: point.copy() for name, point in fixed_world_points.items()}
                for track in tracks:
                    if track.name in fixed_world_points:
                        continue
                    if track.name not in trusted_refined_tracks:
                        continue
                    origins = []
                    directions = []
                    weights = []
                    fallback_point = None
                    world_data = track_world_data.get(track.name)
                    for frame in full_frames:
                        marker_co = frame_markers.get(frame, {}).get(track.name)
                        loc = zdepth_loc_raw.get(frame)
                        if marker_co is None or loc is None:
                            continue
                        if fallback_point is None and world_data is not None and frame in world_data:
                            fallback_point = world_data[frame].copy()
                        origins.append(loc)
                        directions.append(zdepth_base_rotation(frame).to_matrix() @ marker_to_camera_ray(marker_co, tan_ref_x, tan_ref_y, cam_ref.data))
                        weights.append(marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0)
                    if len(origins) >= 2:
                        if fallback_point is None:
                            fallback_point = fixed_world_points.get(track.name, origins[0] + directions[0] * ref_depth_dist)
                        refined_world_points[track.name] = solve_point_from_rays(origins, directions, fallback_point, weights)

                if len(refined_world_points) >= 2:
                    zdepth_refined_world_points = refined_world_points
                    refined_locs = {}
                    for frame in full_frames:
                        points_world = []
                        rays_world = []
                        weights = []
                        for track in tracks:
                            point_world = refined_world_points.get(track.name)
                            marker_co = frame_markers.get(frame, {}).get(track.name)
                            if point_world is None or marker_co is None:
                                continue
                            points_world.append(point_world)
                            rays_world.append(zdepth_base_rotation(frame).to_matrix() @ marker_to_camera_ray(marker_co, tan_ref_x, tan_ref_y, cam_ref.data))
                            weights.append(marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0)
                        if len(points_world) >= 2:
                            depth_anchor_loc = init_t_loc + depth_constraint_normal * depth_raw.get(frame, 0.0)
                            refined_locs[frame] = solve_camera_location_on_depth_plane(
                                points_world,
                                rays_world,
                                zdepth_loc_raw.get(frame, init_t_loc.copy()),
                                depth_constraint_normal,
                                depth_anchor_loc,
                                weights,
                            )
                        else:
                            refined_locs[frame] = zdepth_loc_raw.get(frame, init_t_loc.copy()).copy()
                    zdepth_loc_raw = refined_locs
            else:
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
                        if world_data is None or ref_f not in world_data or frame not in world_data:
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
        elif eff_scale_mode == 'NONE' and not props.tripod_mode:
            def translation_delta_from_markers(frame_a, frame_b):
                markers_a = frame_markers.get(frame_a, {})
                markers_b = frame_markers.get(frame_b, {})
                shared = list(set(markers_a.keys()) & set(markers_b.keys()))
                if not shared:
                    return None, set()
                c_a = weighted_marker_centroid(markers_a, shared, aspect, props.clip_center_weight)
                c_b = weighted_marker_centroid(markers_b, shared, aspect, props.clip_center_weight)
                if c_a is None or c_b is None:
                    return None, set()
                delta = Vector((
                    -(c_b.x - c_a.x) * (2.0 * depth * tan_ref_x),
                    -(c_b.y - c_a.y) * (2.0 * depth * tan_ref_y),
                ))
                return delta, set(shared)

            cur_dx = 0.0
            cur_dy = 0.0
            cur_set = frame_sets.get(ref_f, set()).copy()
            for frame in range(ref_f + 1, frame_end + 1):
                prev_frame = frame - 1
                delta, shared = translation_delta_from_markers(prev_frame, frame)
                if delta is not None:
                    cur_dx += delta.x
                    cur_dy += delta.y
                    cur_set = shared
                frame_sets[frame] = cur_set.copy()
                dx_raw[frame] = cur_dx
                dy_raw[frame] = cur_dy

            cur_dx = 0.0
            cur_dy = 0.0
            cur_set = frame_sets.get(ref_f, set()).copy()
            for frame in range(ref_f - 1, frame_start - 1, -1):
                next_frame = frame + 1
                delta, shared = translation_delta_from_markers(frame, next_frame)
                if delta is not None:
                    cur_dx -= delta.x
                    cur_dy -= delta.y
                    cur_set = shared
                frame_sets[frame] = cur_set.copy()
                dx_raw[frame] = cur_dx
                dy_raw[frame] = cur_dy
        else:
            for frame in full_frames:
                if frame == ref_f:
                    continue
                markers_curr = frame_markers.get(frame, {})
                shared = list(set(ref_markers.keys()) & set(markers_curr.keys()))
                min_shared = 1 if eff_scale_mode == 'NONE' and not props.tripod_mode else 2
                if len(shared) < min_shared:
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

        return _ClipRawPositionResult(
            frame_sets=frame_sets,
            dx_raw=dx_raw,
            dy_raw=dy_raw,
            pan_raw=pan_raw,
            depth_raw=depth_raw,
            zdepth_loc_raw=zdepth_loc_raw,
            zdepth_roll_curve=zdepth_roll_curve,
            zdepth_refined_world_points=zdepth_refined_world_points,
            track_world_data=track_world_data,
        )

    def _build_clip_track_location_curve(
        self,
        context,
        full_frames,
        eff_scale_mode,
        pos_smooth,
        raw_position,
        init_t_loc,
        init_t_quat,
        init_t_rot3,
        keep_existing_position,
        existing_loc_curve,
    ):
        props = context.scene.pcam_solve_props
        frame_sets = raw_position.frame_sets
        zdepth_loc_raw = raw_position.zdepth_loc_raw

        for frame in full_frames:
            frame_sets.setdefault(frame, frame_sets.get(frame - 1, frame_sets.get(frame + 1, set())))

        transition = {}
        for index, frame in enumerate(full_frames):
            curr_set = frame_sets.get(frame, set())
            prev_set = frame_sets.get(full_frames[index - 1], curr_set) if index > 0 else curr_set
            next_set = frame_sets.get(full_frames[index + 1], curr_set) if index < len(full_frames) - 1 else curr_set
            union_a = len(curr_set | prev_set)
            union_b = len(curr_set | next_set)
            coh_a = len(curr_set & prev_set) / max(1, union_a)
            coh_b = len(curr_set & next_set) / max(1, union_b)
            transition[frame] = max(0.0, 1.0 - min(coh_a, coh_b))
        expanded = expand_transition_blends(transition, full_frames, radius=2, decay=0.7)

        if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object and zdepth_loc_raw is None:
            pan_curve = stabilize_vector_curve(raw_position.pan_raw, full_frames, expanded, max_blend=0.08 + 0.28 * pos_smooth)
            pan_curve = bridge_vector_curve(pan_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.36 + 0.28 * pos_smooth)
            depth_curve = stabilize_scalar_curve(raw_position.depth_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            depth_curve = bridge_scalar_curve(depth_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.34 + 0.26 * pos_smooth)
            pan_curve = smooth_vector_curve_global(pan_curve, full_frames, strength=0.10 + 0.90 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
            depth_curve = smooth_scalar_curve_global(depth_curve, full_frames, strength=0.08 + 0.82 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
        elif eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
            depth_curve = stabilize_scalar_curve(raw_position.depth_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            depth_curve = bridge_scalar_curve(depth_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.34 + 0.26 * pos_smooth)
            depth_curve = smooth_scalar_curve_global(depth_curve, full_frames, strength=0.08 + 0.82 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
            zdepth_loc_curve = stabilize_vector_curve(zdepth_loc_raw, full_frames, expanded, max_blend=0.08 + 0.28 * pos_smooth)
            zdepth_loc_curve = bridge_vector_curve(zdepth_loc_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.36 + 0.28 * pos_smooth)
            zdepth_loc_curve = smooth_vector_curve_global(zdepth_loc_curve, full_frames, strength=0.10 + 0.90 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
        else:
            dx_curve = stabilize_scalar_curve(raw_position.dx_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
            dy_curve = stabilize_scalar_curve(raw_position.dy_raw, full_frames, expanded, max_blend=0.06 + 0.24 * pos_smooth)
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
                if zdepth_loc_raw is not None:
                    return zdepth_loc_curve.get(frame, init_t_loc.copy()).copy()
                loc = init_t_loc - (init_t_rot3 @ pan_curve.get(frame, Vector((0.0, 0.0, 0.0))))
                loc += (init_t_quat @ Vector((0.0, 0.0, -1.0))) * depth_curve.get(frame, 0.0)
                return loc
            return init_t_loc + (init_t_rot3 @ Vector((dx_curve.get(frame, 0.0), dy_curve.get(frame, 0.0), 0.0)))

        if keep_existing_position and existing_loc_curve is not None:
            loc_curve = {frame: existing_loc_curve.get(frame, init_t_loc.copy()).copy() for frame in full_frames}
        else:
            loc_curve = {frame: get_loc_for_frame(frame) for frame in full_frames}
            if not props.tripod_mode and pos_smooth > 1e-4 and zdepth_loc_raw is None:
                loc_curve = stabilize_vector_curve(loc_curve, full_frames, expanded, max_blend=0.05 + 0.18 * pos_smooth)
                loc_curve = bridge_vector_curve(loc_curve, full_frames, expanded, threshold=0.24, max_bridge_blend=0.22 + 0.30 * pos_smooth)
                loc_curve = smooth_vector_curve_global(loc_curve, full_frames, strength=0.10 + 0.90 * pos_smooth, passes=1 + int(round(3 * pos_smooth)))
        if props.lock_camera_z:
            loc_curve = {frame: Vector((loc.x, loc.y, init_t_loc.z)) for frame, loc in loc_curve.items()}
        return loc_curve, expanded

    def _refine_clip_track_rotation_curve(
        self,
        context,
        cam_ref,
        frame_start,
        frame_end,
        full_frames,
        ref_f,
        frame_markers,
        frame_sets,
        fixed_world_points,
        zdepth_roll_curve,
        zdepth_refined_world_points,
        track_world_data,
        loc_curve,
        lens_curve,
        expanded,
        init_t_loc,
        init_t_quat,
        tan_ref_x,
        tan_ref_y,
        aspect,
        depth,
        ref_lens,
        eff_scale_mode,
        keep_existing_position,
        pt_smooth,
        roll_smooth,
        rotation_mode,
    ):
        props = context.scene.pcam_solve_props
        dynamic_fixed_world_points = {name: point.copy() for name, point in fixed_world_points.items()}
        if (
            eff_scale_mode == 'Z_DEPTH' and
            props.clip_depth_object and
            zdepth_refined_world_points is not None and
            not keep_existing_position
        ):
            for name, point in zdepth_refined_world_points.items():
                dynamic_fixed_world_points.setdefault(name, point.copy())

        def seed_dynamic_fixed_point(name, frame, marker_co, loc, quat, lens_value):
            if props.clip_depth_object and (eff_scale_mode != 'Z_DEPTH' or keep_existing_position):
                cam_matrix = Matrix.Translation(loc) @ quat.to_matrix().to_4x4()
                hit = raycast_marker_world_from_matrix(
                    context,
                    cam_ref.data,
                    cam_matrix,
                    props.clip_depth_object,
                    marker_co,
                    lens_value,
                )
                if hit is not None:
                    return hit

            if eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object:
                world_data = track_world_data.get(name)
                if world_data and frame in world_data:
                    return world_data[frame].copy()

            tan_x, tan_y = get_camera_tan(cam_ref.data, lens_value, context.scene)
            planar_depth = max(depth, 1e-4)
            point_local = Vector((
                (2.0 * marker_co.x - 1.0) * planar_depth * tan_x,
                (2.0 * marker_co.y - 1.0) * planar_depth * tan_y,
                -planar_depth,
            ))
            return loc + (quat @ point_local)

        def build_rotation_inputs(frame, seed_quat=None):
            marker_map = frame_markers.get(frame, {})
            if eff_scale_mode == 'Z_DEPTH':
                tan_x, tan_y = tan_ref_x, tan_ref_y
            else:
                tan_x, tan_y = get_camera_tan(cam_ref.data, lens_curve[frame], context.scene)
            stable_names = select_stable_track_names(frame, frame_sets, dynamic_fixed_world_points.keys())
            if len(stable_names) < 3:
                stable_names = set(frame_sets.get(frame, set()))
            if eff_scale_mode == 'NONE' and len(stable_names & marker_map.keys()) < 2:
                stable_names = set(marker_map.keys())
            points_world = []
            rays_local = []
            weights = []
            used_names = set()
            for name in stable_names:
                marker_co = marker_map.get(name)
                if marker_co is None:
                    continue
                point_world = dynamic_fixed_world_points.get(name)
                if point_world is None and seed_quat is not None:
                    point_world = seed_dynamic_fixed_point(
                        name,
                        frame,
                        marker_co,
                        loc_curve.get(frame, init_t_loc),
                        seed_quat,
                        lens_curve.get(frame, ref_lens),
                    )
                    dynamic_fixed_world_points[name] = point_world.copy()
                if point_world is None:
                    continue
                stability_w = track_stability_weight(frame, frame_sets, name)
                base_w = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0
                points_world.append(point_world)
                rays_local.append(marker_to_camera_ray(marker_co, tan_x, tan_y, cam_ref.data))
                weights.append(base_w * stability_w)
                used_names.add(name)
            return points_world, rays_local, weights, used_names, sum(weights) / max(1, len(weights))

        def solve_single_ray_to_world(desired_world, ray_local, fallback_quat, roll_reference=None):
            if desired_world.length_squared <= 1e-9 or ray_local.length_squared <= 1e-9:
                return fallback_quat.copy()
            roll_reference = fallback_quat if roll_reference is None else roll_reference
            desired_in_reference = roll_reference.inverted() @ desired_world.normalized()
            reference_euler = (
                roll_reference.to_euler(rotation_mode)
                if rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'} else Euler()
            )
            return solve_single_ray_euler_y_locked(
                roll_reference,
                desired_in_reference,
                ray_local,
                reference_euler,
                rotation_mode,
            )

        def solve_single_point_rotation(point_world, ray_local, frame, fallback_quat, roll_reference=None):
            desired_world = point_world - loc_curve.get(frame, init_t_loc)
            return solve_single_ray_to_world(desired_world, ray_local, fallback_quat, roll_reference)

        def solve_none_tripod_adjacent(frame_ref, frame_curr, base_quat, roll_reference=None):
            markers_ref = frame_markers.get(frame_ref, {})
            markers_curr = frame_markers.get(frame_curr, {})
            shared_names = sorted(set(markers_ref.keys()) & set(markers_curr.keys()))
            if not shared_names:
                return base_quat.copy(), set(), None
            tan_ref_frame = get_camera_tan(cam_ref.data, lens_curve.get(frame_ref, ref_lens), context.scene)
            tan_curr_frame = get_camera_tan(cam_ref.data, lens_curve.get(frame_curr, ref_lens), context.scene)
            reference_rays = [
                marker_to_camera_ray(markers_ref[name], tan_ref_frame[0], tan_ref_frame[1], cam_ref.data)
                for name in shared_names
            ]
            current_rays = [
                marker_to_camera_ray(markers_curr[name], tan_curr_frame[0], tan_curr_frame[1], cam_ref.data)
                for name in shared_names
            ]
            weights = [
                marker_center_weight(markers_curr[name], aspect) if props.clip_center_weight else 1.0
                for name in shared_names
            ]
            if props.clip_lock_roll:
                weight_sum = max(sum(weights), 1e-9)
                center_ref = sum(
                    (ray * weight for ray, weight in zip(reference_rays, weights)),
                    Vector((0.0, 0.0, 0.0)),
                ) / weight_sum
                center_curr = sum(
                    (ray * weight for ray, weight in zip(current_rays, weights)),
                    Vector((0.0, 0.0, 0.0)),
                ) / weight_sum
                desired_world = base_quat @ center_ref
                solved_quat = solve_single_ray_to_world(
                    desired_world,
                    center_curr,
                    base_quat,
                    init_t_quat,
                )
                single_name = shared_names[0] if len(shared_names) == 1 else None
                return solved_quat, set(shared_names), single_name
            if len(shared_names) == 1:
                desired_world = base_quat @ reference_rays[0]
                solved_quat = solve_single_ray_to_world(
                    desired_world,
                    current_rays[0],
                    base_quat,
                    roll_reference,
                )
                return solved_quat, set(shared_names), shared_names[0]
            delta_quat = solve_weighted_kabsch_rotation(
                reference_rays,
                current_rays,
                False,
                weights,
            )
            solved_quat = base_quat @ delta_quat
            if solved_quat.dot(base_quat) < 0.0:
                solved_quat.negate()
            return solved_quat, set(shared_names), None

        def solve_refit_rotation_curve():
            solved = {ref_f: init_t_quat.copy()}
            rotation_frame_sets = {
                ref_f: set(frame_markers.get(ref_f, {}).keys()) & set(dynamic_fixed_world_points.keys())
            }
            single_point_roll_reference_frames = {}
            tripod_single_point_adjacent = {}
            single_roll_reference_frame = None
            for frame in range(ref_f + 1, frame_end + 1):
                prev_quat = solved.get(frame - 1, init_t_quat)
                if eff_scale_mode == 'NONE' and props.tripod_mode:
                    markers_shared = set(frame_markers.get(frame - 1, {}).keys()) & set(frame_markers.get(frame, {}).keys())
                    if props.clip_lock_roll:
                        single_roll_reference_frame = ref_f
                        roll_reference = init_t_quat
                    elif len(markers_shared) == 1:
                        if single_roll_reference_frame is None:
                            single_roll_reference_frame = frame - 1
                        roll_reference = solved.get(single_roll_reference_frame, prev_quat)
                    else:
                        roll_reference = None
                    solved_quat, used_names, single_name = solve_none_tripod_adjacent(
                        frame - 1, frame, prev_quat, roll_reference
                    )
                    solved[frame] = solved_quat
                    rotation_frame_sets[frame] = used_names
                    if single_name is not None:
                        single_point_roll_reference_frames[frame] = single_roll_reference_frame
                        tripod_single_point_adjacent[frame] = (frame - 1, single_name)
                    elif len(used_names) >= 2:
                        single_roll_reference_frame = None
                    continue
                points_world, rays_local, weights, used_names, avg_weight = build_rotation_inputs(frame, prev_quat)
                rotation_frame_sets[frame] = used_names
                if len(points_world) < 2:
                    if eff_scale_mode == 'NONE' and len(points_world) == 1:
                        if single_roll_reference_frame is None:
                            single_roll_reference_frame = frame - 1
                        single_point_roll_reference_frames[frame] = single_roll_reference_frame
                        roll_reference = solved.get(single_roll_reference_frame, prev_quat)
                        solved[frame] = solve_single_point_rotation(
                            points_world[0], rays_local[0], frame, prev_quat, roll_reference
                        )
                    else:
                        solved[frame] = prev_quat.copy()
                    continue
                single_roll_reference_frame = None
                raw_quat = solve_rotation_quat_at_location(points_world, rays_local, loc_curve.get(frame, init_t_loc), prev_quat, False, weights)
                raw_quat = stabilize_camera_roll_step(raw_quat, prev_quat)
                stable_count = len(used_names)
                stability = min(1.0, max(0.28, (stable_count / 5.0) * avg_weight))
                blend = (0.22 + 0.58 * (1.0 - expanded.get(frame, 0.0))) * stability
                solved[frame] = prev_quat.slerp(raw_quat, min(1.0, max(0.0, blend)))
            single_roll_reference_frame = None
            for frame in range(ref_f - 1, frame_start - 1, -1):
                next_quat = solved.get(frame + 1, init_t_quat)
                if eff_scale_mode == 'NONE' and props.tripod_mode:
                    markers_shared = set(frame_markers.get(frame + 1, {}).keys()) & set(frame_markers.get(frame, {}).keys())
                    if props.clip_lock_roll:
                        single_roll_reference_frame = ref_f
                        roll_reference = init_t_quat
                    elif len(markers_shared) == 1:
                        if single_roll_reference_frame is None:
                            single_roll_reference_frame = frame + 1
                        roll_reference = solved.get(single_roll_reference_frame, next_quat)
                    else:
                        roll_reference = None
                    solved_quat, used_names, single_name = solve_none_tripod_adjacent(
                        frame + 1, frame, next_quat, roll_reference
                    )
                    solved[frame] = solved_quat
                    rotation_frame_sets[frame] = used_names
                    if single_name is not None:
                        single_point_roll_reference_frames[frame] = single_roll_reference_frame
                        tripod_single_point_adjacent[frame] = (frame + 1, single_name)
                    elif len(used_names) >= 2:
                        single_roll_reference_frame = None
                    continue
                points_world, rays_local, weights, used_names, avg_weight = build_rotation_inputs(frame, next_quat)
                rotation_frame_sets[frame] = used_names
                if len(points_world) < 2:
                    if eff_scale_mode == 'NONE' and len(points_world) == 1:
                        if single_roll_reference_frame is None:
                            single_roll_reference_frame = frame + 1
                        single_point_roll_reference_frames[frame] = single_roll_reference_frame
                        roll_reference = solved.get(single_roll_reference_frame, next_quat)
                        solved[frame] = solve_single_point_rotation(
                            points_world[0], rays_local[0], frame, next_quat, roll_reference
                        )
                    else:
                        solved[frame] = next_quat.copy()
                    continue
                single_roll_reference_frame = None
                raw_quat = solve_rotation_quat_at_location(points_world, rays_local, loc_curve.get(frame, init_t_loc), next_quat, False, weights)
                raw_quat = stabilize_camera_roll_step(raw_quat, next_quat)
                stable_count = len(used_names)
                stability = min(1.0, max(0.28, (stable_count / 5.0) * avg_weight))
                blend = (0.22 + 0.58 * (1.0 - expanded.get(frame, 0.0))) * stability
                solved[frame] = next_quat.slerp(raw_quat, min(1.0, max(0.0, blend)))

            rotation_transition = {}
            for index, frame in enumerate(full_frames):
                curr_set = rotation_frame_sets.get(frame, set())
                prev_set = rotation_frame_sets.get(full_frames[index - 1], curr_set) if index > 0 else curr_set
                next_set = rotation_frame_sets.get(full_frames[index + 1], curr_set) if index < len(full_frames) - 1 else curr_set
                union_prev = len(curr_set | prev_set)
                union_next = len(curr_set | next_set)
                coherence_prev = len(curr_set & prev_set) / max(1, union_prev)
                coherence_next = len(curr_set & next_set) / max(1, union_next)
                rotation_transition[frame] = max(0.0, 1.0 - min(coherence_prev, coherence_next))
            rotation_expanded = expand_transition_blends(rotation_transition, full_frames, radius=3, decay=0.72)

            solved = smooth_quaternion_curve(solved, full_frames, rotation_expanded, max_blend=0.42)
            solved = bridge_quaternion_curve(solved, full_frames, rotation_expanded, threshold=0.20, max_bridge_blend=0.88)

            if eff_scale_mode == 'NONE' and props.tripod_mode and rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'}:
                continuous_eulers = {ref_f: solved.get(ref_f, init_t_quat).to_euler(rotation_mode)}
                for frame in range(ref_f + 1, frame_end + 1):
                    continuous_eulers[frame] = solved.get(frame, init_t_quat).to_euler(
                        rotation_mode, continuous_eulers[frame - 1]
                    )
                for frame in range(ref_f - 1, frame_start - 1, -1):
                    continuous_eulers[frame] = solved.get(frame, init_t_quat).to_euler(
                        rotation_mode, continuous_eulers[frame + 1]
                    )
                roll_curve = {frame: euler.y for frame, euler in continuous_eulers.items()}
                roll_curve = stabilize_scalar_curve(roll_curve, full_frames, rotation_expanded, max_blend=0.58)
                roll_curve = bridge_scalar_curve(
                    roll_curve, full_frames, rotation_expanded, threshold=0.20, max_bridge_blend=0.92
                )
                solved = {
                    frame: Euler(
                        (continuous_eulers[frame].x, roll_curve[frame], continuous_eulers[frame].z),
                        rotation_mode,
                    ).to_quaternion()
                    for frame in full_frames
                }
            return solved, single_point_roll_reference_frames, tripod_single_point_adjacent

        def apply_rotation_smooth(rot_curve):
            if pt_smooth <= 1e-4 and roll_smooth <= 1e-4:
                return rot_curve
            view_axis = Vector((0.0, 0.0, -1.0))
            pan_tilt_quats = {}
            roll_raw = {}
            for frame in full_frames:
                base_quat = rot_curve.get(frame, init_t_quat.copy())
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
            return {
                frame: replace_quaternion_twist(
                    pan_tilt_quats.get(frame, rot_curve.get(frame, init_t_quat.copy())),
                    view_axis,
                    roll_raw.get(frame, 0.0),
                )
                for frame in full_frames
            }

        use_zdepth_stable_roll = eff_scale_mode == 'Z_DEPTH' and props.clip_depth_object and not props.tripod_mode and zdepth_roll_curve is not None

        rot_quats, single_point_roll_reference_frames, tripod_single_point_adjacent = solve_refit_rotation_curve()
        if use_zdepth_stable_roll and not keep_existing_position:
            roll_curve = zdepth_roll_curve.copy()
            roll_curve = bridge_scalar_curve(roll_curve, full_frames, expanded, threshold=0.28, max_bridge_blend=0.35)
            view_axis = Vector((0.0, 0.0, -1.0))
            rot_quats = {
                frame: replace_quaternion_twist(
                    rot_quats.get(frame, init_t_quat.copy()),
                    view_axis,
                    -roll_curve.get(frame, 0.0),
                )
                for frame in full_frames
            }
        rot_quats = apply_rotation_smooth(rot_quats)
        for frame, roll_reference_frame in single_point_roll_reference_frames.items():
            roll_reference = rot_quats.get(roll_reference_frame, init_t_quat)
            adjacent_data = tripod_single_point_adjacent.get(frame)
            if adjacent_data is not None:
                adjacent_frame, track_name = adjacent_data
                marker_ref = frame_markers.get(adjacent_frame, {}).get(track_name)
                marker_curr = frame_markers.get(frame, {}).get(track_name)
                if marker_ref is None or marker_curr is None:
                    continue
                tan_adjacent = get_camera_tan(
                    cam_ref.data, lens_curve.get(adjacent_frame, ref_lens), context.scene
                )
                tan_current = get_camera_tan(cam_ref.data, lens_curve.get(frame, ref_lens), context.scene)
                ray_ref = marker_to_camera_ray(marker_ref, tan_adjacent[0], tan_adjacent[1], cam_ref.data)
                ray_curr = marker_to_camera_ray(marker_curr, tan_current[0], tan_current[1], cam_ref.data)
                desired_world = rot_quats.get(adjacent_frame, roll_reference) @ ray_ref
                rot_quats[frame] = solve_single_ray_to_world(
                    desired_world,
                    ray_curr,
                    rot_quats.get(frame, roll_reference),
                    roll_reference,
                )
                continue
            points_world, rays_local, _weights, _used_names, _avg_weight = build_rotation_inputs(
                frame, rot_quats.get(frame, roll_reference)
            )
            if len(points_world) == 1:
                rot_quats[frame] = solve_single_point_rotation(
                    points_world[0],
                    rays_local[0],
                    frame,
                    rot_quats.get(frame, roll_reference),
                    roll_reference,
                )

        if eff_scale_mode == 'NONE' and props.clip_lock_roll:
            rot_quats = {
                frame: preserve_camera_roll_from_reference(
                    rot_quats.get(frame, init_t_quat),
                    init_t_quat,
                )
                for frame in full_frames
            }

        return rot_quats

    def _bake_clip_track_camera_curves(
        self,
        context,
        target,
        lens_owner,
        full_frames,
        frame_start,
        frame_end,
        eff_scale_mode,
        keep_existing_position,
        keep_existing_focal,
        suppress_focal_bake,
        pin_existing_focal_range,
        loc_curve,
        rot_quats,
        lens_curve,
        init_t_loc,
        init_t_quat,
        location_curve_snapshot,
        lens_action_copy,
        ref_lens,
        lens_curve_snapshot,
    ):
        self._bake_clip_track_camera_curves(
            context,
            target,
            lens_owner,
            full_frames,
            frame_start,
            frame_end,
            eff_scale_mode,
            keep_existing_position,
            keep_existing_focal,
            suppress_focal_bake,
            pin_existing_focal_range,
            loc_curve,
            rot_quats,
            lens_curve,
            init_t_loc,
            init_t_quat,
            location_curve_snapshot,
            lens_action_copy,
            ref_lens,
            lens_curve_snapshot,
        )

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
        keep_existing_position = pcam_use_existing_position(props)
        lens_owner = cam_ref
        target_curve_snapshot = self.snapshot_animation_action(target)
        lens_curve_snapshot = self.snapshot_animation_action(lens_owner.data) if getattr(lens_owner, "data", None) is not None else []
        has_existing_focal_keys = self.has_camera_focal_length_keys(lens_owner)
        has_focal_variation_in_range = self.camera_lens_varies_over_range(context, lens_owner, frame_start, frame_end) if frame_range is not None else False
        keep_existing_focal = props.clip_use_existing_focal and eff_scale_mode == 'FOCAL_LENGTH' and has_existing_focal_keys
        suppress_focal_bake = props.clip_use_existing_focal and eff_scale_mode == 'FOCAL_LENGTH' and not has_existing_focal_keys
        pin_existing_focal_range = frame_range is not None and eff_scale_mode != 'FOCAL_LENGTH' and not keep_existing_focal and (has_existing_focal_keys or has_focal_variation_in_range)

        restore_frame = context.scene.frame_current
        context.scene.frame_set(ref_f)
        ref_t_mat_before_clear = target.matrix_world.copy()
        if not is_obj:
            ref_t_mat_before_clear = matrix_without_scale(ref_t_mat_before_clear)
        existing_loc_curve = None
        existing_lens_curve = None
        location_curve_snapshot = self.snapshot_animation_curves(target, {"location"}) if keep_existing_position else []
        lens_action_copy = self.copy_animation_action(lens_owner.data) if keep_existing_focal and getattr(lens_owner, "data", None) else None

        def rollback_animation():
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(lens_owner, "data", None) is not None:
                self.restore_animation_snapshot_exact(lens_owner.data, lens_curve_snapshot)
            if lens_action_copy is not None and lens_action_copy.users == 0:
                bpy.data.actions.remove(lens_action_copy)

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

        context.scene.frame_set(ref_f)
        init_t_mat = ref_t_mat_before_clear.copy()
        init_t_loc = init_t_mat.translation.copy()
        init_t_quat = self.get_target_rotation_quaternion(target)
        init_t_rot3 = init_t_quat.to_matrix()
        init_t_inv = init_t_mat.inverted()
        tan_ref_x, tan_ref_y = get_camera_tan(cam_ref.data, ref_lens, context.scene)
        aspect = tan_ref_x / max(tan_ref_y, 1e-6)

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
        init_view_dir = init_t_quat @ Vector((0.0, 0.0, -1.0))
        if init_view_dir.length_squared > 1e-12:
            init_view_dir.normalize()
        depth_constraint_normal = init_view_dir.copy()
        if props.clip_depth_object:
            center_hit = raycast_marker_world_with_normal(context, cam_ref, props.clip_depth_object, Vector((0.5, 0.5)))
            if center_hit is not None and center_hit[1].length_squared > 1e-12:
                depth_constraint_normal = center_hit[1].copy()
                if depth_constraint_normal.dot(init_view_dir) < 0.0:
                    depth_constraint_normal.negate()
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
            rollback_animation()
            context.scene.frame_set(restore_frame)
            self.report({'ERROR'}, "Z-Depth needs valid reference points or depth object.")
            return {'CANCELLED'}

        raw_position = self._solve_clip_track_raw_position(
            context,
            cam_ref,
            clip,
            tracks,
            ref_f,
            frame_start,
            frame_end,
            full_frames,
            depth,
            norm_curve,
            frame_markers,
            eff_scale_mode,
            ref_markers,
            init_t_inv,
            init_t_loc,
            init_t_quat,
            depth_constraint_normal,
            tan_ref_x,
            tan_ref_y,
            aspect,
            fixed_world_points,
            restore_frame,
            rollback_animation,
        )
        if raw_position is None:
            return {'CANCELLED'}
        frame_sets = raw_position.frame_sets
        zdepth_roll_curve = raw_position.zdepth_roll_curve
        zdepth_refined_world_points = raw_position.zdepth_refined_world_points
        track_world_data = raw_position.track_world_data

        loc_curve, expanded = self._build_clip_track_location_curve(
            context,
            full_frames,
            eff_scale_mode,
            pos_smooth,
            raw_position,
            init_t_loc,
            init_t_quat,
            init_t_rot3,
            keep_existing_position,
            existing_loc_curve,
        )

        rot_quats = self._refine_clip_track_rotation_curve(
            context,
            cam_ref,
            frame_start,
            frame_end,
            full_frames,
            ref_f,
            frame_markers,
            frame_sets,
            fixed_world_points,
            zdepth_roll_curve,
            zdepth_refined_world_points,
            track_world_data,
            loc_curve,
            lens_curve,
            expanded,
            init_t_loc,
            init_t_quat,
            tan_ref_x,
            tan_ref_y,
            aspect,
            depth,
            ref_lens,
            eff_scale_mode,
            keep_existing_position,
            pt_smooth,
            roll_smooth,
            target.rotation_mode,
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
        total_frames = frame_end - frame_start + 1
        self.report({'INFO'}, pcam_bake_result_message("Clip Track motion", target.name, total_frames, total_frames))
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

        def adjacent_object_roll_delta(frame_a, frame_b):
            markers_a = frame_markers.get(frame_a, {})
            markers_b = frame_markers.get(frame_b, {})
            shared = list(set(markers_a.keys()) & set(markers_b.keys()))
            if len(shared) < 2:
                return None
            points_a = []
            points_b = []
            weights = []
            for name in shared:
                p_a = markers_a.get(name)
                p_b = markers_b.get(name)
                if p_a is None or p_b is None:
                    continue
                points_a.append(p_a)
                points_b.append(p_b)
                weights.append(marker_center_weight(p_b, aspect) if props.clip_center_weight else 1.0)
            if len(points_a) < 2:
                return None
            return solve_planar_roll_from_points(points_a, points_b, weights, aspect)

        object_roll_curve = {ref_f: 0.0}
        current_roll = 0.0
        for frame in range(ref_f + 1, frame_end + 1):
            delta_roll = adjacent_object_roll_delta(frame - 1, frame)
            if delta_roll is not None:
                current_roll += delta_roll
            object_roll_curve[frame] = current_roll
        current_roll = 0.0
        for frame in range(ref_f - 1, frame_start - 1, -1):
            delta_roll = adjacent_object_roll_delta(frame, frame + 1)
            if delta_roll is not None:
                current_roll -= delta_roll
            object_roll_curve[frame] = current_roll
        object_roll_curve = stabilize_roll_curve(
            object_roll_curve,
            list(range(frame_start, frame_end + 1)),
            despike_threshold_deg=1.2,
            smooth_blend=0.18,
        )

        ref_local_points = {}
        ref_object_local_points = {}
        ref_weights = {}
        for track in tracks:
            marker_co = ref_markers.get(track.name)
            if marker_co is None:
                continue
            hit = raycast_marker_world(context, cam_ref, depth_obj, marker_co)
            if hit is None:
                continue
            ref_local_points[track.name] = ref_depth_inv @ hit
            ref_object_local_points[track.name] = init_t_mat.inverted() @ hit
            ref_weights[track.name] = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0

        if len(ref_local_points) < 2:
            self.report({'ERROR'}, "Clip Track Object needs at least two valid reference hits.")
            return {'CANCELLED'}

        dynamic_local_points = {name: point.copy() for name, point in ref_local_points.items()}
        dynamic_object_local_points = {name: point.copy() for name, point in ref_object_local_points.items()}
        dynamic_weights = ref_weights.copy()

        self.clear_animation_safely(target, (frame_start, frame_end) if props.use_custom_range else None)

        def solve_object_frame(frame, dyn_local_points, dyn_object_local_points, dyn_weights, prev_obj_quat, prev_obj_euler):
            context.scene.frame_set(frame)
            context.view_layer.update()
            marker_map = frame_markers.get(frame, {})
            curr_depth_mat = evaluated_matrix_world(context, depth_obj)
            curr_depth_inv = curr_depth_mat.inverted()

            curr_world_points = []
            ref_object_locals = []
            curr_locals = []
            ref_locals = []
            weights = []
            pending_hits = []

            for track in tracks:
                name = track.name
                marker_co = marker_map.get(name)
                if marker_co is None:
                    continue
                hit = raycast_marker_world(context, cam_ref, depth_obj, marker_co)
                if hit is None:
                    continue
                if name not in dyn_local_points:
                    pending_hits.append((name, marker_co.copy(), hit.copy()))
                    continue
                weight = marker_center_weight(marker_co, aspect) if props.clip_center_weight else dyn_weights.get(name, 1.0)
                curr_world_points.append(hit)
                ref_object_locals.append(dyn_object_local_points[name])
                curr_locals.append(curr_depth_inv @ hit)
                ref_locals.append(dyn_local_points[name])
                weights.append(weight)

            if len(curr_locals) < 2:
                return None, prev_obj_quat, prev_obj_euler

            curr_centroid_world = weighted_points_centroid(curr_world_points, weights)
            object_anchor_local = weighted_points_centroid(ref_object_locals, weights)

            local_angle = object_roll_curve.get(frame, 0.0)
            local_delta_quat = Quaternion(Vector((0.0, 0.0, 1.0)), local_angle)
            solved_quat = curr_depth_mat.to_quaternion() @ local_delta_quat @ ref_depth_quat_inv @ init_t_rot
            target.location = object_location_from_local_anchor(curr_centroid_world, object_anchor_local, solved_quat, init_t_scale)
            solved_quat = self.set_target_rotation_continuous(target, solved_quat, prev_obj_quat, prev_obj_euler)
            prev_obj_quat = solved_quat.copy()
            if prev_obj_euler is not None:
                prev_obj_euler = target.rotation_euler.copy()

            target_mat = target.matrix_world.copy()
            for name, marker_co, hit in pending_hits:
                if name in dyn_local_points:
                    continue
                dyn_local_points[name] = curr_depth_inv @ hit
                dyn_object_local_points[name] = target_mat.inverted() @ hit
                dyn_weights[name] = marker_center_weight(marker_co, aspect) if props.clip_center_weight else 1.0

            return (target.location.copy(), solved_quat.copy(), init_t_scale.copy()), prev_obj_quat, prev_obj_euler

        solved_frames = {}

        forward_local = {name: point.copy() for name, point in dynamic_local_points.items()}
        forward_object_local = {name: point.copy() for name, point in dynamic_object_local_points.items()}
        forward_weights = dynamic_weights.copy()
        prev_obj_quat = init_t_rot.copy()
        prev_obj_euler = init_t_rot.to_euler(target.rotation_mode) if target.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'} else None
        for frame in range(ref_f, frame_end + 1):
            result, prev_obj_quat, prev_obj_euler = solve_object_frame(
                frame,
                forward_local,
                forward_object_local,
                forward_weights,
                prev_obj_quat,
                prev_obj_euler,
            )
            if result is not None:
                solved_frames[frame] = result

        backward_local = {name: point.copy() for name, point in dynamic_local_points.items()}
        backward_object_local = {name: point.copy() for name, point in dynamic_object_local_points.items()}
        backward_weights = dynamic_weights.copy()
        prev_obj_quat = init_t_rot.copy()
        prev_obj_euler = init_t_rot.to_euler(target.rotation_mode) if target.rotation_mode not in {'QUATERNION', 'AXIS_ANGLE'} else None
        for frame in range(ref_f - 1, frame_start - 1, -1):
            result, prev_obj_quat, prev_obj_euler = solve_object_frame(
                frame,
                backward_local,
                backward_object_local,
                backward_weights,
                prev_obj_quat,
                prev_obj_euler,
            )
            if result is not None:
                solved_frames[frame] = result

        baked_frames = 0
        for frame in range(frame_start, frame_end + 1):
            result = solved_frames.get(frame)
            if result is None:
                continue
            loc, quat, scale = result
            context.scene.frame_set(frame)
            target.location = loc
            self.set_target_rotation(target, quat)
            target.scale = scale
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
        self.report({'INFO'}, pcam_bake_result_message("Clip Track object motion", target.name, baked_frames, total_frames))
        return {'FINISHED'}


    # Clip Track dispatcher and object fallback. Camera solves currently return
    # through execute_clip_track_refined() after marker scale analysis.
    def _collect_clip_track_markers(self, clip, tracks, frame_start, frame_end):
        frame_markers = {}
        for frame in range(frame_start, frame_end + 1):
            clip_frame = pcam_scene_to_clip_frame(clip, frame)
            markers = {}
            for track in tracks:
                marker = track.markers.find_frame(clip_frame)
                if marker and not getattr(marker, 'mute', False):
                    markers[track.name] = get_track_display_co(track, marker)
            frame_markers[frame] = markers
        return frame_markers

    def _clip_track_pair_scale_ratio(self, markers_a, markers_b, aspect):
        shared = list(set(markers_a.keys()) & set(markers_b.keys()))
        if len(shared) < 2:
            return None
        ratios = []
        for first_name, second_name in itertools.combinations(shared, 2):
            first_a = markers_a[first_name]
            second_a = markers_a[second_name]
            first_b = markers_b[first_name]
            second_b = markers_b[second_name]
            dist_a = math.sqrt(((first_a.x - second_a.x) * aspect) ** 2 + (first_a.y - second_a.y) ** 2)
            dist_b = math.sqrt(((first_b.x - second_b.x) * aspect) ** 2 + (first_b.y - second_b.y) ** 2)
            if dist_a > 1e-4:
                ratios.append(dist_b / dist_a)
        if not ratios:
            return None
        ratios.sort()
        return ratios[len(ratios) // 2]

    def _build_clip_track_scale_curve(self, frame_markers, frame_start, frame_end, ref_f, aspect):
        step_scale_ratios = {}
        for frame in range(frame_start + 1, frame_end + 1):
            step_scale_ratios[frame] = self._clip_track_pair_scale_ratio(
                frame_markers[frame - 1],
                frame_markers[frame],
                aspect,
            ) or 1.0

        base_curve = {ref_f: 1.0}
        for frame in range(ref_f + 1, frame_end + 1):
            base_curve[frame] = base_curve[frame - 1] * step_scale_ratios.get(frame, 1.0)
        for frame in range(ref_f - 1, frame_start - 1, -1):
            step_ratio = step_scale_ratios.get(frame + 1, 1.0)
            base_curve[frame] = base_curve[frame + 1] / step_ratio if step_ratio > 1e-6 else base_curve[frame + 1]

        ref_markers = frame_markers.get(ref_f, {})
        absolute_scale = {ref_f: 1.0}
        for frame in range(frame_start, frame_end + 1):
            if frame == ref_f:
                continue
            ratio = self._clip_track_pair_scale_ratio(ref_markers, frame_markers[frame], aspect)
            if ratio is not None:
                absolute_scale[frame] = ratio

        correction_keys = sorted(absolute_scale.keys())
        correction_curve = {}
        for frame in correction_keys:
            base_value = base_curve.get(frame, 1.0)
            correction_curve[frame] = absolute_scale[frame] / base_value if base_value > 1e-6 else 1.0

        norm_curve = {}
        for frame in range(frame_start, frame_end + 1):
            base_value = base_curve.get(frame, 1.0)
            if frame in correction_curve:
                correction = correction_curve[frame]
            else:
                previous_keys = [key for key in correction_keys if key < frame]
                next_keys = [key for key in correction_keys if key > frame]
                if previous_keys and next_keys:
                    first_key = previous_keys[-1]
                    second_key = next_keys[0]
                    blend = (frame - first_key) / max(1, second_key - first_key)
                    first_correction = correction_curve[first_key]
                    second_correction = correction_curve[second_key]
                    correction = (
                        first_correction * ((second_correction / first_correction) ** blend)
                        if first_correction > 1e-6 and second_correction > 1e-6 else
                        (1.0 - blend) * first_correction + blend * second_correction
                    )
                elif previous_keys:
                    correction = correction_curve[previous_keys[-1]]
                elif next_keys:
                    correction = correction_curve[next_keys[0]]
                else:
                    correction = 1.0
            norm_curve[frame] = base_value * correction
        return norm_curve

    def execute_clip_track(self, context, target):
        props = context.scene.pcam_solve_props
        clip = props.target_clip
        is_obj = props.apply_to == 'OBJECT'
        cam_ref = context.scene.camera

        try:
            track_object_index = int(props.tracking_object_idx)
            tracks = clip.tracking.objects[track_object_index].tracks
        except Exception:
            return {'CANCELLED'}
        if not tracks:
            return {'CANCELLED'}

        frame_start, frame_end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, frame_start, frame_end)

        context.scene.frame_set(ref_f)
        ref_lens = cam_ref.data.lens
        context.scene.frame_set(frame_start)

        init_target_matrix = target.matrix_world.copy()
        init_camera_matrix = cam_ref.matrix_world.copy()
        if not is_obj:
            init_target_matrix = matrix_without_scale(init_target_matrix)
            init_camera_matrix = matrix_without_scale(init_camera_matrix)

        eff_scale_mode = 'Z_DEPTH' if is_obj and props.scale_mode != 'NONE' else props.scale_mode
        if is_obj:
            depth = max(0.1, (init_target_matrix.translation - init_camera_matrix.translation).length)
        elif props.clip_depth_object:
            origin_depth = max(0.1, (props.clip_depth_object.matrix_world.translation - init_camera_matrix.translation).length)
            track_depth = self.estimate_track_group_depth(
                context,
                cam_ref,
                clip,
                tracks,
                ref_f,
                props.clip_depth_object,
            )
            depth = max(origin_depth, track_depth) if eff_scale_mode == 'Z_DEPTH' and track_depth is not None else origin_depth
        else:
            depth = 1.0

        frame_markers = self._collect_clip_track_markers(clip, tracks, frame_start, frame_end)
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

        tan_ref_x, tan_ref_y = get_camera_tan(cam_ref.data, ref_lens, context.scene)
        aspect = tan_ref_x / tan_ref_y if tan_ref_y > 1e-6 else 1.0
        norm_curve = self._build_clip_track_scale_curve(
            frame_markers,
            frame_start,
            frame_end,
            ref_f,
            aspect,
        )
        return self.execute_clip_track_refined(
            context,
            target,
            clip,
            tracks,
            cam_ref,
            ref_f,
            frame_start,
            frame_end,
            depth,
            norm_curve,
            frame_markers,
            ref_lens,
            eff_scale_mode,
        )
