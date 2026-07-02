# pcam_solver Clip Track solver
from .common import *

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
            object_anchor_local = init_t_mat.inverted() @ ref_centroid_world

            scale_ratio = median_edge_scale(ref_locals, curr_locals)
            if scale_ratio is None:
                ref_dist = point_cloud_avg_distance(ref_locals)
                curr_dist = point_cloud_avg_distance(curr_locals)
                scale_ratio = curr_dist / ref_dist if ref_dist > 1e-6 else 1.0
            if scale_ratio is None or scale_ratio <= 1e-6:
                scale_ratio = 1.0

            target.location = adjust_location_depth_along_camera_axis(cam_mat_curr, target.location, scale_ratio)

            ref_centroid_local = weighted_points_centroid(ref_locals, weights)
            curr_centroid_local = weighted_points_centroid(curr_locals, weights)
            ref_vecs = [point - ref_centroid_local for point in ref_locals]
            curr_vecs = [point - curr_centroid_local for point in curr_locals]
            curr_to_ref_quat = solve_weighted_kabsch_rotation(ref_vecs, curr_vecs, False, weights)
            local_delta_quat = curr_to_ref_quat.inverted()
            solved_quat = curr_depth_mat.to_quaternion() @ local_delta_quat @ ref_depth_quat_inv @ init_t_rot
            target.location = object_location_from_local_anchor(curr_centroid_world, object_anchor_local, solved_quat, init_t_scale)
            target.location = adjust_location_depth_along_camera_axis(cam_mat_curr, target.location, scale_ratio)
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

