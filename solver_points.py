# pcam_solver 1/2/3 point solvers
from .common import *

class PCamPointSolver:
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
        return self.execute_point_none(context, target, [props.track_1], "1-point")

    def project_delta_parallel_to_depth(self, context, delta, depth_obj):
        if depth_obj is None or delta.length_squared <= 1e-12:
            return delta.copy()
        normal = evaluated_matrix_world(context, depth_obj).to_quaternion() @ Vector((0.0, 0.0, 1.0))
        if normal.length_squared <= 1e-12:
            return delta.copy()
        normal.normalize()
        return delta - normal * delta.dot(normal)

    def follow_track_dirs_for_frame(self, track_data, track_names, frame, cam_matrix):
        cam_inv_rot = cam_matrix.to_quaternion().inverted()
        cam_loc = cam_matrix.translation
        dirs = []
        for index, _track_name in enumerate(track_names):
            point = track_data[index].get(frame) if index < len(track_data) else None
            if point is None:
                continue
            vec = point - cam_loc
            if vec.length_squared <= 1e-9:
                continue
            dirs.append((cam_inv_rot @ vec).normalized())
        return dirs

    def follow_track_points_stuck_at_camera(self, track_data, cam_loc, frames, eps=1e-10):
        if not track_data:
            return False
        total = 0
        stuck = 0
        for data in track_data:
            for frame in frames:
                point = data.get(frame)
                if point is None:
                    continue
                total += 1
                if (point - cam_loc).length_squared <= eps:
                    stuck += 1
        return total > 0 and stuck == total

    def report_follow_track_origin_failure(self, context):
        cam = context.scene.camera
        loc = cam.location if cam else Vector((0.0, 0.0, 0.0))
        if loc.length_squared <= 1e-12:
            self.report({'ERROR'}, "Follow Track evaluation failed at exact camera origin. Move the camera slightly from (0, 0, 0) and bake again.")
        else:
            self.report({'ERROR'}, "Follow Track evaluation stayed at the camera position. Check the camera layout and trackers.")

    def execute_point_none(self, context, target, track_names, label):
        props = context.scene.pcam_solve_props
        clip = props.target_clip
        cam_ref = context.scene.camera
        if not clip or not cam_ref:
            return {'CANCELLED'}

        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        is_tripod = props.tripod_mode
        if not is_tripod and not props.clip_depth_object:
            self.report({'ERROR'}, "Depth Reference is required for non-tripod None solving.")
            return {'CANCELLED'}

        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        init_t_mat = matrix_without_scale(target.matrix_world.copy())
        init_t_loc = init_t_mat.to_translation()
        init_t_rot = init_t_mat.to_quaternion()
        init_t_euler = target.rotation_euler.copy()
        follow_cam = self.create_static_follow_camera(context, cam_ref, init_t_mat) if cam_ref else None
        extract_cam = follow_cam or cam_ref
        frame_range = (frame_start, frame_end) if props.use_custom_range else None
        target_curve_snapshot = self.snapshot_animation_action(target)
        lens_curve_snapshot = self.snapshot_animation_action(target.data) if getattr(target, "data", None) is not None else []
        has_existing_focal_keys = self.has_camera_focal_length_keys(cam_ref)
        has_focal_variation_in_range = self.camera_lens_varies_over_range(context, cam_ref, frame_start, frame_end) if props.use_custom_range else False
        pin_existing_focal_range = props.use_custom_range and (has_existing_focal_keys or has_focal_variation_in_range)
        pinned_lens_value = float(target.data.lens) if getattr(target, "data", None) is not None else None

        temp_depth_obj = self.create_static_follow_depth_plane(context, init_t_mat) if is_tripod else None
        try:
            track_data = self.extract_tracks_data(
                context,
                extract_cam,
                clip,
                track_names,
                temp_depth_obj if is_tripod else props.clip_depth_object,
                props.use_undistort,
                props.track_smoothing,
            )
        finally:
            self.remove_static_follow_camera(follow_cam)
            self.remove_static_follow_depth_plane(temp_depth_obj)

        if is_tripod:
            valid_frames = []
            if track_data:
                common = set(range(frame_start, frame_end + 1))
                for data in track_data:
                    common &= set(data.keys())
                for f in sorted(common):
                    dirs = self.follow_track_dirs_for_frame(track_data, track_names, f, init_t_mat)
                    if len(dirs) == len(track_names):
                        valid_frames.append(f)
        else:
            if not track_data:
                valid_frames = []
            else:
                common = set(range(frame_start, frame_end + 1))
                for data in track_data:
                    common &= set(data.keys())
                valid_frames = sorted(common)

        if self.follow_track_points_stuck_at_camera(track_data, init_t_loc, range(frame_start, frame_end + 1)):
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            self.report_follow_track_origin_failure(context)
            return {'CANCELLED'}

        ref_f = pcam_pick_valid_reference_frame(valid_frames, ref_hint, props.use_reference_frame_lock)
        if ref_f is None:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            if init_t_loc.length_squared <= 1e-12:
                self.report_follow_track_origin_failure(context)
            else:
                self.report({'ERROR'}, f"Reference Frame has no valid {label} tracker data.")
            return {'CANCELLED'}

        self.clear_animation_safely(target, (frame_start, frame_end) if props.use_custom_range else None)
        if pin_existing_focal_range and getattr(target, "data", None):
            self.pin_lens_constant_in_range(target.data, frame_start, frame_end, pinned_lens_value, lens_curve_snapshot)

        ref_points = None
        ref_rays = None
        ref_weights = None
        if is_tripod:
            ref_rays = self.follow_track_dirs_for_frame(track_data, track_names, ref_f, init_t_mat)
            ref_weights = [1.0] * len(ref_rays)
        else:
            ref_points = [data[ref_f].copy() for data in track_data]
            ref_center = sum(ref_points, Vector()) / len(ref_points)

        baked_frames = 0
        for f in range(frame_start, frame_end + 1):
            if f not in valid_frames:
                continue
            context.scene.frame_set(f)
            if is_tripod:
                curr_rays = self.follow_track_dirs_for_frame(track_data, track_names, f, init_t_mat)
                weights = [1.0] * len(curr_rays)
                usable_count = min(len(ref_rays), len(curr_rays), len(weights))
                if usable_count <= 0:
                    continue
                lock_roll_for_mode = props.clip_lock_roll or len(track_names) < 2
                if len(track_names) < 2:
                    solved_quat = solve_single_ray_euler_y_locked(
                        init_t_rot,
                        ref_rays[0],
                        curr_rays[0],
                        init_t_euler,
                        target.rotation_mode,
                    )
                else:
                    if lock_roll_for_mode:
                        weight_sum = sum(weights[:usable_count])
                        if weight_sum <= 1e-9:
                            continue
                        ref_center_ray = sum(
                            (ray * weight for ray, weight in zip(ref_rays[:usable_count], weights[:usable_count])),
                            Vector((0.0, 0.0, 0.0)),
                        ) / weight_sum
                        curr_center_ray = sum(
                            (ray * weight for ray, weight in zip(curr_rays[:usable_count], weights[:usable_count])),
                            Vector((0.0, 0.0, 0.0)),
                        ) / weight_sum
                        solved_quat = solve_single_ray_euler_y_locked(
                            init_t_rot,
                            ref_center_ray,
                            curr_center_ray,
                            init_t_euler,
                            target.rotation_mode,
                        )
                    else:
                        delta_quat = solve_weighted_kabsch_rotation(
                            ref_rays[:usable_count],
                            curr_rays[:usable_count],
                            False,
                            weights[:usable_count],
                        )
                        solved_quat = init_t_rot @ delta_quat
                target.location = init_t_loc
                self.set_target_rotation(target, solved_quat)
            else:
                curr_points = [data[f].copy() for data in track_data]
                curr_center = sum(curr_points, Vector()) / len(curr_points)
                delta = self.project_delta_parallel_to_depth(context, curr_center - ref_center, props.clip_depth_object)
                target.location = init_t_loc - delta
                solved_quat = init_t_rot
                if len(track_names) >= 2 and not props.clip_lock_roll:
                    init_inv = init_t_mat.inverted()
                    init_rot_inv = init_t_rot.inverted()
                    ref_xy = []
                    curr_xy = []
                    for ref_point, curr_point in zip(ref_points, curr_points):
                        ref_local = init_inv @ ref_point
                        curr_local = init_rot_inv @ (curr_point - target.location)
                        ref_xy.append(Vector((ref_local.x, ref_local.y)))
                        curr_xy.append(Vector((curr_local.x, curr_local.y)))
                    roll_delta = solve_planar_roll_from_points(ref_xy, curr_xy)
                    if abs(roll_delta) > 1e-9:
                        axis = init_t_rot @ Vector((0.0, 0.0, 1.0))
                        solved_quat = Quaternion(axis, -roll_delta) @ init_t_rot
                self.set_target_rotation(target, solved_quat)
            target.keyframe_insert("location", frame=f)
            self.keyframe_target_rotation(target, f)
            baked_frames += 1

        if baked_frames == 0:
            self.restore_animation_snapshot_exact(target, target_curve_snapshot)
            if getattr(target, "data", None) is not None:
                self.restore_animation_snapshot_exact(target.data, lens_curve_snapshot)
            self.report({'ERROR'}, f"No frames could be baked from {label} trackers.")
            return {'CANCELLED'}

        context.scene.frame_set(ref_f)
        total_frames = frame_end - frame_start + 1
        suffix = f" Solved {baked_frames}/{total_frames} frames." if baked_frames < total_frames else ""
        self.report({'INFO'}, f"Applied {label} None motion to '{target.name}'.{suffix}")
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
        if not is_obj and props.scale_mode == 'NONE':
            return self.execute_point_none(context, target, [props.track_1, props.track_2], "2-point")
        eff_depth_obj = props.clip_depth_object if props.clip_depth_object else (target if is_obj else None)

        clip = props.target_clip
        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        ref_cam_mat = matrix_without_scale(evaluated_matrix_world(context, cam_ref)) if cam_ref else None
        follow_cam = self.create_static_follow_camera(context, cam_ref, ref_cam_mat) if cam_ref and not is_obj else None

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
        center_start_ref = (p1_start + p2_start) / 2.0
        object_anchor_local = init_t_mat.inverted() @ center_start_ref if is_obj else None
        fixed_world_points = {
            props.track_1: p1_start.copy(),
            props.track_2: p2_start.copy(),
        }
        depth_ref_mat = evaluated_matrix_world(context, props.clip_depth_object) if is_obj and props.clip_depth_object else None
        depth_ref_inv = depth_ref_mat.inverted() if depth_ref_mat is not None else None
        depth_ref_quat_inv = depth_ref_mat.to_quaternion().inverted() if depth_ref_mat is not None else None
        camera_anchor_start = None
        if not is_obj and props.clip_depth_object:
            camera_anchor_start = raycast_marker_world_from_matrix(
                context,
                cam_ref.data,
                init_t_mat,
                props.clip_depth_object,
                Vector((0.5, 0.5)),
                init_f_len,
            )
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
            center_start = (p1_start + p2_start) / 2.0
            center_curr = (p1_curr + p2_curr) / 2.0
            camera_anchor_from = center_start
            camera_anchor_to = center_curr
            if not is_obj and camera_anchor_start is not None:
                mapped_anchor = map_anchor_by_segment_in_camera_space(
                    p1_start,
                    p2_start,
                    p1_curr,
                    p2_curr,
                    camera_anchor_start,
                    init_t_mat,
                )
                if mapped_anchor is not None:
                    camera_anchor_from = camera_anchor_start
                    camera_anchor_to = mapped_anchor
                
            if is_obj:
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
                if object_anchor_local is not None:
                    target.location = object_location_from_local_anchor(center_curr, object_anchor_local, solved_obj_quat, init_t_scale)
                else:
                    target.location = init_t_loc + (center_curr - center_start)
                
                scale_ratio = depth_local_scale_ratio if depth_local_scale_ratio is not None else (vec_curr.length / vec_start.length if vec_start.length > 0 else 1.0)
                cam_mat_curr = evaluated_matrix_world(context, cam_ref)
                target.location = adjust_location_depth_along_camera_axis(cam_mat_curr, target.location, scale_ratio)
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
                        solved_focal_rotation = False
                        if props.scale_mode == 'FOCAL_LENGTH':
                            ref_lens_for_rotation = existing_lens_curve.get(ref_f, init_f_len) if keep_existing_focal else init_f_len
                            if keep_existing_focal:
                                frame_lens_for_rotation = existing_lens_curve.get(f, ref_lens_for_rotation)
                            elif suppress_focal_bake:
                                frame_lens_for_rotation = init_f_len
                            else:
                                frame_lens_for_rotation = init_f_len * scale_ratio
                            delta_quat = solve_focal_tripod_rotation_from_markers(
                                context,
                                cam_ref.data,
                                props.target_clip,
                                props.tracking_object_idx,
                                [props.track_1, props.track_2],
                                ref_f,
                                f,
                                ref_lens_for_rotation,
                                frame_lens_for_rotation,
                                props.clip_lock_roll,
                            )
                            if delta_quat is not None:
                                target.location = init_t_loc
                                solved_quat = init_t_rot @ delta_quat
                                if props.clip_lock_roll:
                                    solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                                self.set_target_rotation(target, solved_quat)
                                solved_focal_rotation = True

                        if not solved_focal_rotation:
                            vec_pt_start = camera_anchor_from - target.location
                            init_cam_matrix_inv = init_t_mat.inverted()
                            center_local_curr = init_cam_matrix_inv @ camera_anchor_to
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
                    center_start_local = init_cam_matrix_inv @ camera_anchor_from
                    center_curr_local = init_cam_matrix_inv @ camera_anchor_to
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
                    depth_start = (camera_anchor_from - init_t_loc).length
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
        if not is_obj and props.scale_mode == 'NONE':
            return self.execute_point_none(context, target, [props.track_1, props.track_2, props.track_3], "3-point")
        eff_scale_mode = 'Z_DEPTH' if is_obj and props.scale_mode != 'NONE' else props.scale_mode
        eff_depth_obj = props.clip_depth_object if props.clip_depth_object else (target if is_obj else None)

        clip = props.target_clip
        frame_start = props.bake_start if props.use_custom_range else clip.frame_start + clip.frame_offset
        frame_end = props.bake_end if props.use_custom_range else clip.frame_start + clip.frame_duration - 1 + clip.frame_offset
        ref_hint = pcam_get_reference_frame(context, props, frame_start, frame_end)
        context.scene.frame_set(ref_hint)
        context.view_layer.update()
        ref_cam_mat = matrix_without_scale(evaluated_matrix_world(context, cam_ref)) if cam_ref else None
        follow_cam = self.create_static_follow_camera(context, cam_ref, ref_cam_mat) if cam_ref and not is_obj else None

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
        centroid_start_ref = sum(points_start_ref, Vector()) / 3.0
        object_anchor_local = init_t_mat.inverted() @ centroid_start_ref if is_obj else None
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
            camera_anchor_from = camera_axis_plane_anchor(points_start, init_t_loc, init_t_rot) if not is_obj else None
            if camera_anchor_from is None:
                camera_anchor_from = centroid_from
            camera_anchor_to = transform_matrix @ camera_anchor_from

            if is_obj:
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
                    vec_cam_to_obj = centroid_to - cam_mat_curr.translation
                    if vec_cam_to_obj.length > 1e-9:
                        view_axis = vec_cam_to_obj.normalized()
                        solved_obj_rotation = Quaternion(view_axis, roll_delta) @ init_t_rot

                if solved_obj_rotation is not None:
                    if object_anchor_local is not None:
                        target.location = object_location_from_local_anchor(centroid_to, object_anchor_local, solved_obj_rotation, init_t_scale)
                    else:
                        target.location = init_t_loc + (centroid_to - centroid_from)
                    target.location = adjust_location_depth_along_camera_axis(cam_mat_curr, target.location, scale_ratio)
                    solved_quat = self.set_target_rotation_continuous(
                        target,
                        solved_obj_rotation,
                        prev_obj_quat,
                        prev_obj_euler,
                    )
                    prev_obj_quat = solved_quat.copy()
                    if prev_obj_euler is not None:
                        prev_obj_euler = target.rotation_euler.copy()
                else:
                    target.location = init_t_loc + (centroid_to - centroid_from)
                    target.location = adjust_location_depth_along_camera_axis(cam_mat_curr, target.location, scale_ratio)
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
                        solved_focal_rotation = False
                        ref_lens_for_rotation = existing_lens_curve.get(ref_f, init_f_len) if keep_existing_focal else init_f_len
                        if keep_existing_focal:
                            frame_lens_for_rotation = existing_lens_curve.get(f, ref_lens_for_rotation)
                        elif suppress_focal_bake:
                            frame_lens_for_rotation = init_f_len
                        else:
                            frame_lens_for_rotation = init_f_len * scale_ratio
                        delta_quat = solve_focal_tripod_rotation_from_markers(
                            context,
                            cam_ref.data,
                            props.target_clip,
                            props.tracking_object_idx,
                            [props.track_1, props.track_2, props.track_3],
                            ref_f,
                            f,
                            ref_lens_for_rotation,
                            frame_lens_for_rotation,
                            props.clip_lock_roll,
                        )
                        if delta_quat is not None:
                            solved_quat = init_t_rot @ delta_quat
                            if props.clip_lock_roll:
                                solved_quat = preserve_camera_roll_from_reference(solved_quat, init_t_rot)
                            self.set_target_rotation(target, solved_quat)
                            solved_focal_rotation = True

                        if not solved_focal_rotation:
                            anchor_curr_local = init_cam_inv @ camera_anchor_to
                            anchor_curr_local_unzoomed = Vector((
                                anchor_curr_local.x / scale_ratio if scale_ratio > 1e-6 else anchor_curr_local.x,
                                anchor_curr_local.y / scale_ratio if scale_ratio > 1e-6 else anchor_curr_local.y,
                                anchor_curr_local.z
                            ))
                            anchor_curr_unzoomed = init_t_mat @ anchor_curr_local_unzoomed
                            vec_pt_start = camera_anchor_from - init_t_loc
                            vec_pt_curr_unzoomed = anchor_curr_unzoomed - init_t_loc
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
                        centroid_start_local = init_cam_inv @ camera_anchor_from
                        centroid_curr_local = init_cam_inv @ camera_anchor_to
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
                    lens_for_rotation = (
                        existing_lens_curve.get(f, init_f_len) if keep_existing_focal else
                        init_f_len if suppress_focal_bake else
                        init_f_len * scale_ratio
                    )
                    marker_curr_list = [
                        get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_1, f),
                        get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_2, f),
                        get_track_marker_co(props.target_clip, props.tracking_object_idx, props.track_3, f),
                    ]
                    if not any(marker is None for marker in marker_curr_list):
                        tan_x, tan_y = get_camera_tan(cam_ref.data, lens_for_rotation, context.scene)
                        rays_local = [marker_to_camera_ray(marker, tan_x, tan_y) for marker in marker_curr_list]
                        weights = None
                        if props.clip_center_weight:
                            aspect = tan_x / tan_y if tan_y > 1e-6 else 1.0
                            weights = [marker_center_weight(marker, aspect) for marker in marker_curr_list]
                        refined_quat = solve_rotation_quat_at_location(
                            points_start_ref,
                            rays_local,
                            target.location.copy(),
                            self.get_target_rotation_quaternion(target),
                            props.clip_lock_roll,
                            weights,
                            prefer_center=True,
                        )
                        if props.clip_lock_roll:
                            refined_quat = preserve_camera_roll_from_reference(refined_quat, init_t_rot)
                        self.set_target_rotation(target, refined_quat)
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
                                vec_pt_start = camera_anchor_from - init_t_loc
                                vec_pt_curr = camera_anchor_to - init_t_loc
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
                        centroid_start_local = init_cam_inv @ camera_anchor_from
                        centroid_curr_local = init_cam_inv @ camera_anchor_to
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
                        depth_start = (camera_anchor_from - init_t_mat.to_translation()).length
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

