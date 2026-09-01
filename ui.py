# pcam_solver UI panel
from .common import *
from .operators_misc import OBJECT_OT_set_pcam_solve_bake_start, OBJECT_OT_set_pcam_solve_bake_end, OBJECT_OT_get_pcam_solve_selected_tracks, OBJECT_OT_add_pcam_solve_depth_plane
from .operators_apply import OBJECT_OT_apply_tracking_data

class VIEW3D_PT_pcam_solve_panel(bpy.types.Panel):
    bl_label = "Pseudo Camera Solver"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'P-Cam' 
    
    def draw(self, context):
        layout = self.layout
        props = context.scene.pcam_solve_props
        is_cam = props.apply_to == 'CAMERA'
        capabilities = pcam_get_solve_capabilities(props)

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
             
        if capabilities.depth_reference_required:
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
        if props.mode != 'CLIP_TRACK':
            row.operator(OBJECT_OT_get_pcam_solve_selected_tracks.bl_idname, text="", icon='FILE_REFRESH')
        
        if props.target_clip: 
            labeled_prop(b_track, props, "tracking_object_idx", "Track Layer")
             
            if capabilities.uses_follow_track:
                b_track.prop(props, "use_undistort")
                tr = b_track.box()
                try: track_pool = props.target_clip.tracking.objects[int(props.tracking_object_idx)]
                except Exception: track_pool = None
                
                if track_pool:
                    tr.prop_search(props, "track_1", track_pool, "tracks")
                    if capabilities.track_count >= 2:
                        tr.prop_search(props, "track_2", track_pool, "tracks")
                    if capabilities.track_count >= 3:
                        tr.prop_search(props, "track_3", track_pool, "tracks")
                else:
                    tr.prop(props, "track_1")
                    if capabilities.track_count >= 2: tr.prop(props, "track_2")
                    if capabilities.track_count >= 3: tr.prop(props, "track_3")

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
                if props.scale_mode == 'NONE':
                    row.prop(props, "clip_lock_roll", text="Lock Roll")
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
            roll_control = row.row(align=True)
            roll_control.enabled = not (
                props.scale_mode == 'NONE' and props.clip_lock_roll
            )
            roll_control.prop(props, "clip_roll_smooth", text="Roll Smooth")

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

        existing_position_supported = capabilities.existing_position_supported
        existing_position_reason = capabilities.existing_position_reason
        if props.mode == 'ONE_POINT' and is_cam:
            row = b_bake.row(align=True)
            row.enabled = existing_position_supported
            row.prop(props, "clip_use_existing_position", text="Use Existing Position")
        elif props.mode == 'CLIP_TRACK' and is_cam:
            row = b_bake.row(align=True)
            pos_row = row.row(align=True)
            pos_row.enabled = existing_position_supported
            pos_row.prop(props, "clip_use_existing_position", text="Use Existing Position")
            if capabilities.existing_focal_supported:
                row.prop(props, "clip_use_existing_focal", text="Use Existing Focal")
        elif props.mode in {'TWO_POINT', 'THREE_POINT'} and is_cam:
            row = b_bake.row(align=True)
            pos_row = row.row(align=True)
            pos_row.enabled = existing_position_supported
            pos_row.prop(props, "clip_use_existing_position", text="Use Existing Position")
            if capabilities.existing_focal_supported:
                if capabilities.existing_focal_required_with_position and props.clip_use_existing_position:
                    focal_row = row.row(align=True)
                    focal_row.enabled = False
                    focal_row.label(text="Use Existing Focal", icon='CHECKBOX_HLT')
                else:
                    row.prop(props, "clip_use_existing_focal", text="Use Existing Focal")

        if is_cam and props.clip_use_existing_position and not existing_position_supported:
            b_bake.label(text=existing_position_reason, icon='INFO')

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

