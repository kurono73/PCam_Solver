# pcam_solver Follow Track extraction helpers
from .common import *

class PCamFollowTrackMixin:
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

    def create_static_follow_depth_plane(self, context, matrix_world, distance=10.0):
        dist = max(float(distance), 1.0)
        size = max(dist * 100.0, 1000.0)
        mesh = bpy.data.meshes.new("PCam_FollowTrack_DepthPlaneMesh")
        mesh.from_pydata(
            [
                (-size, -size, -dist),
                ( size, -size, -dist),
                ( size,  size, -dist),
                (-size,  size, -dist),
            ],
            [],
            [(0, 1, 2, 3)],
        )
        mesh.update()
        plane = bpy.data.objects.new("PCam_FollowTrack_DepthPlane", mesh)
        context.scene.collection.objects.link(plane)
        plane.matrix_world = matrix_without_scale(matrix_world)
        plane.hide_render = True
        plane.hide_select = True
        context.view_layer.update()
        return plane

    def remove_static_follow_depth_plane(self, plane):
        if plane is None:
            return
        mesh = getattr(plane, "data", None)
        bpy.data.objects.remove(plane, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)

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

