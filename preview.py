# pcam_solver preview helpers
import gpu
from gpu_extras.batch import batch_for_shader

from .common import *

_handle_3d = None

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
            v_cam = marker_to_camera_ray(marker_co, tan_x, tan_y, cam_eval.data)
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
                if obj_eval is None and not pcam_depth_reference_required(props):
                    points_hit.append(hit_loc)
                else:
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


def remove_preview_handler():
    global _handle_3d
    if _handle_3d:
        try:
            bpy.types.SpaceView3D.draw_handler_remove(_handle_3d, 'WINDOW')
        except Exception:
            pass
        _handle_3d = None

