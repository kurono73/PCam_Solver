# pcam_solver main bake operator
from .common import *
from .animation_io import PCamAnimationIO
from .solver_clip import PCamClipTrackSolver
from .follow_track import PCamFollowTrackMixin
from .solver_points import PCamPointSolver

class OBJECT_OT_apply_tracking_data(PCamAnimationIO, PCamClipTrackSolver, PCamFollowTrackMixin, PCamPointSolver, bpy.types.Operator):
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

