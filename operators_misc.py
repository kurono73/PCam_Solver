# pcam_solver small operators
from .common import *
from .preview import update_custom_range_preview

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

