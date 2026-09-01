# pcam_solver properties
from .common import *
from .preview import update_track_preview, update_custom_range_preview, update_existing_position_lock, update_reference_frame_lock

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
        description="Bake only within a manually specified frame range while preserving keys outside that range",
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

