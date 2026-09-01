# Solve-mode capabilities shared by validation, UI, and bake paths.
from dataclasses import dataclass


@dataclass(frozen=True)
class PCamSolveCapabilities:
    track_count: int
    uses_follow_track: bool
    depth_reference_required: bool
    rotation_only_tripod: bool
    existing_position_supported: bool
    existing_position_reason: str
    existing_focal_supported: bool
    existing_focal_required_with_position: bool


def pcam_get_solve_capabilities(props):
    mode = props.mode
    is_camera = props.apply_to == 'CAMERA'
    scale_mode = props.scale_mode
    tripod = props.tripod_mode

    track_count = {
        'ONE_POINT': 1,
        'TWO_POINT': 2,
        'THREE_POINT': 3,
        'CLIP_TRACK': 0,
    }.get(mode, 1)
    uses_follow_track = mode in {'ONE_POINT', 'TWO_POINT', 'THREE_POINT'}

    if mode == 'ONE_POINT':
        depth_required = not tripod
    elif mode in {'TWO_POINT', 'THREE_POINT', 'CLIP_TRACK'}:
        if not is_camera:
            depth_required = True
        elif scale_mode == 'NONE':
            depth_required = not tripod
        else:
            depth_required = scale_mode in {'Z_DEPTH', 'FOCAL_LENGTH'}
    else:
        depth_required = False

    rotation_only_tripod = is_camera and tripod and not depth_required
    existing_position_supported = is_camera
    existing_position_reason = ""
    if not is_camera:
        existing_position_supported = False
        existing_position_reason = "Available for Camera targets only."
    elif rotation_only_tripod:
        existing_position_supported = False
        existing_position_reason = "Unavailable in rotation-only Tripod mode."
    elif mode in {'TWO_POINT', 'THREE_POINT', 'CLIP_TRACK'} and tripod and scale_mode == 'FOCAL_LENGTH':
        existing_position_supported = False
        existing_position_reason = "Unavailable with Tripod + Focal Length."

    existing_focal_supported = (
        is_camera and
        mode in {'TWO_POINT', 'THREE_POINT', 'CLIP_TRACK'} and
        scale_mode == 'FOCAL_LENGTH'
    )
    existing_focal_required_with_position = (
        existing_position_supported and
        mode in {'TWO_POINT', 'THREE_POINT'} and
        scale_mode == 'FOCAL_LENGTH'
    )

    return PCamSolveCapabilities(
        track_count=track_count,
        uses_follow_track=uses_follow_track,
        depth_reference_required=depth_required,
        rotation_only_tripod=rotation_only_tripod,
        existing_position_supported=existing_position_supported,
        existing_position_reason=existing_position_reason,
        existing_focal_supported=existing_focal_supported,
        existing_focal_required_with_position=existing_focal_required_with_position,
    )


def pcam_required_track_count(props):
    return pcam_get_solve_capabilities(props).track_count


def pcam_depth_reference_required(props):
    return pcam_get_solve_capabilities(props).depth_reference_required


def pcam_existing_position_support(props):
    capabilities = pcam_get_solve_capabilities(props)
    return capabilities.existing_position_supported, capabilities.existing_position_reason


def pcam_use_existing_position(props):
    capabilities = pcam_get_solve_capabilities(props)
    return bool(props.clip_use_existing_position and capabilities.existing_position_supported)
