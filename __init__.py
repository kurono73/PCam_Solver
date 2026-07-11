# pcam_solver
import bpy

from .properties import PCamSolveProperties
from .operators_misc import (
    OBJECT_OT_set_pcam_solve_bake_start,
    OBJECT_OT_set_pcam_solve_bake_end,
    OBJECT_OT_get_pcam_solve_selected_tracks,
    OBJECT_OT_add_pcam_solve_depth_plane,
)
from .operators_apply import OBJECT_OT_apply_tracking_data
from .ui import VIEW3D_PT_pcam_solve_panel
from .preview import remove_preview_handler

classes = (
    PCamSolveProperties,
    OBJECT_OT_set_pcam_solve_bake_start,
    OBJECT_OT_set_pcam_solve_bake_end,
    OBJECT_OT_get_pcam_solve_selected_tracks,
    OBJECT_OT_add_pcam_solve_depth_plane,
    OBJECT_OT_apply_tracking_data,
    VIEW3D_PT_pcam_solve_panel,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.pcam_solve_props = bpy.props.PointerProperty(type=PCamSolveProperties)


def unregister():
    remove_preview_handler()
    if hasattr(bpy.types.Scene, "pcam_solve_props"):
        del bpy.types.Scene.pcam_solve_props
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
