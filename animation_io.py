# pcam_solver animation I/O
from bpy_extras import anim_utils

from .common import *

class PCamAnimationIO:
    def _snapshot_rna_values(self, value):
        result = {}
        for prop in value.bl_rna.properties:
            if prop.is_readonly or prop.identifier in {'rna_type', 'type'}:
                continue
            if prop.type not in {'BOOLEAN', 'INT', 'FLOAT', 'STRING', 'ENUM'}:
                continue
            data = getattr(value, prop.identifier)
            result[prop.identifier] = tuple(data) if getattr(prop, 'is_array', False) else data
        return result

    def _restore_rna_values(self, value, values):
        # Array sizes (notably Generator coefficients) depend on scalar settings.
        for arrays in (False, True):
            for name, data in values.items():
                if isinstance(data, tuple) == arrays:
                    setattr(value, name, data)

    # Animation I/O helpers. These wrap Blender 4.x legacy fcurves and Blender
    # 5.x action slots/channelbags behind the same local API.
    def clear_keyframes_in_range(self, id_data, data_paths, frame_start, frame_end):
        channelbags = self._iter_action_channelbags(id_data)
        if not channelbags:
            return
        for channelbag in channelbags:
            fcurves = getattr(channelbag, "fcurves", None)
            if fcurves is None:
                continue
            for fcurve in list(fcurves):
                if fcurve.data_path not in data_paths:
                    continue
                remove_indices = [
                    i for i, key in enumerate(fcurve.keyframe_points)
                    if frame_start <= key.co.x <= frame_end
                ]
                for i in reversed(remove_indices):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[i])
                if not fcurve.keyframe_points:
                    fcurves.remove(fcurve)

    def clear_animation_channels(self, id_data, keep_paths=None, data_paths=None):
        keep_paths = set(keep_paths or ())
        data_paths = set(data_paths) if data_paths is not None else None
        channelbags = self._iter_action_channelbags(id_data)
        if not channelbags:
            return
        for channelbag in channelbags:
            fcurves = getattr(channelbag, "fcurves", None)
            if fcurves is None:
                continue
            for fcurve in list(fcurves):
                if fcurve.data_path in keep_paths or (data_paths is not None and fcurve.data_path not in data_paths):
                    continue
                fcurves.remove(fcurve)

    def snapshot_animation_curves(self, id_data, data_paths):
        data_paths = set(data_paths or ())
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None or not data_paths:
            return []
        snapshots = []
        for fcurve in fcurves:
            if fcurve.data_path not in data_paths:
                continue
            keys = []
            for key in fcurve.keyframe_points:
                key_data = {
                    "co": (float(key.co.x), float(key.co.y)),
                    "handle_left": (float(key.handle_left.x), float(key.handle_left.y)),
                    "handle_right": (float(key.handle_right.x), float(key.handle_right.y)),
                    "interpolation": key.interpolation,
                    "handle_left_type": key.handle_left_type,
                    "handle_right_type": key.handle_right_type,
                }
                if hasattr(key, "easing"):
                    key_data["easing"] = key.easing
                if hasattr(key, "back"):
                    key_data["back"] = float(key.back)
                if hasattr(key, "amplitude"):
                    key_data["amplitude"] = float(key.amplitude)
                if hasattr(key, "period"):
                    key_data["period"] = float(key.period)
                keys.append(key_data)
            snapshots.append({
                "data_path": fcurve.data_path,
                "array_index": fcurve.array_index,
                "extrapolation": fcurve.extrapolation,
                "keys": keys,
                "settings": self._snapshot_rna_values(fcurve),
                "group": fcurve.group.name if fcurve.group else "",
                "modifiers": [
                    {
                        "type": modifier.type,
                        "settings": self._snapshot_rna_values(modifier),
                        "control_points": [self._snapshot_rna_values(point) for point in modifier.control_points]
                        if modifier.type == 'ENVELOPE' else [],
                    }
                    for modifier in fcurve.modifiers
                ],
            })
        return snapshots

    def snapshot_animation_action(self, id_data):
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return []
        return self.snapshot_animation_curves(id_data, {fcurve.data_path for fcurve in fcurves})

    def copy_animation_action(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if not action:
            return None
        backup = action.copy()
        backup.use_fake_user = False
        return backup

    def _get_action_slot(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        if not anim_data:
            return None
        slot = getattr(anim_data, "action_slot", None)
        if slot is not None:
            return slot
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        slots = getattr(action, "slots", None)
        if slots is None:
            return None
        try:
            return slots[0] if len(slots) == 1 else None
        except Exception:
            return None

    def _iter_action_channelbags(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return []

        legacy_fcurves = getattr(action, "fcurves", None)
        if legacy_fcurves is not None:
            return [action]

        get_channelbag = getattr(anim_utils, "action_get_channelbag_for_slot", None)
        slot = self._get_action_slot(id_data)
        if slot is None or get_channelbag is None:
            return []
        try:
            channelbag = get_channelbag(action, slot)
        except Exception:
            channelbag = None
        if channelbag is None or getattr(channelbag, "fcurves", None) is None:
            return []
        return [channelbag]

    def _iter_action_fcurves(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        legacy_fcurves = getattr(action, "fcurves", None)
        if legacy_fcurves is not None:
            return legacy_fcurves
        slot = self._get_action_slot(id_data)
        if slot is None:
            return None
        get_channelbag = getattr(anim_utils, "action_get_channelbag_for_slot", None)
        if get_channelbag is None:
            return None
        try:
            channelbag = get_channelbag(action, slot)
        except Exception:
            channelbag = None
        if channelbag is None:
            return None
        return getattr(channelbag, "fcurves", None)

    def _ensure_action_fcurve(self, id_data, data_path, index=0, group_name=""):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return None
        try:
            return action.fcurve_ensure_for_datablock(id_data, data_path, index=index, group_name=group_name)
        except TypeError:
            try:
                return action.fcurve_ensure_for_datablock(id_data, data_path, index=index)
            except Exception:
                pass
        except Exception:
            pass

        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return None
        try:
            return fcurves.ensure(data_path, index=index, group_name=group_name)
        except Exception:
            try:
                return fcurves.new(data_path, index=index, group_name=group_name)
            except TypeError:
                return fcurves.new(data_path, index=index)

    def has_camera_focal_length_keys(self, camera_obj):
        cam_data = getattr(camera_obj, "data", None)
        if cam_data is None:
            return False
        fcurves = self._iter_action_fcurves(cam_data)
        if fcurves is None:
            return False
        for fcurve in fcurves:
            if fcurve.data_path == "lens":
                return len(fcurve.keyframe_points) > 0
        return False

    def camera_lens_varies_over_range(self, context, camera_obj, frame_start, frame_end, epsilon=1e-6):
        cam_data = getattr(camera_obj, "data", None)
        if cam_data is None:
            return False
        restore_frame = context.scene.frame_current
        try:
            context.scene.frame_set(frame_start)
            base_value = float(cam_data.lens)
            for frame in range(frame_start + 1, frame_end + 1):
                context.scene.frame_set(frame)
                if abs(float(cam_data.lens) - base_value) > epsilon:
                    return True
            return False
        finally:
            context.scene.frame_set(restore_frame)

    def restore_animation_action_copy(self, id_data, action_copy):
        if action_copy is None:
            return
        anim_data = id_data.animation_data_create()
        original_action = anim_data.action
        original_slot = self._get_action_slot(id_data)
        slot_identifier = original_slot.identifier if original_slot is not None else None
        try:
            anim_data.action = action_copy
            if slot_identifier is not None:
                anim_data.action_slot = next(slot for slot in action_copy.slots if slot.identifier == slot_identifier)
            snapshots = self.snapshot_animation_action(id_data)
        finally:
            anim_data.action = original_action
            if original_slot is not None:
                anim_data.action_slot = original_slot
        # Restore only this datablock's slot; the Object may share the same Action.
        self.restore_animation_snapshot_exact(id_data, snapshots)
        if action_copy.users == 0:
            bpy.data.actions.remove(action_copy)

    def restore_animation_curves(self, id_data, snapshots):
        if not snapshots:
            return
        anim_data = id_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{id_data.name}_Action")
        for snap in snapshots:
            fcurves = self._iter_action_fcurves(id_data)
            if fcurves is not None:
                for fcurve in list(fcurves):
                    if fcurve.data_path == snap["data_path"] and fcurve.array_index == snap["array_index"]:
                        fcurves.remove(fcurve)
            fcurve = self._ensure_action_fcurve(id_data, snap["data_path"], index=snap["array_index"], group_name=snap.get("group", ""))
            if fcurve is None:
                continue
            fcurve.extrapolation = snap["extrapolation"]
            self._restore_rna_values(fcurve, snap.get("settings", {}))
            fcurve.keyframe_points.add(len(snap["keys"]))
            for key, key_data in zip(fcurve.keyframe_points, snap["keys"]):
                key.co = key_data["co"]
                key.interpolation = key_data["interpolation"]
                key.handle_left_type = key_data["handle_left_type"]
                key.handle_right_type = key_data["handle_right_type"]
                key.handle_left = key_data["handle_left"]
                key.handle_right = key_data["handle_right"]
                if "easing" in key_data and hasattr(key, "easing"):
                    key.easing = key_data["easing"]
                if "back" in key_data and hasattr(key, "back"):
                    key.back = key_data["back"]
                if "amplitude" in key_data and hasattr(key, "amplitude"):
                    key.amplitude = key_data["amplitude"]
                if "period" in key_data and hasattr(key, "period"):
                    key.period = key_data["period"]
            fcurve.update()
            for data in snap.get("modifiers", []):
                modifier = fcurve.modifiers.new(data["type"])
                self._restore_rna_values(modifier, data["settings"])
                for point_data in data["control_points"]:
                    point = modifier.control_points.add(point_data["frame"])
                    self._restore_rna_values(point, point_data)

    def restore_animation_snapshot_exact(self, id_data, snapshots):
        if id_data is None:
            return
        self.clear_animation_channels(id_data)
        self.restore_animation_curves(id_data, snapshots)

    def clear_animation_safely(self, target, frame_range=None, keep_target_paths=None, keep_data_paths=None):
        keep_target_paths = set(keep_target_paths or ())
        keep_data_paths = set(keep_data_paths or ())
        target_paths = {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"}
        data_paths = {"lens"}
        if frame_range is None:
            self.clear_animation_channels(target, keep_target_paths, target_paths)
            if getattr(target, "data", None) is not None:
                self.clear_animation_channels(target.data, keep_data_paths, data_paths)
            return

        if not keep_target_paths and not keep_data_paths:
            frame_start, frame_end = frame_range
            self.clear_keyframes_in_range(
                target,
                target_paths,
                frame_start,
                frame_end,
            )
            if getattr(target, "data", None):
                self.clear_keyframes_in_range(target.data, data_paths, frame_start, frame_end)
            return

        frame_start, frame_end = frame_range
        self.clear_keyframes_in_range(
            target,
            target_paths - keep_target_paths,
            frame_start,
            frame_end,
        )
        if getattr(target, "data", None):
            self.clear_keyframes_in_range(target.data, data_paths - keep_data_paths, frame_start, frame_end)

    def pin_lens_constant_in_range(self, cam_data, frame_start, frame_end, lens_value, source_snapshots=None):
        if cam_data is None:
            return
        anim_data = cam_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{cam_data.name}_Action")
        source_lens = [snap for snap in (source_snapshots or ()) if snap["data_path"] == "lens"]
        if source_lens:
            # Restore the original curve first so its modifiers, group and settings survive.
            self.restore_animation_curves(cam_data, source_lens)
        lens_fcurve = next(
            (fc for fc in (self._iter_action_fcurves(cam_data) or ()) if fc.data_path == "lens"),
            None,
        )
        if lens_fcurve is None:
            lens_fcurve = self._ensure_action_fcurve(cam_data, "lens")
        if lens_fcurve is None:
            return

        for key in reversed(list(lens_fcurve.keyframe_points)):
            if frame_start <= key.co.x <= frame_end:
                lens_fcurve.keyframe_points.remove(key)
        lens_fcurve.keyframe_points.add(frame_end - frame_start + 1)
        for key, frame in zip(list(lens_fcurve.keyframe_points)[-(frame_end - frame_start + 1):], range(frame_start, frame_end + 1)):
            key.co = (frame, lens_value)
            key.interpolation = 'CONSTANT'
            key.handle_left_type = 'VECTOR'
            key.handle_right_type = 'VECTOR'
        lens_fcurve.update()

        if len(lens_fcurve.modifiers):
            # F-curve modifiers act after key interpolation; compensate their effect
            # at each baked frame without deleting the user's modifier stack.
            pinned_keys = {int(key.co.x): key for key in lens_fcurve.keyframe_points if frame_start <= key.co.x <= frame_end}
            for _ in range(6):
                max_error = 0.0
                for frame, key in pinned_keys.items():
                    error = lens_value - lens_fcurve.evaluate(frame)
                    max_error = max(max_error, abs(error))
                    if abs(error) > 1e-5:
                        key.co = (frame, key.co.y + error)
                if max_error <= 1e-5:
                    break
                lens_fcurve.update()
            max_error = max(abs(lens_value - lens_fcurve.evaluate(frame)) for frame in pinned_keys)
            if max_error > 1e-4:
                self.report({'WARNING'}, "Lens F-curve modifiers prevent an exact constant lens in part of the Custom Range.")
        cam_data.lens = float(lens_value)

