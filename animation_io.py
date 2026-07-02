# pcam_solver animation I/O
from bpy_extras import anim_utils

from .common import *

class PCamAnimationIO:
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

    def clear_animation_channels(self, id_data, keep_paths=None):
        keep_paths = set(keep_paths or ())
        channelbags = self._iter_action_channelbags(id_data)
        if not channelbags:
            return
        for channelbag in channelbags:
            fcurves = getattr(channelbag, "fcurves", None)
            if fcurves is None:
                continue
            for fcurve in list(fcurves):
                if fcurve.data_path in keep_paths:
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
        return action.copy()

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
            if getattr(slots, "active", None) is not None:
                return slots.active
        except Exception:
            pass
        try:
            return slots[0] if len(slots) else None
        except Exception:
            return None

    def _iter_action_channelbags(self, id_data):
        anim_data = getattr(id_data, "animation_data", None)
        action = getattr(anim_data, "action", None)
        if action is None:
            return []

        bags = []
        get_channelbag = getattr(anim_utils, "action_get_channelbag_for_slot", None)
        ensure_channelbag = getattr(anim_utils, "action_ensure_channelbag_for_slot", None)
        slots = getattr(action, "slots", None)
        if slots is not None:
            try:
                for slot in slots:
                    channelbag = None
                    try:
                        if get_channelbag is not None:
                            channelbag = get_channelbag(action, slot)
                        elif ensure_channelbag is not None:
                            channelbag = ensure_channelbag(action, slot)
                    except Exception:
                        channelbag = None
                    if channelbag is not None and getattr(channelbag, "fcurves", None) is not None:
                        bags.append(channelbag)
            except Exception:
                pass

        if bags:
            return bags

        slot = self._get_action_slot(id_data)
        if slot is None:
            return []
        try:
            if get_channelbag is not None:
                channelbag = get_channelbag(action, slot)
            elif ensure_channelbag is not None:
                channelbag = ensure_channelbag(action, slot)
            else:
                channelbag = None
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
        try:
            channelbag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
        except Exception:
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
        # Preserve the full camera-data action when reusing an existing focal curve.
        # Lens-only restoration proved brittle in Blender when the action was recreated during solve.
        anim_data.action = action_copy

    def restore_animation_curves(self, id_data, snapshots):
        if not snapshots:
            return
        anim_data = id_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{id_data.name}_Action")
        fcurves = self._iter_action_fcurves(id_data)
        if fcurves is None:
            return
        for snap in snapshots:
            for fcurve in list(fcurves):
                if fcurve.data_path == snap["data_path"] and fcurve.array_index == snap["array_index"]:
                    fcurves.remove(fcurve)
            fcurve = self._ensure_action_fcurve(id_data, snap["data_path"], index=snap["array_index"])
            if fcurve is None:
                continue
            fcurve.extrapolation = snap["extrapolation"]
            fcurve.keyframe_points.add(len(snap["keys"]))
            for key, key_data in zip(fcurve.keyframe_points, snap["keys"]):
                key.co = key_data["co"]
                key.handle_left = key_data["handle_left"]
                key.handle_right = key_data["handle_right"]
                key.interpolation = key_data["interpolation"]
                key.handle_left_type = key_data["handle_left_type"]
                key.handle_right_type = key_data["handle_right_type"]
                if "easing" in key_data and hasattr(key, "easing"):
                    key.easing = key_data["easing"]
                if "back" in key_data and hasattr(key, "back"):
                    key.back = key_data["back"]
                if "amplitude" in key_data and hasattr(key, "amplitude"):
                    key.amplitude = key_data["amplitude"]
                if "period" in key_data and hasattr(key, "period"):
                    key.period = key_data["period"]
            fcurve.update()

    def restore_animation_snapshot_exact(self, id_data, snapshots):
        if id_data is None:
            return
        self.clear_animation_channels(id_data)
        self.restore_animation_curves(id_data, snapshots)

    def clear_animation_safely(self, target, frame_range=None, keep_target_paths=None, keep_data_paths=None):
        keep_target_paths = set(keep_target_paths or ())
        keep_data_paths = set(keep_data_paths or ())
        if frame_range is None:
            if target.animation_data and target.animation_data.action:
                fcurves = self._iter_action_fcurves(target)
                if fcurves is not None:
                    for fcurve in list(fcurves):
                        if fcurve.data_path in keep_target_paths:
                            continue
                        fcurves.remove(fcurve)
            if getattr(target, "data", None) and getattr(target.data, "animation_data", None):
                if target.data.animation_data.action:
                    fcurves = self._iter_action_fcurves(target.data)
                    if fcurves is not None:
                        for fcurve in list(fcurves):
                            if fcurve.data_path in keep_data_paths:
                                continue
                            fcurves.remove(fcurve)
            return

        if not keep_target_paths and not keep_data_paths:
            frame_start, frame_end = frame_range
            self.clear_keyframes_in_range(
                target,
                {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"},
                frame_start,
                frame_end,
            )
            if getattr(target, "data", None):
                self.clear_keyframes_in_range(target.data, {"lens"}, frame_start, frame_end)
            return

        frame_start, frame_end = frame_range
        self.clear_keyframes_in_range(
            target,
            {"location", "rotation_euler", "rotation_quaternion", "rotation_axis_angle", "scale"} - keep_target_paths,
            frame_start,
            frame_end,
        )
        if getattr(target, "data", None):
            self.clear_keyframes_in_range(target.data, {"lens"} - keep_data_paths, frame_start, frame_end)

    def pin_lens_constant_in_range(self, cam_data, frame_start, frame_end, lens_value, source_snapshots=None):
        if cam_data is None:
            return
        anim_data = cam_data.animation_data_create()
        if not anim_data.action:
            anim_data.action = bpy.data.actions.new(name=f"{cam_data.name}_Action")
        fcurves = self._iter_action_fcurves(cam_data)
        if fcurves is None:
            return

        preserved_keys = []
        if source_snapshots:
            for snap in source_snapshots:
                if snap.get("data_path") != "lens":
                    continue
                for key_data in snap.get("keys", []):
                    frame = float(key_data["co"][0])
                    if frame_start <= frame <= frame_end:
                        continue
                    preserved_keys.append({
                        "frame": frame,
                        "value": float(key_data["co"][1]),
                        "interpolation": key_data.get("interpolation", 'BEZIER'),
                        "handle_left_type": key_data.get("handle_left_type", 'AUTO'),
                        "handle_right_type": key_data.get("handle_right_type", 'AUTO'),
                    })
        for fcurve in list(fcurves):
            if fcurve.data_path == "lens":
                fcurves.remove(fcurve)

        lens_fcurve = self._ensure_action_fcurve(cam_data, "lens")
        if lens_fcurve is None:
            return
        for key in list(lens_fcurve.keyframe_points):
            lens_fcurve.keyframe_points.remove(key)

        rebuilt_keys = preserved_keys + [
            {
                "frame": float(frame),
                "value": float(lens_value),
                "interpolation": 'CONSTANT',
                "handle_left_type": 'VECTOR',
                "handle_right_type": 'VECTOR',
            }
            for frame in range(frame_start, frame_end + 1)
        ]
        rebuilt_keys.sort(key=lambda item: item["frame"])

        for key_data in rebuilt_keys:
            cam_data.lens = key_data["value"]
            lens_fcurve.keyframe_points.add(1)
            key = lens_fcurve.keyframe_points[-1]
            key.co = (key_data["frame"], key_data["value"])
            key.interpolation = key_data["interpolation"]
            key.handle_left_type = key_data["handle_left_type"]
            key.handle_right_type = key_data["handle_right_type"]
        lens_fcurve.update()
        cam_data.lens = float(lens_value)

