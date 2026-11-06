"""Calibrated planar and depth-reference camera solving without external libraries."""

from .common import *
from .animation_io import PCamAnimationIO


def _linear_solve(matrix, values):
    size = len(values)
    rows = [list(matrix[index]) + [float(values[index])] for index in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) < 1e-12:
            raise ValueError("Degenerate tracking geometry")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [item / divisor for item in rows[column]]
        for row in range(size):
            if row == column:
                continue
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
    return [row[-1] for row in rows]


def _plane_corners(marker):
    # Blender stores plane corners in BL, BR, TR, TL order.
    return [Vector(marker.corners[index]) for index in (3, 2, 1, 0)]


def _plane_marker(clip, track, scene_frame):
    return track.markers.find_frame(pcam_scene_to_clip_frame(clip, scene_frame), exact=True)


def _held_frame_ranges(frames, fitted_frames):
    held = sorted(set(frames) - fitted_frames)
    ranges = []
    for frame in held:
        if ranges and frame == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], frame)
        else:
            ranges.append((frame, frame))
    return ", ".join(str(start) if start == end else f"{start}-{end}"
                     for start, end in ranges)


def _update_existing_keys(fcurves, frame, values):
    for fcurve in fcurves or ():
        value = values.get(fcurve.data_path)
        if value is None:
            continue
        value = value if isinstance(value, (int, float)) else value[fcurve.array_index]
        for key in fcurve.keyframe_points:
            if abs(key.co.x - frame) > 1e-4:
                continue
            delta = value - key.co.y
            key.co.y = value
            key.handle_left.y += delta
            key.handle_right.y += delta
            fcurve.update()
            break


def _homography(source, target):
    rows, values = [], []
    for (x, y), (u, v) in zip(source, target):
        rows.extend(((x, y, 1, 0, 0, 0, -u * x, -u * y),
                     (0, 0, 0, x, y, 1, -v * x, -v * y)))
        values.extend((u, v))
    h = _linear_solve(rows, values)
    return Matrix(((h[0], h[1], h[2]),
                   (h[3], h[4], h[5]),
                   (h[6], h[7], 1.0)))


def _plane_layout(context, camera, corners, size, alignment, width=None, lens=None):
    use_existing_camera_pose = alignment == 'CAMERA'
    turns = []
    for index in range(4):
        first = corners[(index + 1) % 4] - corners[index]
        second = corners[(index + 2) % 4] - corners[(index + 1) % 4]
        turns.append(first.x * second.y - first.y * second.x)
    if not all(math.isfinite(value) for corner in corners for value in corner) or not (
        all(value > 1e-6 for value in turns) or all(value < -1e-6 for value in turns)
    ):
        raise ValueError("Plane Track needs four non-crossing, well-spaced corners")
    tan_x, tan_y = get_camera_tan(camera.data, lens if lens is not None else camera.data.lens, context.scene)
    normalized = []
    for corner in corners:
        ray = marker_to_camera_ray(corner, tan_x, tan_y, camera.data)
        normalized.append((ray.x / -ray.z, ray.y / -ray.z))
    source = ((-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))
    h = _homography(source, normalized)
    x_axis = h.col[0].copy()
    y_axis = h.col[1].copy()
    translation = h.col[2].copy()
    if translation.z < 0:
        x_axis.negate()
        y_axis.negate()
        translation.negate()
    if x_axis.length < 1e-9 or y_axis.length < 1e-9 or translation.z < 1e-9:
        raise ValueError("Plane Track corners do not define a usable surface")
    aspect = x_axis.length / y_axis.length
    if not 0.05 <= aspect <= 20.0:
        raise ValueError("Plane Track aspect is outside the usable range")
    axis_scale = y_axis.length
    scale = size if size is not None else width / aspect
    x_axis.normalize()
    y_axis -= x_axis * y_axis.dot(x_axis)
    if y_axis.length < 1e-9:
        raise ValueError("Plane Track is degenerate")
    y_axis.normalize()
    # Convert CV +Z-forward coordinates to Blender -Z-forward coordinates.
    plane_x = Vector((x_axis.x, x_axis.y, -x_axis.z))
    plane_y = Vector((y_axis.x, y_axis.y, -y_axis.z))
    plane_z = plane_x.cross(plane_y).normalized()
    plane_cam_rotation = Matrix(((plane_x.x, plane_y.x, plane_z.x),
                                 (plane_x.y, plane_y.y, plane_z.y),
                                 (plane_x.z, plane_y.z, plane_z.z)))
    camera_matrix = matrix_without_scale(camera.matrix_world)
    camera_rotation = camera_matrix.to_3x3()
    plane_rotation = camera_rotation @ plane_cam_rotation
    wanted_rotation = (plane_rotation if use_existing_camera_pose else
                       Matrix.Rotation(math.pi / 2, 3, 'X') if alignment == 'WALL'
                       else Matrix.Identity(3))
    aligned_rotation = wanted_rotation @ plane_cam_rotation.inverted()
    relative_center = aligned_rotation @ Vector((translation.x, translation.y, -translation.z)) * (scale / axis_scale)
    plane_location = camera_matrix.translation + relative_center if use_existing_camera_pose else Vector((0, 0, 0))
    camera_location = camera_matrix.translation if use_existing_camera_pose else -relative_center
    aligned_camera = Matrix.Translation(camera_location) @ aligned_rotation.to_4x4()
    plane_matrix = Matrix.Translation(plane_location) @ wanted_rotation.to_4x4()
    return aligned_camera, plane_matrix, width if width is not None else aspect * scale, scale


def _estimate_plane_lens(context, camera, corners, width, height):
    tan_x, tan_y = get_camera_tan(camera.data, 1.0, context.scene)
    normalized = []
    for corner in corners:
        ray = marker_to_camera_ray(corner, tan_x, tan_y, camera.data)
        normalized.append((ray.x / -ray.z, ray.y / -ray.z))
    source = ((-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5))
    homography = _homography(source, normalized)
    x_axis, y_axis = homography.col[0], homography.col[1]
    xy_dot = x_axis.x * y_axis.x + x_axis.y * y_axis.y
    xy_length_diff = ((x_axis.x ** 2 + x_axis.y ** 2) / width ** 2 -
                      (y_axis.x ** 2 + y_axis.y ** 2) / height ** 2)
    depth_dot = x_axis.z * y_axis.z
    depth_length_diff = x_axis.z ** 2 / width ** 2 - y_axis.z ** 2 / height ** 2
    candidates = []
    for horizontal, depth in ((xy_dot, depth_dot), (xy_length_diff, depth_length_diff)):
        if abs(depth) > 1e-12 and horizontal * depth < 0:
            lens_sq = -horizontal / depth
            if math.isfinite(lens_sq) and 1.0 <= lens_sq <= 5000.0 ** 2:
                candidates.append(math.sqrt(lens_sq))
    if not candidates:
        raise ValueError("Auto Lens needs a tilted plane with known width and height")
    if len(candidates) == 2 and max(candidates) / min(candidates) > 1.3:
        raise ValueError("Auto Lens estimates disagree; check the corners and plane dimensions")
    lens = sum(candidates) / len(candidates)
    x_direction = Vector((x_axis.x / lens, x_axis.y / lens, x_axis.z)) / width
    y_direction = Vector((y_axis.x / lens, y_axis.y / lens, y_axis.z)) / height
    tilt = max(abs(x_direction.z) / x_direction.length,
               abs(y_direction.z) / y_direction.length)
    orthogonality = abs(x_direction.dot(y_direction)) / (x_direction.length * y_direction.length)
    scale_mismatch = abs(x_direction.length - y_direction.length) / max(x_direction.length, y_direction.length)
    if tilt < 0.05 or orthogonality > 0.05 or scale_mismatch > 0.05:
        raise ValueError("Auto Lens is unreliable for these corners; adjust the track or set the lens manually")
    return lens


def _store_plane_alignment(plane, camera, frame):
    plane["pcam_align_frame"] = frame
    plane["pcam_align_camera"] = camera.name
    plane["pcam_align_plane_matrix"] = [value for row in plane.matrix_world for value in row]
    plane["pcam_align_camera_matrix"] = [value for row in matrix_without_scale(camera.matrix_world) for value in row]
    plane["pcam_align_vertices"] = [value for vertex in plane.data.vertices[:4] for value in vertex.co]


def _is_unedited_plane_alignment(plane, camera, frame):
    if plane.get("pcam_align_frame") != frame or plane.get("pcam_align_camera") != camera.name:
        return False
    values = (
        (plane.get("pcam_align_plane_matrix"), [value for row in plane.matrix_world for value in row]),
        (plane.get("pcam_align_camera_matrix"),
         [value for row in matrix_without_scale(camera.matrix_world) for value in row]),
        (plane.get("pcam_align_vertices"),
         [value for vertex in plane.data.vertices[:4] for value in vertex.co]),
    )
    return all(stored is not None and len(stored) == len(current) and
               max(abs(a - b) for a, b in zip(stored, current)) < 1e-5
               for stored, current in values)


def _project(context, camera_data, location, rotation, world_point, lens):
    local = rotation.inverted() @ (world_point - location)
    if local.z >= -1e-7:
        return None
    tan_x, tan_y = get_camera_tan(camera_data, lens, context.scene)
    shift_x, shift_y = camera_data.shift_x, camera_data.shift_y
    fit = camera_data.sensor_fit
    shift_tan = tan_y if fit == 'VERTICAL' else tan_x if fit == 'HORIZONTAL' else max(tan_x, tan_y)
    return Vector((
        0.5 + (local.x / -local.z - 2 * shift_x * shift_tan) / (2 * tan_x),
        0.5 + (local.y / -local.z - 2 * shift_y * shift_tan) / (2 * tan_y),
    ))


def _depth_hit(context, camera, matrix, depth, co, lens):
    hit = raycast_marker_world_from_matrix(context, camera.data, matrix, depth, co, lens)
    if hit is not None or depth.type != 'MESH' or len(depth.data.vertices) != 4:
        return hit
    local_corners = [vertex.co for vertex in depth.data.vertices]
    if max(abs(corner.z - local_corners[0].z) for corner in local_corners) > 1e-5:
        return None
    tan_x, tan_y = get_camera_tan(camera.data, lens, context.scene)
    ray = marker_to_camera_ray(co, tan_x, tan_y, camera.data)
    inverse = depth.matrix_world.inverted()
    origin = inverse @ matrix.translation
    direction = inverse.to_3x3() @ (matrix.to_3x3() @ ray)
    if abs(direction.z) < 1e-10:
        return None
    distance = (local_corners[0].z - origin.z) / direction.z
    if distance <= 0:
        return None
    point = origin + direction * distance
    for axis in (0, 1):
        values = [corner[axis] for corner in local_corners]
        margin = (max(values) - min(values)) * 0.01
        if not min(values) - margin <= point[axis] <= max(values) + margin:
            return None
    return depth.matrix_world @ point


def _residuals(context, camera_data, observations, location, rotation, lens, width, height):
    result = []
    for point, observed in observations:
        projected = _project(context, camera_data, location, rotation, point, lens)
        if projected is None:
            return None
        result.extend(((projected.x - observed.x) * width,
                       (projected.y - observed.y) * height))
    return result


def _fit_plane_pose(context, camera, local_corners, observed_corners, initial_pose, lens=None):
    scene = context.scene
    camera_matrix = matrix_without_scale(evaluated_matrix_world(context, camera))
    camera_location, camera_rotation = camera_matrix.translation, camera_matrix.to_quaternion()
    lens = camera.data.lens if lens is None else lens
    width, height = scene.render.resolution_x, scene.render.resolution_y
    location, rotation = initial_pose.translation.copy(), initial_pose.to_quaternion()

    def residuals(point_location, point_rotation):
        transform = Matrix.Translation(point_location) @ point_rotation.to_matrix().to_4x4()
        result = []
        for corner, observed in zip(local_corners, observed_corners):
            projected = _project(context, camera.data, camera_location, camera_rotation,
                                 transform @ corner, lens)
            if projected is None:
                return None
            result.extend(((projected.x - observed.x) * width,
                           (projected.y - observed.y) * height))
        return result

    residual = residuals(location, rotation)
    if residual is None:
        return initial_pose
    cost = sum(value * value for value in residual)
    damping = 1e-3
    for _ in range(25):
        columns = []
        for index in range(6):
            step = max(1e-4, (location - camera_location).length * 1e-5) if index < 3 else 1e-4
            trial_location, trial_rotation = location.copy(), rotation.copy()
            axis = Vector((0, 0, 0))
            if index < 3:
                trial_location[index] += step
            else:
                axis[index - 3] = 1
                trial_rotation = Quaternion(axis, step) @ trial_rotation
            trial = residuals(trial_location, trial_rotation)
            if trial is None:
                return Matrix.Translation(location) @ rotation.to_matrix().to_4x4()
            columns.append([(a - b) / step for a, b in zip(trial, residual)])
        normal = [[sum(a * b for a, b in zip(columns[i], columns[j]))
                   for j in range(6)] for i in range(6)]
        gradient = [-sum(a * b for a, b in zip(column, residual)) for column in columns]
        for index in range(6):
            normal[index][index] += damping * max(1.0, normal[index][index])
        try:
            delta = _linear_solve(normal, gradient)
        except ValueError:
            break
        trial_location, trial_rotation = location.copy(), rotation.copy()
        turn = Vector((delta[3], delta[4], delta[5]))
        trial_location += Vector(delta[:3])
        if turn.length > 0.35:
            turn *= 0.35 / turn.length
        if turn.length > 1e-12:
            trial_rotation = Quaternion(turn.normalized(), turn.length) @ trial_rotation
        trial = residuals(trial_location, trial_rotation)
        trial_cost = sum(value * value for value in trial) if trial is not None else float('inf')
        if trial_cost < cost:
            location, rotation, residual, cost = trial_location, trial_rotation, trial, trial_cost
            damping = max(1e-9, damping * 0.3)
            if cost < 1e-8:
                break
        else:
            damping = min(1e9, damping * 10)
            if damping >= 1e9:
                break
    return Matrix.Translation(location) @ rotation.to_matrix().to_4x4()


def _fit_camera(context, camera_data, observations, location, rotation, lens,
                fixed_location=False, fixed_height=None):
    width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
    indices = [] if fixed_location else [0, 1, 2]
    if fixed_height is not None and 2 in indices:
        indices.remove(2)
    indices += [3, 4, 5]
    location, rotation = location.copy(), rotation.copy()
    if fixed_height is not None and not fixed_location:
        location.z = fixed_height
    residual = _residuals(context, camera_data, observations, location, rotation, lens, width, height)
    if residual is None:
        return None
    cost = sum(value * value for value in residual)
    damping = 1e-3
    for _ in range(30):
        columns = []
        for index in indices:
            step = max(1e-5, location.length * 1e-5) if index < 3 else 1e-5
            trial_loc, trial_rot = location.copy(), rotation.copy()
            if index < 3:
                trial_loc[index] += step
            else:
                axis = Vector((0, 0, 0))
                axis[index - 3] = 1
                trial_rot = Quaternion(axis, step) @ trial_rot
            trial = _residuals(context, camera_data, observations, trial_loc, trial_rot, lens, width, height)
            if trial is None:
                return None
            columns.append([(a - b) / step for a, b in zip(trial, residual)])
        normal = [[sum(a * b for a, b in zip(columns[i], columns[j]))
                   for j in range(len(indices))] for i in range(len(indices))]
        gradient = [-sum(a * b for a, b in zip(column, residual)) for column in columns]
        for index in range(len(indices)):
            normal[index][index] += damping * max(1.0, normal[index][index])
        try:
            delta = _linear_solve(normal, gradient)
        except ValueError:
            break
        trial_loc, trial_rot = location.copy(), rotation.copy()
        turn = Vector((0, 0, 0))
        for index, value in zip(indices, delta):
            if index < 3:
                trial_loc[index] += value
            else:
                turn[index - 3] = value
        move = trial_loc - location
        depth_scale = sum((point - location).length for point, _co in observations) / len(observations)
        move_limit = max(0.25, depth_scale * 0.25)
        if move.length > move_limit:
            trial_loc = location + move * (move_limit / move.length)
        if turn.length > 0.35:
            turn *= 0.35 / turn.length
        if turn.length > 1e-12:
            trial_rot = Quaternion(turn.normalized(), turn.length) @ trial_rot
        trial = _residuals(context, camera_data, observations, trial_loc, trial_rot, lens, width, height)
        trial_cost = sum(value * value for value in trial) if trial is not None else float('inf')
        if trial_cost < cost:
            location, rotation, residual, cost = trial_loc, trial_rot, trial, trial_cost
            damping = max(1e-9, damping * 0.3)
            if cost < 1e-8:
                break
        else:
            damping = min(1e9, damping * 10)
            if damping >= 1e9:
                break
    return location, rotation, math.sqrt(cost / len(observations))


def _fit_camera_rotation_lens(context, camera_data, observations, location, rotation, lens):
    width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
    rotation, lens = rotation.copy(), float(lens)
    residual = _residuals(context, camera_data, observations, location, rotation, lens, width, height)
    if residual is None:
        return None
    cost = sum(value * value for value in residual)
    damping = 1e-3
    for _ in range(35):
        columns = []
        for index in range(4):
            trial_rotation, trial_lens = rotation.copy(), lens
            if index < 3:
                axis = Vector((0, 0, 0))
                axis[index] = 1
                trial_rotation = Quaternion(axis, 1e-5) @ trial_rotation
            else:
                trial_lens *= math.exp(1e-4)
            trial = _residuals(context, camera_data, observations,
                               location, trial_rotation, trial_lens, width, height)
            if trial is None:
                return None
            step = 1e-5 if index < 3 else 1e-4
            columns.append([(a - b) / step for a, b in zip(trial, residual)])
        normal = [[sum(a * b for a, b in zip(columns[i], columns[j]))
                   for j in range(4)] for i in range(4)]
        gradient = [-sum(a * b for a, b in zip(column, residual)) for column in columns]
        for index in range(4):
            normal[index][index] += damping * max(1.0, normal[index][index])
        try:
            delta = _linear_solve(normal, gradient)
        except ValueError:
            break
        turn = Vector(delta[:3])
        if turn.length > 0.35:
            turn *= 0.35 / turn.length
        trial_rotation = (Quaternion(turn.normalized(), turn.length) @ rotation
                          if turn.length > 1e-12 else rotation.copy())
        trial_lens = lens * math.exp(max(-0.25, min(0.25, delta[3])))
        trial_lens = max(1.0, min(5000.0, trial_lens))
        trial = _residuals(context, camera_data, observations,
                           location, trial_rotation, trial_lens, width, height)
        trial_cost = sum(value * value for value in trial) if trial is not None else float('inf')
        if trial_cost < cost:
            rotation, lens, residual, cost = trial_rotation, trial_lens, trial, trial_cost
            damping = max(1e-9, damping * 0.3)
            if cost < 1e-8:
                break
        else:
            damping = min(1e9, damping * 10)
            if damping >= 1e9:
                break
    return rotation, lens, math.sqrt(cost / len(observations))


def _fit_camera_planar_lens(context, camera_data, observations, location, rotation, lens,
                            plane_axes):
    width, height = context.scene.render.resolution_x, context.scene.render.resolution_y
    location, rotation, lens = location.copy(), rotation.copy(), float(lens)
    residual = _residuals(context, camera_data, observations, location, rotation, lens, width, height)
    if residual is None:
        return None
    cost = sum(value * value for value in residual)
    damping = 1e-3
    for _ in range(35):
        columns = []
        for index in range(6):
            trial_location, trial_rotation, trial_lens = location.copy(), rotation.copy(), lens
            if index < 2:
                step = max(1e-4, (observations[0][0] - location).length * 1e-5)
                trial_location += plane_axes[index] * step
            elif index < 5:
                step = 1e-5
                axis = Vector((0, 0, 0))
                axis[index - 2] = 1
                trial_rotation = Quaternion(axis, step) @ rotation
            else:
                step = 1e-4
                trial_lens *= math.exp(step)
            trial = _residuals(context, camera_data, observations,
                               trial_location, trial_rotation, trial_lens, width, height)
            if trial is None:
                return None
            columns.append([(a - b) / step for a, b in zip(trial, residual)])
        normal = [[sum(a * b for a, b in zip(columns[i], columns[j]))
                   for j in range(6)] for i in range(6)]
        gradient = [-sum(a * b for a, b in zip(column, residual)) for column in columns]
        for index in range(6):
            normal[index][index] += damping * max(1.0, normal[index][index])
        try:
            delta = _linear_solve(normal, gradient)
        except ValueError:
            break
        move = plane_axes[0] * delta[0] + plane_axes[1] * delta[1]
        move_limit = max(0.25, (observations[0][0] - location).length * 0.25)
        if move.length > move_limit:
            move *= move_limit / move.length
        trial_location = location + move
        turn = Vector(delta[2:5])
        if turn.length > 0.35:
            turn *= 0.35 / turn.length
        trial_rotation = (Quaternion(turn.normalized(), turn.length) @ rotation
                          if turn.length > 1e-12 else rotation.copy())
        trial_lens = max(1.0, min(5000.0, lens * math.exp(max(-0.25, min(0.25, delta[5])))))
        trial = _residuals(context, camera_data, observations,
                           trial_location, trial_rotation, trial_lens, width, height)
        trial_cost = sum(value * value for value in trial) if trial is not None else float('inf')
        if trial_cost < cost:
            location, rotation, lens, residual, cost = (
                trial_location, trial_rotation, trial_lens, trial, trial_cost)
            damping = max(1e-9, damping * 0.3)
            if cost < 1e-8:
                break
        else:
            damping = min(1e9, damping * 10)
            if damping >= 1e9:
                break
    return location, rotation, lens, math.sqrt(cost / len(observations))


class PCAM_OT_align_plane_track(PCamAnimationIO, bpy.types.Operator):
    bl_idname = "view3d.pcam_align_plane_track"
    bl_label = "Align Camera and Plane"
    bl_description = "Align the camera and Depth Reference to Plane Track corners on the current frame"
    bl_options = {'REGISTER', 'UNDO'}

    size: bpy.props.FloatProperty(
        name="Plane Height", description="Plane height in scene units",
        default=2.0, min=0.01, unit='LENGTH')
    width: bpy.props.FloatProperty(
        name="Plane Width", description="Plane width in scene units",
        default=2.0, min=0.01, unit='LENGTH')
    size_mode: bpy.props.EnumProperty(
        name="Size Mode",
        items=[('HEIGHT', "Height", "Specify height and infer width"),
               ('WIDTH', "Width", "Specify width and infer height"),
               ('DIMENSIONS', "Dimensions", "Specify both dimensions and refit to the corners")],
        default='HEIGHT',
    )
    alignment: bpy.props.EnumProperty(
        name="Alignment",
        items=[('WALL', "Wall", "Place a vertical plane at the origin and align the camera"),
               ('FLOOR', "Floor", "Place a horizontal plane at the origin and align the camera"),
               ('CAMERA', "Camera", "Keep the camera pose and align the plane")],
        default='WALL',
    )
    auto_lens: bpy.props.BoolProperty(
        name="Auto Lens",
        description="Estimate focal length from the known width and height before aligning",
        default=False,
    )

    def execute(self, context):
        props = context.scene.pcam_solve_props
        camera, clip = context.scene.camera, props.target_clip
        if not camera or not clip:
            self.report({'ERROR'}, "Camera and Movie Clip are required.")
            return {'CANCELLED'}
        try:
            track = clip.tracking.objects[int(props.tracking_object_idx)].plane_tracks.get(props.plane_track_name)
            if track is None:
                raise ValueError("Select a Plane Track")
            frame = context.scene.frame_current
            marker = _plane_marker(clip, track, frame)
            if marker is None:
                raise ValueError("Plane Track has no corners on the current frame")
            context.scene.frame_set(frame)
            alignment = self.alignment if props.apply_to == 'CAMERA' else 'CAMERA'
            corners = _plane_corners(marker)
            if self.auto_lens and (props.apply_to != 'CAMERA' or self.size_mode != 'DIMENSIONS'):
                raise ValueError("Auto Lens requires Camera mode and Dimensions")
            estimated_lens = (_estimate_plane_lens(context, camera, corners, self.width, self.size)
                              if self.auto_lens else None)
            camera_matrix, plane_matrix, width, height = _plane_layout(
                context, camera, corners,
                self.size if self.size_mode != 'WIDTH' else None,
                alignment, self.width if self.size_mode != 'HEIGHT' else None,
                lens=estimated_lens)
            local_corners = [Vector(co) for co in
                ((-width / 2, height / 2, 0), (width / 2, height / 2, 0),
                 (width / 2, -height / 2, 0), (-width / 2, -height / 2, 0))]
            if props.apply_to == 'OBJECT' or (alignment == 'CAMERA' and self.size_mode == 'DIMENSIONS'):
                plane_matrix = _fit_plane_pose(context, camera, local_corners, corners, plane_matrix,
                                               lens=estimated_lens)
            elif self.size_mode == 'DIMENSIONS':
                observations = [(plane_matrix @ co, observed) for co, observed in zip(local_corners, corners)]
                fitted = _fit_camera(context, camera.data, observations,
                                     camera_matrix.translation, camera_matrix.to_quaternion(),
                                     estimated_lens if estimated_lens is not None else camera.data.lens)
                if fitted is None:
                    raise ValueError("Could not align the specified plane dimensions")
                camera_matrix = Matrix.Translation(fitted[0]) @ fitted[1].to_matrix().to_4x4()
        except (IndexError, ValueError) as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        plane = props.clip_depth_object
        reuse = (plane is not None and plane.name.startswith("PCam_Plane") and
                 plane.type == 'MESH' and len(plane.data.vertices) == 4)
        if reuse:
            if plane.data.users > 1:
                plane.data = plane.data.copy()
            for vertex, co in zip(plane.data.vertices, local_corners):
                vertex.co = co
            plane.data.update()
        else:
            mesh = bpy.data.meshes.new("PCam_Plane_Mesh")
            mesh.from_pydata(local_corners, [], [(3, 2, 1, 0)])
            mesh.update()
            plane = bpy.data.objects.new("PCam_Plane", mesh)
            pcam_camera_collection(context, camera).objects.link(plane)
        if props.apply_to == 'CAMERA':
            camera.matrix_world = camera_matrix
            if estimated_lens is not None:
                camera.data.lens = estimated_lens
                _update_existing_keys(self._iter_action_fcurves(camera.data), frame,
                                      {'lens': estimated_lens})
            rotation_path = ('rotation_quaternion' if camera.rotation_mode == 'QUATERNION' else
                             'rotation_axis_angle' if camera.rotation_mode == 'AXIS_ANGLE' else 'rotation_euler')
            values = {rotation_path: getattr(camera, rotation_path)}
            if alignment != 'CAMERA':
                values['location'] = camera.location
            _update_existing_keys(self._iter_action_fcurves(camera), frame, values)
        plane.matrix_world = plane_matrix
        rotation_path = ('rotation_quaternion' if plane.rotation_mode == 'QUATERNION' else
                         'rotation_axis_angle' if plane.rotation_mode == 'AXIS_ANGLE' else 'rotation_euler')
        _update_existing_keys(self._iter_action_fcurves(plane), frame,
                              {'location': plane.location, rotation_path: getattr(plane, rotation_path),
                               'scale': plane.scale})
        plane.display_type = 'WIRE'
        plane.show_in_front = True
        props.clip_depth_object = plane
        assigned_target = props.apply_to == 'OBJECT' and props.target_object is None
        if assigned_target:
            props.target_object = plane
        if props.use_reference_frame_lock:
            props.reference_frame = frame
        _store_plane_alignment(plane, camera, frame)
        subject = "camera and plane" if props.apply_to == 'CAMERA' else "plane"
        lens_note = f" Auto Lens: {estimated_lens:.2f} mm." if estimated_lens is not None else ""
        target_note = " Set as Target Object." if assigned_target else ""
        self.report({'INFO'}, f"{'Updated' if reuse else 'Created'} '{plane.name}' and aligned {subject} on frame {frame}.{lens_note}{target_note}")
        return {'FINISHED'}


class PCamPlaneSolver:
    def execute_plane_track_focal(self, context, target):
        props = context.scene.pcam_solve_props
        scene, clip, depth = context.scene, props.target_clip, props.clip_depth_object
        track = clip.tracking.objects[int(props.tracking_object_idx)].plane_tracks.get(props.plane_track_name)
        start, end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, start, end)
        if depth.type != 'MESH' or len(depth.data.vertices) < 4:
            self.report({'ERROR'}, "Depth Reference must be a four-corner plane.")
            return {'CANCELLED'}
        scene.frame_set(ref_f)
        depth_matrix = evaluated_matrix_world(context, depth)
        world_points = [depth_matrix @ depth.data.vertices[index].co for index in range(4)]
        solve_reference = not _is_unedited_plane_alignment(depth, target, ref_f)
        ref_location = target.matrix_world.translation.copy()
        ref_rotation = target.matrix_world.to_quaternion()
        ref_lens = target.data.lens
        use_existing = pcam_use_existing_position(props)
        solve_planar_motion = not props.plane_tripod and not use_existing
        plane_rotation = matrix_without_scale(depth_matrix).to_quaternion()
        plane_axes = (plane_rotation @ Vector((1, 0, 0)),
                      plane_rotation @ Vector((0, 1, 0)))
        locations, observations = {}, {}
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            locations[frame] = (target.matrix_world.translation.copy() if use_existing
                                else ref_location.copy())
            marker = _plane_marker(clip, track, frame)
            if marker is not None:
                observations[frame] = list(zip(world_points, _plane_corners(marker)))
        if ref_f not in observations:
            self.report({'ERROR'}, "Reference Frame needs a valid Plane Track marker.")
            return {'CANCELLED'}
        if solve_reference:
            scene.frame_set(ref_f)
            fitted = _fit_camera(context, target.data, observations[ref_f], ref_location,
                                 ref_rotation, ref_lens, fixed_location=True)
            if fitted is None:
                self.report({'ERROR'}, "Could not fit the adjusted plane on the Reference Frame.")
                return {'CANCELLED'}
            ref_rotation = fitted[1]

        rotations, lenses = {ref_f: ref_rotation}, {ref_f: ref_lens}
        fitted_frames = {ref_f}
        for direction in (range(ref_f + 1, end + 1), range(ref_f - 1, start - 1, -1)):
            previous = ref_f
            for frame in direction:
                scene.frame_set(frame)
                pairs = observations.get(frame)
                if solve_planar_motion:
                    result = (_fit_camera_planar_lens(context, target.data, pairs,
                              locations[previous], rotations[previous], lenses[previous], plane_axes)
                              if pairs else None)
                    valid = result is not None and result[3] <= 12.0
                else:
                    result = (_fit_camera_rotation_lens(context, target.data, pairs,
                              locations[frame], rotations[previous], lenses[previous])
                              if pairs else None)
                    valid = result is not None and result[2] <= 12.0
                if valid:
                    if solve_planar_motion:
                        locations[frame], rotations[frame], lenses[frame], _error = result
                    else:
                        rotations[frame], lenses[frame], _error = result
                    fitted_frames.add(frame)
                else:
                    rotations[frame], lenses[frame] = rotations[previous].copy(), lenses[previous]
                    if solve_planar_motion:
                        locations[frame] = locations[previous].copy()
                previous = frame
        if len(fitted_frames) == 1 and end > start:
            self.report({'ERROR'}, "No usable Plane Track focal frames were fitted in the bake range.")
            return {'CANCELLED'}

        frames = list(range(start, end + 1))
        if solve_planar_motion and props.clip_position_smooth > 1e-4:
            locations = smooth_vector_curve_global(locations, frames,
                strength=props.clip_position_smooth, passes=1 + int(3 * props.clip_position_smooth))
            locations[ref_f] = ref_location
            for frame in frames:
                if frame == ref_f or frame not in observations:
                    continue
                scene.frame_set(frame)
                fitted = _fit_camera_rotation_lens(context, target.data, observations[frame],
                                                   locations[frame], rotations[frame], lenses[frame])
                if fitted is not None:
                    rotations[frame], lenses[frame] = fitted[:2]
        if props.clip_focal_smooth > 1e-4:
            lenses = smooth_scalar_curve_global(lenses, frames,
                strength=props.clip_focal_smooth, passes=1 + int(3 * props.clip_focal_smooth))
            lenses[ref_f] = ref_lens
            for frame in frames:
                if frame == ref_f or frame not in observations:
                    continue
                scene.frame_set(frame)
                fitted = _fit_camera(context, target.data, observations[frame],
                                     locations[frame], rotations[frame], lenses[frame],
                                     fixed_location=True)
                if fitted is not None:
                    rotations[frame] = fitted[1]
        if props.solve_3d_rotation_smooth > 1e-4:
            rotations = smooth_quaternion_curve_global(rotations, frames,
                strength=props.solve_3d_rotation_smooth,
                passes=1 + int(3 * props.solve_3d_rotation_smooth))
            rotations[ref_f] = ref_rotation

        target_snapshot = self.snapshot_animation_action(target)
        lens_snapshot = self.snapshot_animation_action(target.data)
        try:
            self.clear_animation_safely(target,
                (start, end) if props.use_custom_range else None,
                keep_target_paths={"location"} if use_existing else None)
            for frame in frames:
                scene.frame_set(frame)
                if not use_existing:
                    target.location = locations[frame]
                    target.keyframe_insert(data_path="location", frame=frame)
                self.set_target_rotation(target, rotations[frame])
                self.keyframe_target_rotation(target, frame)
                target.data.lens = lenses[frame]
                target.data.keyframe_insert(data_path="lens", frame=frame)
        except Exception:
            self.restore_animation_snapshot_exact(target, target_snapshot)
            self.restore_animation_snapshot_exact(target.data, lens_snapshot)
            raise
        scene.frame_set(ref_f)
        self.report({'INFO'}, f"Baked Plane Track focal motion to '{target.name}': {len(fitted_frames)}/{len(frames)} frames fitted.")
        return {'FINISHED'}

    def execute_plane_track_object(self, context, target):
        props = context.scene.pcam_solve_props
        scene, camera, clip, depth = context.scene, context.scene.camera, props.target_clip, props.clip_depth_object
        track = clip.tracking.objects[int(props.tracking_object_idx)].plane_tracks.get(props.plane_track_name)
        start, end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, start, end)
        if depth.type != 'MESH' or len(depth.data.vertices) != 4:
            self.report({'ERROR'}, "Depth Reference must be a four-corner plane.")
            return {'CANCELLED'}
        scene.frame_set(ref_f)
        marker = _plane_marker(clip, track, ref_f)
        if marker is None:
            self.report({'ERROR'}, "Reference Frame needs a valid Plane Track marker.")
            return {'CANCELLED'}
        guide_matrix = matrix_without_scale(evaluated_matrix_world(context, depth))
        guide_height = (depth.matrix_world @ depth.data.vertices[0].co -
                        depth.matrix_world @ depth.data.vertices[3].co).length
        if guide_height < 1e-6:
            self.report({'ERROR'}, "Depth Reference has no usable height.")
            return {'CANCELLED'}
        target_reference = target.matrix_world.copy()
        frame_poses = {}
        frame_corners = {}
        local_corners = [vertex.co.copy() for vertex in depth.data.vertices[:4]]
        guide_scale = depth.matrix_world.to_scale()
        local_corners = [Vector((point.x * guide_scale.x, point.y * guide_scale.y,
                                 point.z * guide_scale.z)) for point in local_corners]
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            marker = _plane_marker(clip, track, frame)
            if marker is None:
                continue
            try:
                layout_camera, layout_plane, _width, _height = _plane_layout(
                    context, camera, _plane_corners(marker), guide_height, 'WALL')
            except ValueError:
                continue
            camera_matrix = matrix_without_scale(evaluated_matrix_world(context, camera))
            initial_pose = camera_matrix @ layout_camera.inverted() @ layout_plane
            frame_corners[frame] = _plane_corners(marker)
            frame_poses[frame] = _fit_plane_pose(context, camera, local_corners,
                                                frame_corners[frame], initial_pose)
        if ref_f not in frame_poses or len(frame_poses) < 2:
            self.report({'ERROR'}, "Plane Track needs valid corners beyond the Reference Frame.")
            return {'CANCELLED'}
        reference_inverse = frame_poses[ref_f].inverted()
        guide_inverse = guide_matrix.inverted()
        matrices = {}
        for frame in range(start, end + 1):
            nearest = min(frame_poses, key=lambda known: (abs(known - frame), known))
            pose = frame_poses.get(frame, frame_poses[nearest])
            guide_pose = guide_matrix @ reference_inverse @ pose
            if frame != ref_f and frame in frame_corners:
                scene.frame_set(frame)
                guide_pose = _fit_plane_pose(context, camera, local_corners,
                                             frame_corners[frame], guide_pose)
            matrices[frame] = guide_pose @ guide_inverse @ target_reference
        frames = list(range(start, end + 1))
        positions = {frame: matrices[frame].translation.copy() for frame in frames}
        rotations = {frame: matrices[frame].to_quaternion() for frame in frames}
        if props.clip_position_smooth > 1e-4:
            positions = smooth_vector_curve_global(positions, frames,
                strength=props.clip_position_smooth, passes=1 + int(3 * props.clip_position_smooth))
            positions[ref_f] = target_reference.translation.copy()
        if props.solve_3d_rotation_smooth > 1e-4:
            rotations = smooth_quaternion_curve_global(rotations, frames,
                strength=props.solve_3d_rotation_smooth,
                passes=1 + int(3 * props.solve_3d_rotation_smooth))
            rotations[ref_f] = target_reference.to_quaternion()
        snapshot = self.snapshot_animation_action(target)
        try:
            self.clear_animation_safely(target,
                (start, end) if props.use_custom_range else None)
            for frame in frames:
                scene.frame_set(frame)
                location, _rotation, scale = matrices[frame].decompose()
                target.matrix_world = (Matrix.Translation(positions[frame]) @
                    rotations[frame].to_matrix().to_4x4() @
                    Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0)))
                target.keyframe_insert(data_path="location", frame=frame)
                self.keyframe_target_rotation(target, frame)
                target.keyframe_insert(data_path="scale", frame=frame)
        except Exception:
            self.restore_animation_snapshot_exact(target, snapshot)
            raise
        scene.frame_set(ref_f)
        self.report({'INFO'}, f"Baked Plane Track to '{target.name}': {len(frame_poses)}/{len(frames)} frames tracked.")
        return {'FINISHED'}

    def _bake_3d_observations(self, context, target, ref_f, frame_start, frame_end,
                              observations, lens_values, fixed_position=False,
                              dynamic_markers=None, depth=None, reference_anchors=None,
                              solve_reference=False):
        props = context.scene.pcam_solve_props
        scene = context.scene
        scene.frame_set(ref_f)
        ref_location = target.matrix_world.translation.copy()
        ref_rotation = target.matrix_world.to_quaternion()
        existing = {}
        if fixed_position:
            for frame in range(frame_start, frame_end + 1):
                scene.frame_set(frame)
                existing[frame] = target.matrix_world.translation.copy()
        if solve_reference:
            result = _fit_camera(context, target.data, observations[ref_f], ref_location,
                ref_rotation, lens_values[ref_f], fixed_location=fixed_position,
                fixed_height=ref_location.z if props.lock_camera_z and not fixed_position else None)
            if result is None:
                self.report({'ERROR'}, "Could not fit the adjusted plane on the Reference Frame.")
                return {'CANCELLED'}
            ref_location, ref_rotation = result[:2]
        positions = {ref_f: ref_location}
        rotations = {ref_f: ref_rotation}
        fitted_frames = {ref_f}
        for direction in (range(ref_f + 1, frame_end + 1), range(ref_f - 1, frame_start - 1, -1)):
            previous = ref_f
            anchors = dict(reference_anchors or ())
            for frame in direction:
                current = dynamic_markers.get(frame, {}) if dynamic_markers else {}
                pairs = ([(anchors[name], co) for name, co in current.items() if name in anchors]
                         if dynamic_markers else observations.get(frame, []))
                observations[frame] = pairs
                seed_loc = existing[frame] if fixed_position else positions[previous]
                if len(pairs) >= 4:
                    result = _fit_camera(context, target.data, pairs, seed_loc,
                                         rotations[previous], lens_values[frame],
                                         fixed_location=fixed_position,
                                         fixed_height=ref_location.z if props.lock_camera_z and not fixed_position else None)
                else:
                    result = None
                if result is not None and result[2] > 12.0:
                    result = None
                if result is None:
                    positions[frame] = seed_loc.copy()
                    rotations[frame] = rotations[previous].copy()
                else:
                    positions[frame], rotations[frame], _error = result
                    fitted_frames.add(frame)
                if dynamic_markers:
                    scene.frame_set(frame)
                    camera_matrix = Matrix.Translation(positions[frame]) @ rotations[frame].to_matrix().to_4x4()
                    for name, co in current.items():
                        if name not in anchors:
                            hit = _depth_hit(context, target, camera_matrix, depth, co, lens_values[frame])
                            if hit is not None:
                                anchors[name] = hit
                previous = frame
        frames = list(range(frame_start, frame_end + 1))
        if props.clip_position_smooth > 1e-4 and not fixed_position:
            positions = smooth_vector_curve_global(positions, frames,
                strength=props.clip_position_smooth, passes=1 + int(3 * props.clip_position_smooth))
            positions[ref_f] = ref_location
            for frame in frames:
                pairs = observations.get(frame, [])
                if frame != ref_f and len(pairs) >= 4:
                    fitted = _fit_camera(context, target.data, pairs, positions[frame],
                        rotations[frame], lens_values[frame], fixed_location=True)
                    if fitted is not None:
                        rotations[frame] = fitted[1]
        if props.solve_3d_rotation_smooth > 1e-4:
            rotations = smooth_quaternion_curve_global(rotations, frames,
                strength=props.solve_3d_rotation_smooth,
                passes=1 + int(3 * props.solve_3d_rotation_smooth))
            rotations[ref_f] = ref_rotation
        if len(fitted_frames) == 1 and len(frames) > 1:
            self.report({'ERROR'}, "No usable 3D camera poses were fitted in the bake range.")
            return {'CANCELLED'}
        snapshot = self.snapshot_animation_action(target)
        try:
            self.clear_animation_safely(target,
                (frame_start, frame_end) if props.use_custom_range else None,
                keep_target_paths={"location"} if fixed_position else None,
                keep_data_paths={"lens"})
            for frame in frames:
                scene.frame_set(frame)
                if not fixed_position:
                    target.location = positions[frame]
                    target.keyframe_insert(data_path="location", frame=frame)
                self.set_target_rotation(target, rotations[frame])
                self.keyframe_target_rotation(target, frame)
        except Exception:
            self.restore_animation_snapshot_exact(target, snapshot)
            raise
        scene.frame_set(ref_f)
        self.report({'INFO'}, f"Baked 3D motion to '{target.name}': {len(fitted_frames)}/{len(frames)} frames fitted.")
        held_ranges = _held_frame_ranges(frames, fitted_frames)
        if held_ranges:
            self.report({'WARNING'}, f"3D Solve held the previous pose on frames: {held_ranges}.")
        return {'FINISHED'}

    def execute_plane_track(self, context, target):
        props = context.scene.pcam_solve_props
        if props.apply_to == 'OBJECT':
            return self.execute_plane_track_object(context, target)
        if props.plane_solve_mode == 'FOCAL_LENGTH':
            return self.execute_plane_track_focal(context, target)
        clip = props.target_clip
        depth = props.clip_depth_object
        track = clip.tracking.objects[int(props.tracking_object_idx)].plane_tracks.get(props.plane_track_name)
        start, end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, start, end)
        if depth.type != 'MESH' or len(depth.data.vertices) < 4:
            self.report({'ERROR'}, "Depth Reference must be the four-corner plane created by Align Camera & Plane.")
            return {'CANCELLED'}
        context.scene.frame_set(ref_f)
        depth_matrix = evaluated_matrix_world(context, depth)
        world_points = [depth_matrix @ depth.data.vertices[index].co for index in range(4)]
        solve_reference = not _is_unedited_plane_alignment(depth, target, ref_f)
        observations, lenses = {}, {}
        for frame in range(start, end + 1):
            context.scene.frame_set(frame)
            lenses[frame] = target.data.lens
            marker = _plane_marker(clip, track, frame)
            if marker:
                observations[frame] = list(zip(world_points, _plane_corners(marker)))
        if ref_f not in observations:
            self.report({'ERROR'}, "Reference Frame needs a valid Plane Track marker.")
            return {'CANCELLED'}
        return self._bake_3d_observations(context, target, ref_f, start, end,
            observations, lenses, fixed_position=pcam_use_existing_position(props),
            solve_reference=solve_reference)

    def execute_clip_track_3d(self, context, target):
        props = context.scene.pcam_solve_props
        clip, depth = props.target_clip, props.clip_depth_object
        start, end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, start, end)
        tracks = clip.tracking.objects[int(props.tracking_object_idx)].tracks
        context.scene.frame_set(ref_f)
        ref_matrix = matrix_without_scale(evaluated_matrix_world(context, target))
        anchors = {}
        for track in tracks:
            marker = track.markers.find_frame(pcam_scene_to_clip_frame(clip, ref_f), exact=True)
            if marker and not marker.mute:
                hit = _depth_hit(context, target, ref_matrix, depth,
                    get_track_display_co(track, marker), target.data.lens)
                if hit is not None:
                    anchors[track.name] = hit
        if len(anchors) < 4:
            self.report({'ERROR'}, "Reference Frame needs four tracks hitting the Depth Reference.")
            return {'CANCELLED'}
        observations, lenses, markers_by_frame = {}, {}, {}
        for frame in range(start, end + 1):
            context.scene.frame_set(frame)
            lenses[frame] = target.data.lens
            current = {}
            for track in tracks:
                marker = track.markers.find_frame(pcam_scene_to_clip_frame(clip, frame), exact=True)
                if marker and not marker.mute:
                    current[track.name] = get_track_display_co(track, marker)
            markers_by_frame[frame] = current
        observations[ref_f] = [(anchors[name], co) for name, co in markers_by_frame[ref_f].items() if name in anchors]
        return self._bake_3d_observations(context, target, ref_f, start, end,
            observations, lenses, fixed_position=pcam_use_existing_position(props),
            dynamic_markers=markers_by_frame, depth=depth, reference_anchors=anchors)

    def execute_clip_track_object_3d(self, context, target):
        props = context.scene.pcam_solve_props
        scene, camera, clip, depth = context.scene, context.scene.camera, props.target_clip, props.clip_depth_object
        if depth.type != 'MESH':
            self.report({'ERROR'}, "3D Solve needs a mesh Depth Reference.")
            return {'CANCELLED'}
        start, end = pcam_get_frame_range(props)
        ref_f = pcam_get_reference_frame(context, props, start, end)
        tracks = clip.tracking.objects[int(props.tracking_object_idx)].tracks
        scene.frame_set(ref_f)
        camera_reference = matrix_without_scale(evaluated_matrix_world(context, camera))
        target_reference = target.matrix_world.copy()
        depth_reference = evaluated_matrix_world(context, depth)

        # A static copy keeps the reference surface fixed even when it is also the target.
        depth_proxy = depth.copy()
        proxy_mesh = depth.data.copy()
        depth_proxy.data = proxy_mesh
        depth_proxy.animation_data_clear()
        proxy_mesh.animation_data_clear()
        for constraint in list(depth_proxy.constraints):
            depth_proxy.constraints.remove(constraint)
        depth_proxy.parent = None
        pcam_camera_collection(context, camera).objects.link(depth_proxy)
        depth_proxy.matrix_world = depth_reference
        try:
            return self._bake_clip_track_object_3d(
                context, target, camera, clip, tracks, depth_proxy,
                start, end, ref_f, camera_reference, target_reference)
        finally:
            bpy.data.objects.remove(depth_proxy, do_unlink=True)
            if proxy_mesh.users == 0:
                bpy.data.meshes.remove(proxy_mesh)

    def _bake_clip_track_object_3d(self, context, target, camera, clip, tracks, depth,
                                   start, end, ref_f, camera_reference, target_reference):
        props = context.scene.pcam_solve_props
        scene = context.scene
        markers_by_frame, lenses, actual_cameras = {}, {}, {}
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            lenses[frame] = camera.data.lens
            actual_cameras[frame] = matrix_without_scale(evaluated_matrix_world(context, camera))
            current = {}
            for track in tracks:
                marker = track.markers.find_frame(pcam_scene_to_clip_frame(clip, frame), exact=True)
                if marker and not marker.mute:
                    current[track.name] = get_track_display_co(track, marker)
            markers_by_frame[frame] = current

        anchors = {}
        scene.frame_set(ref_f)
        for name, co in markers_by_frame[ref_f].items():
            hit = _depth_hit(context, camera, camera_reference, depth, co, lenses[ref_f])
            if hit is not None:
                anchors[name] = hit
        if len(anchors) < 4:
            self.report({'ERROR'}, "Reference Frame needs four tracks hitting the Depth Reference.")
            return {'CANCELLED'}

        virtual_cameras = {ref_f: camera_reference}
        fitted_frames = {ref_f}
        for direction in (range(ref_f + 1, end + 1), range(ref_f - 1, start - 1, -1)):
            known = dict(anchors)
            previous = ref_f
            for frame in direction:
                scene.frame_set(frame)
                current = markers_by_frame[frame]
                pairs = [(known[name], co) for name, co in current.items() if name in known]
                previous_pose = virtual_cameras[previous]
                result = None
                if len(pairs) >= 4:
                    result = _fit_camera(context, camera.data, pairs,
                        previous_pose.translation, previous_pose.to_quaternion(), lenses[frame])
                    if result is not None and result[2] > 12.0:
                        result = None
                if result is None:
                    virtual_pose = previous_pose.copy()
                else:
                    virtual_pose = Matrix.Translation(result[0]) @ result[1].to_matrix().to_4x4()
                    fitted_frames.add(frame)
                virtual_cameras[frame] = virtual_pose
                for name, co in current.items():
                    if name not in known:
                        hit = _depth_hit(context, camera, virtual_pose, depth, co, lenses[frame])
                        if hit is not None:
                            known[name] = hit
                previous = frame
        if len(fitted_frames) == 1 and end > start:
            self.report({'ERROR'}, "No usable 3D object poses were fitted in the bake range.")
            return {'CANCELLED'}

        frames = list(range(start, end + 1))
        matrices = {frame: actual_cameras[frame] @ virtual_cameras[frame].inverted() @ target_reference
                    for frame in frames}
        positions = {frame: matrices[frame].translation.copy() for frame in frames}
        rotations = {frame: matrices[frame].to_quaternion() for frame in frames}
        if props.clip_position_smooth > 1e-4:
            positions = smooth_vector_curve_global(positions, frames,
                strength=props.clip_position_smooth, passes=1 + int(3 * props.clip_position_smooth))
            positions[ref_f] = target_reference.translation.copy()
        if props.solve_3d_rotation_smooth > 1e-4:
            rotations = smooth_quaternion_curve_global(rotations, frames,
                strength=props.solve_3d_rotation_smooth,
                passes=1 + int(3 * props.solve_3d_rotation_smooth))
            rotations[ref_f] = target_reference.to_quaternion()

        snapshot = self.snapshot_animation_action(target)
        try:
            self.clear_animation_safely(target,
                (start, end) if props.use_custom_range else None)
            for frame in frames:
                scene.frame_set(frame)
                _location, _rotation, scale = matrices[frame].decompose()
                target.matrix_world = (Matrix.Translation(positions[frame]) @
                    rotations[frame].to_matrix().to_4x4() @
                    Matrix.Diagonal((scale.x, scale.y, scale.z, 1.0)))
                target.keyframe_insert(data_path="location", frame=frame)
                self.keyframe_target_rotation(target, frame)
                target.keyframe_insert(data_path="scale", frame=frame)
        except Exception:
            self.restore_animation_snapshot_exact(target, snapshot)
            raise
        scene.frame_set(ref_f)
        self.report({'INFO'}, f"Baked 3D object motion to '{target.name}': {len(fitted_frames)}/{len(frames)} frames fitted.")
        held_ranges = _held_frame_ranges(frames, fitted_frames)
        if held_ranges:
            self.report({'WARNING'}, f"3D Solve held the previous virtual pose on frames: {held_ranges}.")
        return {'FINISHED'}
