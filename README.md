# PCam_Solver
**A practical fallback for difficult matchmove shots that do not require a full camera solve.**

---

# Overview

**PCamSolver** is a pseudo 3D tracking tool for Blender. It converts 2D movie clip tracks into practical camera or object animation, including pseudo depth movement and focal length animation.

The addon is designed for matchmove workflows where a full camera solve is unnecessary, unstable, or too time-consuming. Instead of reconstructing a complete 3D camera, PCamSolver solves only the motion components required for the shot.

Using a small number of tracked points and an optional **Depth Reference** object, it can generate practical camera movement, object tracking, depth motion, or focal length animation from ordinary 2D tracks.

Unlike a full camera solver, PCamSolver lets you solve only the motion you actually need. This makes it especially useful for difficult shots with limited tracking information.

---

# Use Cases

PCamSolver is useful when a full camera solve is unnecessary or difficult, but some camera or object motion still needs to be matched.

Examples include:

* Close-up shots with only a few reliable tracking points.
* Camera motion that is mostly parallel to a wall or another flat surface.
* Simple forward or backward camera movement.
* Shots where all available trackers lie on a single plane and a standard camera solve cannot reconstruct reliable depth.
* Quick approximation of zoom or lens breathing, which is not directly handled by Blender's standard tracking tools.
* Cases where the camera position is animated manually, but the viewing direction needs to be refitted to tracked points.
* Simple object tracking for elongated objects such as swords, rods, pipes, or similar shapes.
* Tripod shots with only one reliable tracking point.

PCamSolver is **not** intended to replace a full camera solve when sufficient tracking information is available.

It is less suitable for:

* Shots with many feature points where Blender's standard camera solver works well.
* Shots with large camera movement.
* Shots with strong parallax that require true 3D camera reconstruction.

---

# Features

PCamSolver focuses on fast, controllable pseudo 3D solving rather than full camera reconstruction.

It converts limited 2D tracker motion into practical camera or object animation. Depending on the shot, it can use one, two, three, or many clip tracks. An optional **Depth Reference** object provides a spatial reference for depth-aware solving.

### 1 Point, 2 Point and 3 Point

These modes convert a small number of tracked points into camera or object animation.

Internally, they build on Blender's **Follow Track** constraint and bake the evaluated result into standard transform or lens animation.

In **2 Point** and **3 Point** modes, apparent scale changes between tracked points can be interpreted in two different ways:

* **Z-Depth** converts scale changes into movement along the camera's depth axis.
* **Focal Length** converts scale changes into focal length animation.

The tracked points remain fixed reference targets while the camera or object orientation is continuously refitted toward them.

### Clip Track

Clip Track mode uses many available movie clip tracks directly.

Its standard solve supports tracks that appear and disappear, with optional smoothing and rotation refinement. For Camera targets, `Scale Method: Focal Length` converts apparent scale change into lens animation.

`Scale Method: 3D Solve` fits Camera or Object motion from at least four visible tracks on the Depth Reference. It uses the camera's existing lens value at each frame rather than estimating zoom. Track changes are supported when enough tracks overlap to carry the solve forward.

### Plane Track

Plane Track mode uses the four corners of a Blender Plane Track to solve a Camera or Object against a four-corner Depth Reference plane. It is suited to tracked flat surfaces such as screens and walls.

> **Tip:** With the [Clip Tools add-on](https://extensions.blender.org/add-ons/clip-tools/), select four point tracks with enabled markers outlining the surface on the current frame, then use **Create Plane Track** in the Movie Clip Editor's Track sidebar. It creates a Plane Track from the selected marker positions, which you can then select in PCamSolver.

**Plane Setup** aligns on the current timeline frame. For Camera targets, `Alignment: Wall` and `Floor` place the plane at the world origin and align the camera; `Camera` preserves the camera pose and aligns the plane instead. For Object targets, the camera pose is preserved and the plane is aligned to the track. If a four-vertex mesh plane is selected as the Depth Reference, Align updates it regardless of its name; an incompatible selected object produces an error instead of creating another plane. Alignment updates keys already present on that frame but does not add new keys.

The plane size can be specified by `Height`, `Width`, or `Dimensions`. With one dimension, the other is inferred from the tracked plane. `Dimensions` keeps both supplied values and refits the pose to the four corners. In Camera and Object modes, optional **Auto Lens** beside the Align button estimates the camera focal length from known width and height; it is unavailable with a single specified dimension. If the four corners do not constrain the estimate reliably, alignment continues with the existing lens and reports that estimation was unavailable.

For Camera targets, `Solve Method: 3D Position` solves camera position and rotation with the existing lens animation. `Focal Length` solves rotation and lens animation relative to the Reference Frame's lens value. With `Tripod` on, camera position stays fixed; with it off, the camera can move parallel to the reference plane while its depth remains fixed. **Use Existing Position** takes priority over `Tripod` and preserves the position curve. For Object targets, Plane Track bakes object motion using the scene camera and its existing animation. If no Target Object is selected, **Align Plane to Track** assigns the aligned Depth Reference plane as the target.

### Depth Reference

The **Depth Reference** defines where tracker rays intersect the scene.

It provides the spatial reference used to estimate depth-aware movement and determines how much apparent 2D motion becomes real 3D movement.

Some rotation-only modes and tripod-style modes do not require a Depth Reference.


---

# How to Use

Before baking, set the camera focal length and sensor settings as close as possible to the footage.  
In **Focal Length** mode, the lens value on the **Reference Frame** becomes the base value for the solved focal length animation.

1. Track points in Blender's **Movie Clip Editor**.

2. Open the **P-Cam** tab in the 3D Viewport sidebar.

3. Choose a solve mode:

   * `1 Point Track` – Pan/Tilt or simple tripod motion from one tracker.
   * `2 Point Track` – Adds scale and roll estimation from two trackers.
   * `3 Point Track` – Provides a more stable pseudo 3D solve.
   * `Clip Track` – Uses many movie clip tracks for a broader solve.
   * `Plane Track` – Uses the four corners of a Plane Track to fit Camera or Object motion to a flat surface.

4. Choose whether to bake to the active `Camera` or a target `Object`.

5. Select the Movie Clip, track layer, and required tracks or Plane Track.

   **Get Selected Tracks** automatically fills the point-track fields in 1/2/3 Point modes.

6. Set a **Depth Reference** when required.

   This is usually a plane or object placed near the tracked surface. The `+` button creates a camera-facing Depth Reference Plane in the active camera's collection.

   In Plane Track mode, use **Plane Setup** to choose `Alignment` and size, then click **Align Camera & Plane** (or **Align Plane to Track** for Object targets). `Height` and `Width` infer the unspecified dimension; `Dimensions` uses both supplied values. In either target mode, turn on **Auto Lens** beside the Align button to estimate camera focal length from `Dimensions`. Align uses the current timeline frame. For Object targets, the aligned plane becomes the Target Object only if none is selected.

7. For Camera targets in 2/3 Point or Clip Track, choose a **Scale Method**:

   * `Z-Depth` – Converts apparent scale changes into movement along the camera's depth axis.
   * `Focal Length` – Converts apparent scale changes into focal length animation.
   * `None` – Ignores scale changes and solves only rotation and position. This behaves similarly to Blender's standard **Tripod Solve**, but uses a simpler solving method.

   In Clip Track, `3D Solve` is another Scale Method. It fits Camera or Object motion from four or more tracks on the Depth Reference using the existing lens animation. Object targets offer `Z-Depth` and `3D Solve`.

   Plane Track Camera mode instead offers `Solve Method: 3D Position` or `Focal Length`.

8. Adjust the solve settings if needed:

   * `Tripod` – In Plane Track `Focal Length`, ON keeps the camera at its Reference Frame position; OFF allows movement parallel to the plane while solving rotation and zoom. **Use Existing Position** overrides this setting. Other modes retain their existing Tripod behavior.
   * `Dolly Motion` – In `Z-Depth` mode, converts apparent scale changes into camera movement along the depth axis.
   * `Lock Height` – Keeps the camera height fixed while solving horizontal movement and rotation.
   * `Smooth Jitter` – Smooths tracker motion before solving.
   * `Center Weighting` – Gives more influence to tracks near the image center.
   * `Lock Roll` – Available in **2 Point** and **3 Point** modes to prevent roll rotation.
   * Plane Track `Focal Smooth`, `Position Smooth`, and `Rotation Smooth` appear only where applicable to the selected solve method and target.
   * Standard Clip Track smoothing controls can smooth position, focal length, pan/tilt, and roll during the solve.

9. Set the **Reference Frame**.

   The current frame is used by default. You can also lock a specific Reference Frame to produce repeatable results. Plane Track alignment uses the active frame; when the Reference Frame is locked, alignment updates it to that frame.

10. Optional settings:

   * `Custom Range` – Bake only part of the clip while preserving keys outside that range.
   * `Use Existing Position` – Keep existing position animation and solve the remaining channels where supported. In Plane Track `Focal Length`, it keeps position while solving rotation and lens animation.
   * `Use Existing Focal` – Keep existing focal length animation where this option is available.
   * `Preview Tracker Raycast` – Visualize where tracker rays intersect the Depth Reference.

11. Click `Bake Tracking to Target`.

12. Review the baked animation in the Viewport and Graph Editor.

If necessary, smooth or edit the baked curves manually. Where supported, `Use Existing Position` and `Use Existing Focal` let you refit the remaining channels while preserving those edits.

---

# Usage Notes

* PCamSolver is a pseudo solve tool. It is not a full 3D camera reconstruction system.
* Before baking, roughly align the camera position and rotation on the **Reference Frame**. This usually produces more predictable results.
* PCamSolver does not generally calibrate the camera's base focal length or sensor parameters. Set them before baking; an incorrect field of view can increase solve error. Plane Track **Auto Lens** is an optional single-frame estimate when both plane dimensions are known, not a full lens calibration. Check its result against the footage.
* In **Focal Length** mode, the focal length on the **Reference Frame** is used as the base value for the solved animation.
* In **1/2/3 Point** camera modes, the camera cannot be exactly at the world origin. Move the camera slightly away from `(0, 0, 0)` before baking.
* In **1/2/3 Point** modes, the selected trackers should remain valid throughout the bake range.
* If trackers appear or disappear during the shot, consider using **Clip Track** mode. You can also solve shorter sections with **Custom Range** and use different trackers for each section.
* **Plane Track** requires valid tracked corners over the frames you want to solve. Align the plane on the intended Reference Frame before baking. In `Dimensions` mode, a supplied aspect ratio that conflicts with the tracked corners will leave a reprojection mismatch.
* Clip Track **3D Solve** needs at least four Depth Reference hits on the Reference Frame and four usable tracks to fit a frame. Frames without enough information hold the previous solved relative pose; overlapping tracks help when trackers change. After baking, a warning lists the frame ranges where the previous pose was held.
* In Plane Track Object mode, position the target object as desired on the Reference Frame before baking. Its initial offset from the Depth Reference is preserved.
* **Preview Tracker Raycast** is intended to help place the **Depth Reference**. It does not display undistorted tracker positions.
* Make sure the **Depth Reference** covers the full movement range of the trackers used for the solve. A simple plane usually provides the most stable and predictable results.
* In Object mode, animated rotation on the **Depth Reference** can be used as a simple guide for depth-related object rotation. Use **Preview Tracker Raycast** to verify that tracker rays remain inside the Depth Reference. Large rotations or insufficient coverage may produce unstable results.
* Dolly zoom shots are not directly supported. PCamSolver can interpret apparent scale changes as either depth movement or focal length animation, but it cannot separate both simultaneously.
* In Object target modes, place the Depth Reference near the depth of the target object's origin on the Reference Frame. 
