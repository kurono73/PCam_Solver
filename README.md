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

* Shots with many  feature points where Blender's standard camera solver works well.
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

It is intended for shots with more tracking points and provides a broader pseudo solve with optional smoothing and rotation refinement.

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

4. Choose whether to bake to the active `Camera` or a target `Object`.

5. Select the Movie Clip, tracking object, and required tracks.

   **Get Selected Tracks** automatically fills the track fields from the selected Movie Clip tracks.

6. Set a **Depth Reference** when required.

   This is usually a plane or object placed near the tracked surface. The `+` button creates a camera-facing Depth Reference Plane.

7. Choose a **Scale Method**:

   * `Z-Depth` – Converts apparent scale changes into movement along the camera's depth axis.
   * `Focal Length` – Converts apparent scale changes into focal length animation.
   * `None` – Ignores scale changes and solves only rotation and position. This behaves similarly to Blender's standard **Tripod Solve**, but uses a simpler solving method.

8. Adjust the solve settings if needed:

   * `Tripod` – Solves rotation without translating the camera. Useful for tripod shots or nearly stationary cameras.
   * `Dolly Motion` – In `Z-Depth` mode, converts apparent scale changes into camera movement along the depth axis.
   * `Lock Height` – Keeps the camera height fixed while solving horizontal movement and rotation.
   * `Smooth Jitter` – Smooths tracker motion before solving.
   * `Center Weighting` – Gives more influence to tracks near the image center.
   * `Lock Roll` – Available in **2 Point** and **3 Point** modes to prevent roll rotation.
   * Clip Track smoothing controls can smooth position, focal length, pan/tilt, and roll after baking.

9. Set the **Reference Frame**.

   The current frame is used by default. You can also lock a specific Reference Frame to produce repeatable results.

10. Optional settings:

* `Custom Range` – Bake only part of the clip.
* `Use Existing Position` – Keep existing position animation and recompute only rotation.
* `Use Existing Focal` – Keep existing focal length animation.
* `Preview Tracker Raycast` – Visualize where tracker rays intersect the Depth Reference.

11. Click `Bake Tracking to Target`.

12. Review the baked animation in the Viewport and Graph Editor.

If necessary, smooth or edit the baked curves manually. You can then use `Use Existing Position` or `Use Existing Focal` to recalculate only the remaining motion channels while preserving the edited animation.

---

# Usage Notes

* PCamSolver is a pseudo solve tool. It is not a full 3D camera reconstruction system.
* Before baking, roughly align the camera position and rotation on the **Reference Frame**. This usually produces more predictable results.
* PCamSolver does not estimate the real camera focal length or lens parameters. Set the camera focal length and sensor settings before baking.
* In **Focal Length** mode, the focal length on the **Reference Frame** is used as the base value for the solved animation.
* In **1/2/3 Point** camera modes, the camera cannot be exactly at the world origin. Move the camera slightly away from `(0, 0, 0)` before baking.
* In **1/2/3 Point** modes, the selected trackers should remain valid throughout the bake range.
* If trackers appear or disappear during the shot, consider using **Clip Track** mode. You can also solve shorter sections with **Custom Range** and use different trackers for each section.
* **Preview Tracker Raycast** is intended to help place the **Depth Reference**. It does not display undistorted tracker positions.
* Make sure the **Depth Reference** covers the full movement range of the trackers used for the solve. A simple plane usually provides the most stable and predictable results.
* In Object mode, animated rotation on the **Depth Reference** can be used as a simple guide for depth-related object rotation. Use **Preview Tracker Raycast** to verify that tracker rays remain inside the Depth Reference. Large rotations or insufficient coverage may produce unstable results.
* Dolly zoom shots are not directly supported. PCamSolver can interpret apparent scale changes as either depth movement or focal length animation, but it cannot separate both simultaneously.
* In Object target modes, place the Depth Reference near the depth of the target object's origin on the Reference Frame. 
