# Standalone ROS hand scene viewer

This directory is an isolated test harness. It does not modify or replace the
existing `live_display/index.html`.

Start the normal ROS launch first so rosbridge is available at
`ws://127.0.0.1:9090`, then run:

```bash
python3 gesture_detector/gesture_detector/live_display/scene_viewer_test/server.py --port 6358
```

Open <http://127.0.0.1:6358/>.

The viewport subscribes to the hand frame, scene, pointing beam,
`/teleop_gesture_toolbox/deictic_solution`, and `/tf_static`. The deictic
solution shows the object the ray is on right now as a blue halo growing with
its evidence. The steady green ring is the actual selection, taken from
`/teleop_gesture_toolbox/pending_object_selection`, which the sentence maker
publishes from the same `deictic_evidence.select` that decides what it sends —
the viewer never runs a selection rule of its own. Use the compact
**Stats** overlay button to reveal connection and data diagnostics.

Options:

- `?hand_hz=60` changes the ROS hand-message cap from its 30 Hz default.
- `?demo=1` runs a synthetic scene without ROS or hand hardware.

Run the dependency-free JavaScript tests from the repository root:

```bash
node --test gesture_detector/gesture_detector/live_display/scene_viewer_test/tests/*.test.mjs
```

Three.js r185 and `OrbitControls` are vendored under `vendor/three` so the
viewer requires no runtime internet access. Their MIT license is included.
