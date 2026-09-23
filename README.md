# RoboFinal

## macOS setup (Apple Silicon and Intel)

`pip install robomaster` does not work on a Mac: DJI only publishes Linux and
Windows packages. The SDK is pure Python except for its video decoder
(`libmedia_codec`). The setup script installs the SDK's Python code and
replaces the decoder with a PyAV (FFmpeg) version that works on macOS:

```bash
bash tools/macos/setup_robomaster_mac.sh
```

Then set the robot's connection switch to Wi-Fi direct (AP), join its
`RMEP-xxxxxx` Wi-Fi on the Mac, and check the connection:

```bash
.venv/bin/python tools/macos/check_robomaster.py --snapshot frame.jpg
```

The check shows OK or FAIL for each step (Wi-Fi, SDK connection, version and
battery, camera stream). If a step fails, it prints a hint. Run the project
scripts with the same Python, for example
`.venv/bin/python src/gimbal_shooter.py`.

## RoboMaster camera viewer

Connect the computer to the RoboMaster robot (the default configuration uses
the robot's Wi-Fi access-point mode), then install the dependencies and run:

```bash
python3 -m pip install -r requirements.txt
python3 src/camera_view.py
```

Press `q` or `Esc` in the camera window to stop. The connection type and video
resolution can also be selected explicitly:

```bash
python3 src/camera_view.py --connection ap --resolution 720p
```

Supported connection types are `ap`, `sta`, and `rndis`; supported resolutions
are `360p`, `540p`, and `720p`.

The viewer automatically looks for a solid red rectangular card. When it finds
one, it draws a green box around the card and shows `RED CARD DETECTED` in the
video window. If a small card is too far away to detect, lower the minimum pixel
area (lower values can also cause more false detections):

```bash
python3 src/camera_view.py --min-card-area 700
```

## Auto-aim water ball shooter

`src/gimbal_shooter.py` drives the robot with the keyboard, tracks a target
with the gimbal, and shoots water balls. It only fires at an object that
matches all three checks set in `target_shooting` in `config/settings.yaml`:
**color**, **shape** (circle/square/rectangle/triangle), and **distance**
(worked out from the target's real `size_m`).

```bash
python3 src/gimbal_shooter.py                                   # use the config
python3 src/gimbal_shooter.py --color blue --shape square --size 0.15
python3 src/gimbal_shooter.py --auto-fire                       # start with auto-fire on
```

| Keys | Action |
| --- | --- |
| `W` `A` `S` `D` | drive forward / left / back / right |
| `Q` `E` | rotate the chassis |
| `I` `J` `K` `L` | move the gimbal by hand (overrides auto-aim) |
| `Space` | fire one water ball |
| `T` / `F` | turn auto-track / auto-fire on or off (auto-fire starts **off**) |
| `X` / `Z` | switch the target color / shape |
| `C` | calibrate distance (put the target `calibration_distance_m` away) |
| `[` `]` / `,` `.` | adjust pitch / yaw aim trim while running |
| `R` / `Esc` | recenter the gimbal / quit |

Install `pynput` to hold several keys at once (for example `W`+`D`). On macOS,
the terminal also needs the Accessibility and Input Monitoring permissions.
Without `pynput`, the program reads keys from the video window instead.

To test without a robot, use a webcam, a video, or an image (robot commands
are printed instead of sent):

```bash
python3 src/gimbal_shooter.py --source 0
python3 src/target_detection.py --source photo.jpg
python3 -m unittest discover -s tests -v
```

Field tuning: 1) press `C` with the target 1 m away and copy the
`focal_length_px` value into the config; 2) fire at 0.5 m, 1 m, 2 m and 3 m,
then adjust `pitch_compensation` until the shots land on the target.

## SLAM: explore an unknown area (Class Work 8)

`src/slam_explore.py` explores an area with no map, builds the map from the
ToF sensor on top of the gimbal, estimates the robot's own position, and
reports where the robot started and ended. A live console in the browser shows
everything while it runs.

```bash
.venv/bin/python src/slam_explore.py --sim        # try it first with the built-in simulator
.venv/bin/python src/slam_explore.py              # real robot (Wi-Fi AP mode)
.venv/bin/python src/slam_explore.py --gt data/slam/ground_truth_example.json   # robot + ground truth for scoring
```

A console window (pygame) opens. Press **Start mission**; the robot
calibrates, then repeats *scan → localise → update map → pick frontier →
drive* until nothing is left to explore. **STOP** (or Space) halts all motion,
**Pause** freezes the mission, **Finish & report** ends it and writes the
report. Add `--web` to get the same console in a browser at
<http://localhost:8765> instead (`--host 0.0.0.0` to open it from a phone).

Window keys: Space = STOP, W/A/S/D = drive/strafe, Q/E = rotate (while no text
box is being edited), F = fit map, R = follow robot, +/- = zoom, mouse wheel =
zoom, drag = pan. Number boxes apply on Enter.

| Console part | What it does |
| --- | --- |
| Map | Live occupancy grid, trajectory, last scan, frontiers, planned path, ground truth. Zoom (wheel), pan (drag), rotate the view (Rotate L/R, also rotates saved images), follow robot, layer toggles. |
| Map modes | **Go to** (click a target) · **Set pose** (drag to fix the robot's pose) · **Border** (drag the area to explore and score) · **GT wall** / **GT arena** (draw the ground truth). |
| Status | Map Accuracy and Coverage, SLAM pose vs odometry vs drift correction, start pose, pose in the arena frame, ToF, gimbal, mission stats, report (start / end), charts of ToF, coverage, accuracy, speed and drift correction over the whole run, polar plot of the last scan. |
| Control | WASD/QE manual drive with speed sliders, rotate robot by ±45/90/180 or any angle, move, go to x/y, go home, scan now, auto calibrate, aim gimbal, ToF wall calibration, set / rotate the pose estimate, clear map, new session. |
| Settings | Every setting (map size and resolution, border, scan range/speed/mode, noise filters, localisation, speeds, safety distances, exploration limits, scoring) with ranges and help. Changes apply at once; **Save settings** writes `config/slam_settings.yaml`. |
| Ground truth | Load (file path) / draw / save the arena map as JSON and set where the robot starts in it. |
| Results | The run folder's files (click to open) and the latest saved map. |

### Auto grid (maze arenas)

The arena is treated as a maze whose walls lie on a square grid
(`grid_mode`, Settings > Grid):

- **Finding the grid** – the grid angle comes from the directions of the walls
  (they meet at 90°); the cell size and offset are the ones that put the most
  wall points on grid lines. The angle locks once three detections agree; the
  cell size only locks after three confident, agreeing detections, so a noisy
  start never locks a wrong grid. If you know your tile size, set
  `grid_mode: fixed` and `grid_cell_m` and it locks at once.
- **Straight maps** – each scan is snapped to the locked grid (heading, then
  position), and every grid edge is voted *wall / open / unknown*. A wall seen
  along part of an edge becomes the whole edge (gaps filled); blobs off the grid
  lines disappear. The Clean layer shows this map; Maze draws the voted walls
  (unknown edges dashed).
- **Moving grid by grid, scanning grid by grid** – once the grid is known the
  robot moves one cell at a time (`grid_cells_per_step: 1`) and scans in every
  cell. Each step: turn to the grid axis, strafe back onto the cell's centre
  line (mecanum wheels), point the ToF ahead and check the edge is really open
  (if a wall is closer than the edge, it is marked as a wall and the robot does
  not move), then drive exactly to the next cell centre and scan. Routes only
  use open edges, toward the nearest cell that is unseen or still has an unknown
  wall; the mission ends when none is left. With `grid_mode: fixed` this starts
  after the first scan; in auto mode the robot explores freely until the grid
  is confirmed.

Rebuild a recorded run with the current settings (no robot needed), e.g. to try
another cell size:

```bash
python3 analysis/replay_slam.py data/slam/run_20260923_103629 --set grid_mode=fixed grid_cell_m=0.63
```

### What gets saved (`data/slam/run_<date>_<time>/`)

| File | Content |
| --- | --- |
| `map.png`, `map_with_ground_truth.png` | The map (clean grid map when a grid was found) with trajectory, start (green) and end (red); `map_raw.png` before clean-up |
| `map_grid.png`, `grid_model.json` | The maze drawn straight on its grid, and every cell edge (wall / open / unknown) |
| `map_cells.csv`, `map_prob.npy`, `map_meta.json` | The map cells (0 unknown, 1 free, 2 wall) and probabilities |
| `trajectory.csv`, `log_pose.csv` | Robot trajectory (SLAM pose and odometry at 10 Hz) |
| `log_events.csv`, `log_scans_raw.csv`, `log_scans_filtered.csv` | Exploration log, every raw ToF reading, every filtered beam |
| `comparison.png` | Ground-truth check: white/black correct, red missed wall, orange false wall, grey unexplored |
| `report.md`, `report.json` | Start and end pose (map and arena frame), Map Accuracy, Coverage, calibration |

**Map Accuracy** = correct cells / all arena cells × 100 and **Coverage** =
explored cells / all arena cells × 100. Unknown cells never count as correct.
`wall_tolerance_cells` (default 1) lets a wall found one cell off still count;
the report also gives the strict score. Without a ground truth, set a
**Border** to get Coverage.

### Ground truth

Measure the arena and write it as JSON in metres (see
`data/slam/ground_truth_example.json`), or draw it in the console:

```json
{"name": "Lab maze", "border": [0, 0, 3.6, 3.0], "wall_thickness": 0.02,
 "walls": [[0.6, 0.0, 0.6, 1.2], [1.2, 0.6, 2.4, 0.6]]}
```

Then enter the robot's start pose in that frame (Ground truth tab, or
`gt_start_x`, `gt_start_y`, `gt_start_deg`).

### How it works (for the presentation)

1. **Sensing** – the gimbal turns the ToF sensor through 360° (sweep mode, or
   stop-and-average step mode). Each reading is tagged with the gimbal angle at
   the moment it was measured (latency-compensated).
2. **Noise reduction** – range gate, per-angle median with MAD outlier
   rejection, removal of spikes and dropouts that disagree with both
   neighbours, lone points with no neighbouring wall point (the ToF's wide beam
   makes them at wall ends), readings that pass through a wall the map is
   already sure of are dropped, then log-odds fusion averages many scans, small
   wall blobs are hidden, and the auto grid snaps walls onto grid lines.
3. **Mapping** – a log-odds occupancy grid: cells along each beam become more
   *free*, the cell where it ends more *occupied*; the wedge between
   neighbouring beams is cleared too so no gaps are left.
4. **Localisation** – wheel odometry + IMU heading give a first guess; each new
   scan is then matched to the map (correlative search over a small window,
   then Gauss-Newton refinement on a likelihood field) to remove the drift.
5. **Exploration** – frontier-based: the boundary between known free space and
   unknown space. The robot drives (Dijkstra path on a costmap that keeps it
   away from walls) to the best frontier, scans again, and stops when no
   reachable frontier is left. A forward-ToF emergency stop guards every move.
6. **Auto calibration** – ToF noise (sets the filter), gyro drift, the sign of
   the reported yaw and of turn commands (by scanning before and after a turn),
   the ToF latency (sweeping both ways), and the odometry y sign. A known-distance
   wall calibration corrects the ToF offset.

The simulator (`--sim`) uses the ground-truth file as its world and adds ToF
noise, outliers, dropouts, latency, odometry scale error and gyro drift, so the
whole pipeline can be tested without the robot. Useful flags: `--time-scale 10`
(faster simulation), `--auto --headless` (run a mission and exit),
`--set scan_mode=step linear_speed_mps=0.2` (override settings).

**On the real robot:** if the saved map comes out mirrored, flip
`gimbal_yaw_sign` and run **Auto calibrate** again. Check `stop_distance_m`
against your robot's bumper before the first run.
