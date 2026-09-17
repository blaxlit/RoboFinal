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
