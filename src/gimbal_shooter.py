"""Drive with WASD, auto-aim the gimbal at a detected target and shoot water balls.

A target is only engaged when it matches the configured COLOUR, SHAPE and a
DISTANCE inside the allowed range (see ``target_shooting`` in
config/settings.yaml). Detection lives in ``target_detection.py``.

Run on the robot:
    python3 src/gimbal_shooter.py
    python3 src/gimbal_shooter.py --color blue --shape square --auto-fire

Test without a robot (webcam / video / image, robot commands are only printed):
    python3 src/gimbal_shooter.py --source 0
    python3 src/gimbal_shooter.py --source clip.mp4 --headless --max-frames 300 --save out.mp4

Keys (click the video window first):
    W/S forward/back   A/D strafe left/right   Q/E rotate left/right
    I/K gimbal up/down J/L gimbal left/right   (manual gimbal overrides auto-aim)
    SPACE fire         T auto-track on/off     F auto-fire on/off
    X next colour      Z next shape            C calibrate distance
    [ / ] pitch trim   , / . yaw trim          R recenter gimbal   ESC quit
    G patrol along the wall (auto drive, stops to shoot)   N measure card size
    CLICK a card: learn its colour for this room's light
    M show colour mask P save raw camera frame (data/captures/)
"""

import argparse
import collections
import math
import os
import queue
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from config_loader import load_config
from target_detection import (
    TARGET_SHAPES,
    ColorShapeDetector,
    TargetTracker,
    _deep_merge,
    _is_number,
    _require,
    _text,
    build_detection_settings,
    draw_detections,
    pixel_to_angles,
)

# ==============================================================================
# 1. Settings
# ==============================================================================
DEFAULT_SHOOTER_SETTINGS = {
    "aiming": {
        "yaw_kp": 3.0, "yaw_ki": 0.0, "yaw_kd": 0.0,
        "pitch_kp": 3.0, "pitch_ki": 0.0, "pitch_kd": 0.0,
        "max_yaw_speed": 180.0,      # deg/s
        "max_pitch_speed": 120.0,
        "min_yaw_speed": 6.0,        # friction compensation
        "min_pitch_speed": 8.0,
        "hold_enter_ratio": 0.5,     # stop when error < ratio * lock tolerance (hysteresis)
        "max_accel_dps2": 600.0,     # smooth speed changes
        "target_smoothing": 0.25,    # EMA on the target's absolute angle (lower = steadier)
        "refine_delay_s": 0.5,       # after stopping, correct a small leftover error once
        "reseed_deg": 4.0,           # jump bigger than this = new position, no smoothing
        "camera_latency_s": 0.20,    # image delay over Wi-Fi; used to look up past gimbal angle
        "yaw_offset_deg": 0.0,       # barrel vs camera trim (+ = aim right)
        "pitch_offset_deg": 0.0,     # (+ = aim higher)
        # [distance_m, extra pitch up deg]: camera-above-barrel parallax + ball drop.
        "pitch_compensation": [[0.5, 5.0], [1.0, 3.0], [2.0, 2.0], [3.0, 2.2]],
        "lock_tolerance_ratio": 0.4, # of the target's angular radius
        "min_lock_tolerance_deg": 1.0,
        "max_lock_tolerance_deg": 2.0,
        "lock_frames": 3,
        "max_lock_rate_dps": 8.0,    # error must be settling slower than this to fire
        "pitch_limits_deg": [-20.0, 30.0],
    },
    "firing": {
        "auto_fire": False,          # safety: press F (or --auto-fire) to arm
        "cooldown_s": 1.5,           # time between shots: let the picture settle and re-detect
        "max_shots_per_target": 3,
        "fire_times": 1,             # balls per trigger pull
    },
    "knockdown": {
        "enabled": True,
        "fall_pitch_deg": 2.5,       # card centre dropping this much = it fell over
        "fall_missing_frames": 8,    # ...or it simply disappeared after being shot
        "blacklist_seconds": 30.0,   # ignore that spot afterwards (do not re-shoot a downed card)
        "blacklist_radius_deg": 4.0,
        # Firing shakes the gimbal and blurs the picture, so the card is often missed
        # for a few frames. Nothing is judged until this long after the shot.
        "settle_seconds": 1.2,
        "skip_after_max_shots": True,  # card survived its ammo budget: leave it, try another
    },
    "patrol": {
        "enabled": False,            # press G to start; WASD always overrides
        "speed_mps": 0.25,           # sideways speed along the wall
        "wall_distance_mm": 1200,    # hold this distance to the wall (front ToF)
        "distance_kp": 0.0006,       # m/s per mm of distance error
        "max_approach_mps": 0.15,
        "min_distance_mm": 1000,     # never get closer than this, patrol or manual
        "leg_seconds": 8.0,          # reverse direction after this long
        "stop_to_shoot": True,
    },
    "controls": {
        "keyboard_backend": "auto",  # auto | pynput | opencv
        "chassis_speed_mps": 0.5,
        "chassis_rotate_dps": 90.0,
        "gimbal_speed_dps": 60.0,
        "command_rate_hz": 20.0,
        "robot_mode": "free",        # free | gimbal_lead | chassis_lead
        "auto_track": True,
    },
}


def validate_shooter_settings(s):
    aim, fire, ctl = s["aiming"], s["firing"], s["controls"]
    for key in ("yaw_kp", "yaw_ki", "yaw_kd", "pitch_kp", "pitch_ki", "pitch_kd",
                "min_yaw_speed", "min_pitch_speed", "camera_latency_s",
                "lock_tolerance_ratio", "min_lock_tolerance_deg", "max_lock_rate_dps"):
        _require(_is_number(aim[key]) and aim[key] >= 0, f"aiming.{key} must be >= 0")
    for key in ("max_yaw_speed", "max_pitch_speed"):
        _require(_is_number(aim[key]) and 0 < aim[key] <= 540, f"aiming.{key} must be in (0, 540]")
    _require(aim["min_yaw_speed"] < aim["max_yaw_speed"] and aim["min_pitch_speed"] < aim["max_pitch_speed"],
             "aiming.min_*_speed must be below max_*_speed")
    _require(_is_number(aim["hold_enter_ratio"]) and 0 < aim["hold_enter_ratio"] <= 1,
             "aiming.hold_enter_ratio must be in (0, 1]")
    _require(_is_number(aim["max_accel_dps2"]) and aim["max_accel_dps2"] >= 50,
             "aiming.max_accel_dps2 must be >= 50")
    _require(_is_number(aim["target_smoothing"]) and 0 < aim["target_smoothing"] <= 1,
             "aiming.target_smoothing must be in (0, 1]")
    _require(_is_number(aim["refine_delay_s"]) and aim["refine_delay_s"] >= 0,
             "aiming.refine_delay_s must be >= 0")
    _require(_is_number(aim["reseed_deg"]) and aim["reseed_deg"] > 0, "aiming.reseed_deg must be > 0")
    _require(_is_number(aim["camera_latency_s"]) and aim["camera_latency_s"] <= 1.0,
             "aiming.camera_latency_s must be <= 1.0")
    _require(_is_number(aim["yaw_offset_deg"]) and _is_number(aim["pitch_offset_deg"]),
             "aiming offsets must be numbers")
    _require(_is_number(aim["max_lock_tolerance_deg"])
             and aim["max_lock_tolerance_deg"] >= aim["min_lock_tolerance_deg"],
             "aiming.max_lock_tolerance_deg must be >= min_lock_tolerance_deg")
    _require(isinstance(aim["lock_frames"], int) and aim["lock_frames"] >= 1,
             "aiming.lock_frames must be an integer >= 1")
    table = aim["pitch_compensation"]
    _require(isinstance(table, list) and table
             and all(isinstance(r, list) and len(r) == 2 and all(_is_number(v) for v in r) for r in table),
             "aiming.pitch_compensation must be a list of [distance_m, degrees]")
    distances = [r[0] for r in table]
    _require(distances == sorted(distances) and len(set(distances)) == len(distances),
             "aiming.pitch_compensation distances must be strictly increasing")
    limits = aim["pitch_limits_deg"]
    _require(isinstance(limits, list) and len(limits) == 2 and all(_is_number(v) for v in limits)
             and limits[0] < limits[1], "aiming.pitch_limits_deg must be [min, max]")

    _require(isinstance(fire["auto_fire"], bool), "firing.auto_fire must be true/false")
    _require(_is_number(fire["cooldown_s"]) and fire["cooldown_s"] >= 0.1, "firing.cooldown_s must be >= 0.1")
    _require(isinstance(fire["max_shots_per_target"], int) and fire["max_shots_per_target"] >= 1,
             "firing.max_shots_per_target must be an integer >= 1")
    _require(isinstance(fire["fire_times"], int) and 1 <= fire["fire_times"] <= 8,
             "firing.fire_times must be an integer 1-8")

    kd, pat = s["knockdown"], s["patrol"]
    _require(isinstance(kd["enabled"], bool) and isinstance(kd["skip_after_max_shots"], bool),
             "knockdown.enabled and knockdown.skip_after_max_shots must be true/false")
    for key in ("fall_pitch_deg", "blacklist_seconds", "blacklist_radius_deg", "settle_seconds"):
        _require(_is_number(kd[key]) and kd[key] >= 0, f"knockdown.{key} must be >= 0")
    _require(isinstance(kd["fall_missing_frames"], int) and kd["fall_missing_frames"] >= 1,
             "knockdown.fall_missing_frames must be an integer >= 1")

    _require(isinstance(pat["enabled"], bool) and isinstance(pat["stop_to_shoot"], bool),
             "patrol.enabled and patrol.stop_to_shoot must be true/false")
    _require(_is_number(pat["speed_mps"]) and 0 < pat["speed_mps"] <= 1.0,
             "patrol.speed_mps must be in (0, 1]")
    _require(_is_number(pat["max_approach_mps"]) and 0 <= pat["max_approach_mps"] <= 1.0,
             "patrol.max_approach_mps must be in [0, 1]")
    _require(_is_number(pat["wall_distance_mm"]) and pat["wall_distance_mm"] > 0,
             "patrol.wall_distance_mm must be > 0")
    _require(_is_number(pat["min_distance_mm"]) and 0 < pat["min_distance_mm"] < pat["wall_distance_mm"],
             "patrol.min_distance_mm must be > 0 and below patrol.wall_distance_mm")
    _require(_is_number(pat["distance_kp"]) and pat["distance_kp"] >= 0, "patrol.distance_kp must be >= 0")
    _require(_is_number(pat["leg_seconds"]) and pat["leg_seconds"] > 0, "patrol.leg_seconds must be > 0")

    _require(ctl["keyboard_backend"] in ("auto", "pynput", "opencv"),
             "controls.keyboard_backend must be auto, pynput or opencv")
    _require(_is_number(ctl["chassis_speed_mps"]) and 0 < ctl["chassis_speed_mps"] <= 3.5,
             "controls.chassis_speed_mps must be in (0, 3.5]")
    _require(_is_number(ctl["chassis_rotate_dps"]) and 0 < ctl["chassis_rotate_dps"] <= 600,
             "controls.chassis_rotate_dps must be in (0, 600]")
    _require(_is_number(ctl["gimbal_speed_dps"]) and 0 < ctl["gimbal_speed_dps"] <= 540,
             "controls.gimbal_speed_dps must be in (0, 540]")
    _require(_is_number(ctl["command_rate_hz"]) and 1 <= ctl["command_rate_hz"] <= 50,
             "controls.command_rate_hz must be in [1, 50]")
    _require(ctl["robot_mode"] in ("free", "gimbal_lead", "chassis_lead"),
             "controls.robot_mode must be free, gimbal_lead or chassis_lead")
    _require(isinstance(ctl["auto_track"], bool), "controls.auto_track must be true/false")


def build_settings(config):
    settings = build_detection_settings(config)
    user = (config or {}).get("target_shooting", {}) or {}
    for key, defaults in DEFAULT_SHOOTER_SETTINGS.items():
        settings[key] = _deep_merge(defaults, user.get(key))
    validate_shooter_settings(settings)
    return settings


# ==============================================================================
# 2. Aim controller
# ==============================================================================
class PID:
    def __init__(self, kp, ki, kd, limit, integral_limit=10.0, d_alpha=0.4):
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, limit
        self.integral_limit, self.d_alpha = integral_limit, d_alpha
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.last_error = None
        self.last_time = None
        self.d_filtered = 0.0

    def compute(self, error, now):
        dt = 0.05 if self.last_time is None else now - self.last_time
        if dt <= 0 or dt > 0.5:
            dt = 0.05
        if self.last_error is not None:
            raw_d = (error - self.last_error) / dt
            self.d_filtered = self.d_alpha * raw_d + (1 - self.d_alpha) * self.d_filtered
        output = self.kp * error + self.ki * self.integral + self.kd * self.d_filtered
        # Anti-windup: only integrate while not saturated.
        if abs(output) < self.limit:
            self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral + error * dt))
        self.last_error, self.last_time = error, now
        return max(-self.limit, min(self.limit, output))


class AimController:
    """Turns a tracked target into smooth gimbal speeds and a 'locked' flag.

    Why it does not shake:
    * Every frame is converted to the target's ABSOLUTE gimbal angle using the
      gimbal angle from when that frame was captured (camera latency). The target
      does not move when the gimbal moves, so this value can be heavily smoothed
      without adding lag to the gimbal's own motion.
    * The raw detection centre is used (the tracker's pixel smoothing would lag
      behind while the camera turns and cause overshoot).
    * Hysteresis hold zone: the gimbal stops once the error is small and only
      moves again when the error clearly grows, so it does not dither.
    * Friction compensation only outside the hold zone and an acceleration
      limit removes jerks.
    * During short detection dropouts it keeps steering to the last estimate.
    """

    def __init__(self, settings):
        self.cfg = settings["aiming"]
        c = self.cfg
        self.pid_yaw = PID(c["yaw_kp"], c["yaw_ki"], c["yaw_kd"], c["max_yaw_speed"])
        self.pid_pitch = PID(c["pitch_kp"], c["pitch_ki"], c["pitch_kd"], c["max_pitch_speed"])
        table = np.array(c["pitch_compensation"], dtype=float)
        self.comp_d, self.comp_deg = table[:, 0], table[:, 1]
        self.reset()

    def reset(self):
        self.pid_yaw.reset()
        self.pid_pitch.reset()
        self.estimate = None          # [yaw, pitch] absolute aim angles
        self.track_id = None
        self.hold = [False, False]
        self.hold_since = [0.0, 0.0]
        self.cmd = [0.0, 0.0]
        self.last_time = None
        self.lock_count = 0
        self.tol = self.cfg["max_lock_tolerance_deg"]
        self.last = {}

    def pitch_compensation(self, distance_m):
        if distance_m is None:
            return 0.0
        return float(np.interp(distance_m, self.comp_d, self.comp_deg))

    def _axis_speed(self, axis, error, pid, min_speed, now):
        hold_exit = self.tol
        hold_enter = self.cfg["hold_enter_ratio"] * self.tol
        if self.hold[axis]:
            # Stay still unless the error clearly grew, or a small residual error is
            # still there after the (now motionless, hence accurate) estimate settled.
            settled = now - self.hold_since[axis] >= self.cfg["refine_delay_s"]
            if abs(error) <= hold_enter or (abs(error) <= hold_exit and not settled):
                return 0.0
        elif abs(error) <= hold_enter:
            pid.reset()
            self.hold[axis] = True
            self.hold_since[axis] = now
            return 0.0
        self.hold[axis] = False
        out = pid.compute(error, now)
        if abs(out) < min_speed:
            out = math.copysign(min_speed, error)  # always enough to beat friction outside the hold zone
        return out

    def _speeds(self, gimbal_now, now):
        now_pitch, now_yaw = gimbal_now
        yaw_err = self.estimate[0] - now_yaw
        pitch_err = self.estimate[1] - now_pitch
        target = [
            self._axis_speed(0, yaw_err, self.pid_yaw, self.cfg["min_yaw_speed"], now),
            self._axis_speed(1, pitch_err, self.pid_pitch, self.cfg["min_pitch_speed"], now),
        ]
        lo, hi = self.cfg["pitch_limits_deg"]
        if (now_pitch <= lo and target[1] < 0) or (now_pitch >= hi and target[1] > 0):
            target[1] = 0.0

        dt = 0.05 if self.last_time is None else min(0.2, max(1e-3, now - self.last_time))
        self.last_time = now
        max_step = self.cfg["max_accel_dps2"] * dt
        for i in (0, 1):
            if target[i] == 0.0:
                self.cmd[i] = 0.0          # stopping is never delayed
            else:
                self.cmd[i] += max(-max_step, min(max_step, target[i] - self.cmd[i]))
        return yaw_err, pitch_err

    def update(self, track, frame_shape, focal_px, gimbal_at_capture, gimbal_now, now, gimbal_rate_dps=0.0):
        """New measurement. Return (yaw_speed, pitch_speed, locked, info)."""
        if track.track_id != self.track_id:
            self.reset()
            self.track_id = track.track_id
        h, w = frame_shape[:2]
        cx, cy = track.detection.center
        img_yaw, img_pitch = pixel_to_angles(cx, cy, w, h, focal_px)
        comp = self.pitch_compensation(track.distance_m)
        cap_pitch, cap_yaw = gimbal_at_capture
        measured = [cap_yaw + img_yaw + self.cfg["yaw_offset_deg"],
                    cap_pitch + img_pitch + self.cfg["pitch_offset_deg"] + comp]

        if self.estimate is None:
            self.estimate = measured
        else:
            a = self.cfg["target_smoothing"]
            for i in (0, 1):
                jump = measured[i] - self.estimate[i]
                self.estimate[i] = measured[i] if abs(jump) > self.cfg["reseed_deg"] else self.estimate[i] + a * jump

        radius_deg = math.degrees(math.atan2(track.detection.size_px / 2.0, focal_px))
        self.tol = min(self.cfg["max_lock_tolerance_deg"],
                       max(self.cfg["min_lock_tolerance_deg"], self.cfg["lock_tolerance_ratio"] * radius_deg))
        yaw_err, pitch_err = self._speeds(gimbal_now, now)

        steady = gimbal_rate_dps <= self.cfg["max_lock_rate_dps"]
        if abs(yaw_err) <= self.tol and abs(pitch_err) <= self.tol and not track.detection.clipped and steady:
            self.lock_count += 1
        else:
            self.lock_count = 0
        locked = self.lock_count >= self.cfg["lock_frames"]
        self.last = {"yaw_err": yaw_err, "pitch_err": pitch_err, "comp": comp, "tol": self.tol}
        return self.cmd[0], self.cmd[1], locked, self.last

    def coast(self, gimbal_now, now):
        """No measurement this frame (short dropout): keep steering to the estimate.
        The lock count is paused, not reset; firing still needs a visible target."""
        if self.estimate is None:
            return 0.0, 0.0
        self._speeds(gimbal_now, now)
        return self.cmd[0], self.cmd[1]


# ==============================================================================
# 3. Fire control
# ==============================================================================
class FireControl:
    def __init__(self, settings):
        self.cfg = settings["firing"]
        self.auto_fire = self.cfg["auto_fire"]
        self.last_shot = -1e9
        self.shots_at_track = 0
        self.track_id = None
        self.total_shots = 0

    def should_auto_fire(self, track, locked, now):
        if track is None:
            return False
        if track.track_id != self.track_id:     # new target -> new ammo budget
            self.track_id, self.shots_at_track = track.track_id, 0
        return (self.auto_fire and locked
                and self.shots_at_track < self.cfg["max_shots_per_target"]
                and now - self.last_shot >= self.cfg["cooldown_s"])

    def can_manual_fire(self, now):
        return now - self.last_shot >= min(0.25, self.cfg["cooldown_s"])

    def record_shot(self, now, auto):
        self.last_shot = now
        self.total_shots += 1
        if auto:
            self.shots_at_track += 1


class Engagement:
    """Knocks one card down, then moves on.

    A card that has been shot is watched for a moment: if its centre drops, or it
    disappears, or it ends up lying down (the detector then refuses it), it counts
    as DOWN. Its direction is remembered for a while so the gimbal does not go
    back to the empty spot or shoot a card that is already falling.
    """

    def __init__(self, settings):
        self.cfg = settings["knockdown"]
        self.reset()
        self.downed = 0

    def reset(self):
        self.track_id = None
        self.best_pitch = None        # highest (absolute) pitch seen for this card
        self.shot_at = None
        self.missing = 0
        self.last_bearing = None      # (yaw, pitch) in gimbal ground angles
        self.blacklist = []           # [(yaw, pitch, expiry_time)]

    # ---------------------------------------------------------------- blacklist
    def is_blacklisted(self, yaw, pitch, now):
        self.blacklist = [b for b in self.blacklist if b[2] > now]
        radius = self.cfg["blacklist_radius_deg"]
        return any(abs(yaw - by) <= radius and abs(pitch - bp) <= radius
                   for by, bp, _ in self.blacklist)

    def clear_blacklist(self):
        """The robot drove somewhere else, so old directions mean nothing."""
        self.blacklist = []

    # ---------------------------------------------------------------- engagement
    def update(self, track, bearing, now, visible):
        """Feed one frame. Returns True when the engaged card has just been declared down.

        ``track`` may be None: the tracker gives up on a vanished card sooner than
        this watcher does, and a card that disappears right after being shot is
        exactly the case we are looking for.
        """
        if not self.cfg["enabled"]:
            return False
        if track is not None and track.track_id != self.track_id:
            self.track_id, self.best_pitch, self.shot_at, self.missing = track.track_id, None, None, 0
        if visible and bearing is not None:
            self.last_bearing = bearing
            self.missing = 0
            self.best_pitch = bearing[1] if self.best_pitch is None else max(self.best_pitch, bearing[1])
        else:
            self.missing += 1

        if self.shot_at is None:
            return False
        if now - self.shot_at < self.cfg["settle_seconds"]:
            # Recoil shake / motion blur right after the shot: frames missed here
            # mean nothing, so they must not count as "the card is gone".
            self.missing = 0
            return False
        dropped = (visible and bearing is not None and self.best_pitch is not None
                   and self.best_pitch - bearing[1] >= self.cfg["fall_pitch_deg"])
        vanished = self.missing >= self.cfg["fall_missing_frames"]
        if dropped or vanished:
            self.mark_down(now, reason="fell over" if dropped else "gone")
            return True
        return False

    def record_shot(self, now):
        self.shot_at = now
        self.missing = 0

    def give_up(self, now):
        """The card took every shot in its budget and is still standing: skip it."""
        if self.last_bearing is not None:
            self.blacklist.append((self.last_bearing[0], self.last_bearing[1],
                                   now + self.cfg["blacklist_seconds"]))
        print("Card still standing after all shots - skipping it and looking for another.")
        self.reset_current()

    def mark_down(self, now, reason="down"):
        if self.last_bearing is not None:
            self.blacklist.append((self.last_bearing[0], self.last_bearing[1],
                                   now + self.cfg["blacklist_seconds"]))
        self.downed += 1
        print(f"Target down ({reason}) - {self.downed} total. Looking for the next card.")
        self.reset_current()

    def reset_current(self):
        self.track_id = None
        self.best_pitch = None
        self.shot_at = None
        self.missing = 0
        self.last_bearing = None


def limit_forward(chassis, front_mm, min_distance_mm):
    """Never drive into the wall: forward motion is cut (and turned into a small
    back-off) when the front ToF says we are closer than min_distance_mm.
    Backing up and strafing sideways stay available."""
    x, y, z = chassis
    if front_mm is None or front_mm <= 0 or front_mm >= min_distance_mm:
        return (x, y, z), False
    return (min(x, 0.0), y, z), True


class PatrolController:
    """Drives sideways along the wall, holding its distance with the front ToF,
    and stops while a card is being engaged."""

    def __init__(self, settings):
        self.cfg = settings["patrol"]
        self.enabled = self.cfg["enabled"]
        self.direction = 1.0
        self.leg_started = None
        self.state = "off"

    def toggle(self, now):
        self.enabled = not self.enabled
        self.leg_started = now
        self.state = "scanning" if self.enabled else "off"
        return self.enabled

    def update(self, now, front_mm, engaged):
        """Return (x, y, z) chassis speeds."""
        if not self.enabled:
            self.state = "off"
            return 0.0, 0.0, 0.0
        if engaged and self.cfg["stop_to_shoot"]:
            self.state = "holding (target)"
            return 0.0, 0.0, 0.0
        if self.leg_started is None:
            self.leg_started = now
        if now - self.leg_started >= self.cfg["leg_seconds"]:
            self.direction *= -1.0
            self.leg_started = now

        x = 0.0
        if front_mm is not None and front_mm > 0:
            if front_mm < self.cfg["min_distance_mm"] * 0.8:
                x = -self.cfg["max_approach_mps"]          # much too close: back off
                self.state = "backing off"
            elif front_mm < self.cfg["min_distance_mm"]:
                x = -self.cfg["max_approach_mps"] * 0.5     # inside the safety margin
                self.state = "keeping 1 m"
            else:
                error = front_mm - self.cfg["wall_distance_mm"]
                x = max(-self.cfg["max_approach_mps"],
                        min(self.cfg["max_approach_mps"], self.cfg["distance_kp"] * error))
                self.state = "scanning"
        else:
            self.state = "scanning (no ToF)"
        return x, self.direction * self.cfg["speed_mps"], 0.0


# ==============================================================================
# 4. Keyboard (held keys + one-shot key presses)
# ==============================================================================
HOLD_KEYS = set("wasdqeijkl")


class OpenCVKeyboard:
    """Uses cv2.waitKey. Held keys rely on OS auto-repeat, so the first press is
    held a bit longer to bridge the auto-repeat delay. Only one held key at a
    time is reliable (install pynput for W+D style combinations)."""

    name = "opencv"
    FIRST_HOLD_S, REPEAT_HOLD_S = 0.55, 0.14

    def __init__(self):
        self.expiry = {}
        self.events = []

    def feed_cv_key(self, code):
        if code is None or code < 0:
            return
        code &= 0xFF
        key = "esc" if code == 27 else "space" if code == 32 else chr(code).lower()
        now = time.monotonic()
        if key in HOLD_KEYS:
            repeating = self.expiry.get(key, 0) > now
            self.expiry[key] = now + (self.REPEAT_HOLD_S if repeating else self.FIRST_HOLD_S)
        else:
            self.events.append(key)

    def held(self, key):
        return self.expiry.get(key, 0) > time.monotonic()

    def pop_events(self):
        events, self.events = self.events, []
        return events

    def release_all(self):
        self.expiry.clear()

    def stop(self):
        pass


class PynputKeyboard:
    """True key-down/key-up state (multiple keys at once). Listens globally."""

    name = "pynput"

    def __init__(self):
        from pynput import keyboard as kb  # noqa: imported lazily (optional dependency)

        self._kb = kb
        self._lock = threading.Lock()
        self._down = set()
        self._events = []
        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        if sys.platform == "darwin" and not getattr(self.listener, "IS_TRUSTED", True):
            self.listener.stop()
            raise RuntimeError("macOS has not granted Accessibility/Input Monitoring to this terminal")

    def _name(self, key):
        if key == self._kb.Key.esc:
            return "esc"
        if key == self._kb.Key.space:
            return "space"
        char = getattr(key, "char", None)
        return char.lower() if char else None

    def _on_press(self, key):
        name = self._name(key)
        if name is None:
            return
        with self._lock:
            if name in HOLD_KEYS:
                self._down.add(name)
            elif name not in self._down:     # ignore auto-repeat of one-shot keys
                self._down.add(name)
                self._events.append(name)

    def _on_release(self, key):
        name = self._name(key)
        with self._lock:
            self._down.discard(name)

    def feed_cv_key(self, code):
        pass  # pynput already sees every key

    def held(self, key):
        with self._lock:
            return key in self._down

    def pop_events(self):
        with self._lock:
            events, self._events = self._events, []
        return events

    def release_all(self):
        with self._lock:
            self._down.clear()

    def stop(self):
        self.listener.stop()


def make_keyboard(backend):
    if backend in ("auto", "pynput"):
        try:
            return PynputKeyboard()
        except Exception as error:
            if backend == "pynput":
                raise
            print(f"[keys] pynput unavailable ({error}); using OpenCV window keys "
                  "(click the video window; one held key at a time).")
    return OpenCVKeyboard()


# ==============================================================================
# 5. Robot I/O (real robot or offline video source)
# ==============================================================================
class CommandLimiter:
    """Send only when the value changes or as a keep-alive at the command rate."""

    def __init__(self, rate_hz, keepalive_s=0.3):
        self.period = 1.0 / rate_hz
        self.keepalive = keepalive_s
        self.last_value = None
        self.last_time = -1e9

    def due(self, value, now):
        changed = value != self.last_value
        if (changed and now - self.last_time >= self.period) or now - self.last_time >= self.keepalive:
            self.last_value, self.last_time = value, now
            return True
        return False

    def force(self, value, now):
        self.last_value, self.last_time = value, now


class RobotIO:
    def __init__(self, connection, settings):
        from robomaster import blaster, robot  # imported here so offline mode needs no SDK

        self._robot_mod, self._blaster_mod = robot, blaster
        self.connection = connection
        self.settings = settings
        self.ep_robot = None
        self.angles = collections.deque(maxlen=200)  # (t, pitch, yaw) in GROUND angles
        self.distance_mm = None
        self.distance_time = 0.0
        self._distance_sub = False
        self._fire_queue = queue.Queue(maxsize=1)
        self._fire_thread = None
        self._stream = False
        self._angle_sub = False

    def start(self):
        robot, blaster = self._robot_mod, self._blaster_mod
        self.ep_robot = robot.Robot()
        print(f"Connecting to RoboMaster using '{self.connection}' mode...")
        if self.connection == "ap":
            self.ep_robot.initialize(conn_type="ap", proto_type="udp")
        else:
            self.ep_robot.initialize(conn_type=self.connection)

        mode = {"free": robot.FREE, "gimbal_lead": robot.GIMBAL_LEAD,
                "chassis_lead": robot.CHASSIS_LEAD}[self.settings["controls"]["robot_mode"]]
        self.ep_robot.set_robot_mode(mode=mode)

        self.ep_robot.camera.start_video_stream(display=False, resolution=self.settings["camera"]["resolution"])
        self._stream = True
        self.ep_robot.gimbal.sub_angle(freq=50, callback=self._on_angle)
        self._angle_sub = True
        try:
            self.ep_robot.sensor.sub_distance(freq=10, callback=self._on_distance)
            self._distance_sub = True
        except Exception as error:      # ToF is only needed for patrol mode
            print(f"[sensor] front distance unavailable: {error}")
        self.recenter()

        self.water_fire = getattr(blaster, "WATER_FIRE", "water")
        self._fire_thread = threading.Thread(target=self._fire_worker, daemon=True)
        self._fire_thread.start()
        print("Robot ready.")

    def _on_angle(self, info):
        # (pitch, yaw, pitch_ground, yaw_ground): the ground angles do not jump when
        # the chassis turns, so the target estimate survives driving.
        pitch, yaw = (info[2], info[3]) if len(info) >= 4 else (info[0], info[1])
        self.angles.append((time.monotonic(), float(pitch), float(yaw)))

    def _on_distance(self, info):
        if isinstance(info, (list, tuple)) and info:
            self.distance_mm = float(info[0])
            self.distance_time = time.monotonic()

    def front_distance_mm(self):
        """Front ToF reading, or None when it is missing or stale."""
        if self.distance_mm is None or time.monotonic() - self.distance_time > 1.0:
            return None
        return self.distance_mm

    def gimbal_angle(self, at_time=None):
        """(pitch, yaw) now, or at a past monotonic time (latency compensation)."""
        if not self.angles:
            return 0.0, 0.0
        if at_time is None:
            _, p, y = self.angles[-1]
            return p, y
        for t, p, y in reversed(self.angles):
            if t <= at_time:
                return p, y
        _, p, y = self.angles[0]
        return p, y

    def gimbal_rate(self, window_s=0.15):
        """Measured gimbal speed (deg/s) over the last window, for 'is it steady?' checks."""
        if len(self.angles) < 2:
            return 0.0
        t1, p1, y1 = self.angles[-1]
        p0, y0 = self.gimbal_angle(t1 - window_s)
        return max(abs(y1 - y0), abs(p1 - p0)) / window_s

    def read_frame(self):
        try:
            return self.ep_robot.camera.read_cv2_image(strategy="newest", timeout=0.5)
        except Exception:
            return None  # queue.Empty when no frame arrived in time

    def drive_chassis(self, x, y, z):
        # timeout: the robot stops by itself if this program freezes or crashes.
        self.ep_robot.chassis.drive_speed(x=x, y=y, z=z, timeout=0.5)

    def drive_gimbal(self, yaw_speed, pitch_speed):
        self.ep_robot.gimbal.drive_speed(pitch_speed=pitch_speed, yaw_speed=yaw_speed)

    def recenter(self):
        self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        try:
            self.ep_robot.gimbal.recenter(pitch_speed=150, yaw_speed=150).wait_for_completed(timeout=4)
        except Exception as error:
            print(f"[gimbal] recenter failed: {error}")

    def fire(self, times):
        """Non-blocking: returns False if a shot is still being sent."""
        try:
            self._fire_queue.put_nowait(times)
            return True
        except queue.Full:
            return False

    def _fire_worker(self):
        while True:
            times = self._fire_queue.get()
            if times is None:
                return
            try:
                self.ep_robot.blaster.fire(fire_type=self.water_fire, times=times)
            except Exception as error:
                print(f"[blaster] fire failed: {error}")

    def close(self):
        if self.ep_robot is None:
            return
        steps = [
            lambda: self.ep_robot.chassis.drive_speed(x=0, y=0, z=0),
            lambda: self.ep_robot.gimbal.drive_speed(pitch_speed=0, yaw_speed=0),
            lambda: self._fire_queue.put_nowait(None),
            lambda: self._angle_sub and self.ep_robot.gimbal.unsub_angle(),
            lambda: self._distance_sub and self.ep_robot.sensor.unsub_distance(),
            lambda: self._stream and self.ep_robot.camera.stop_video_stream(),
            lambda: self.ep_robot.close(),
        ]
        for step in steps:
            try:
                step()
            except Exception as error:
                print(f"[shutdown] {error}")


class OfflineIO:
    """Webcam/video/image source for testing; robot commands are printed."""

    def __init__(self, source, verbose=True):
        self.source = source
        self.verbose = verbose
        self.image = None
        self.capture = None
        self.last_print = {}

    def start(self):
        if not self.source.isdigit():
            self.image = cv2.imread(self.source)
        if self.image is None:
            self.capture = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
            if not self.capture.isOpened():
                raise RuntimeError(f"cannot open source '{self.source}'")
        print(f"Offline mode: reading '{self.source}' (no robot commands are sent).")

    def read_frame(self):
        if self.image is not None:
            return self.image.copy()
        ok, frame = self.capture.read()
        return frame if ok else None

    def finished(self):
        return self.capture is not None and not self.capture.isOpened()

    def gimbal_angle(self, at_time=None):
        return 0.0, 0.0

    def gimbal_rate(self, window_s=0.15):
        return 0.0

    def front_distance_mm(self):
        return None

    def _log(self, name, text):
        if self.verbose and self.last_print.get(name) != text:
            self.last_print[name] = text
            print(f"[sim] {name}: {text}")

    def drive_chassis(self, x, y, z):
        self._log("chassis", f"x={x:+.2f} y={y:+.2f} z={z:+.0f}")

    def drive_gimbal(self, yaw_speed, pitch_speed):
        self._log("gimbal", f"yaw={yaw_speed:+.1f} pitch={pitch_speed:+.1f}")

    def recenter(self):
        self._log("gimbal", "recenter")

    def fire(self, times):
        print(f"[sim] FIRE water x{times}")
        return True

    def close(self):
        if self.capture is not None:
            self.capture.release()


# ==============================================================================
# 6. Main loop
# ==============================================================================
def draw_hud(frame, st):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (430, 238), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    on = lambda flag: "ON" if flag else "off"  # noqa: E731
    lines = [
        (f"TARGET: {st['color']} {st['shape']}  {st['size_m']:.2f} m  "
         f"range {st['min_d']:.1f}-{st['max_d']:.1f} m", (255, 255, 255)),
        (f"AUTO-TRACK {on(st['auto_track'])}   AUTO-FIRE {on(st['auto_fire'])}",
         (0, 80, 255) if st["auto_fire"] else (255, 255, 255)),
        (f"STATUS: {st['status']}", st["status_color"]),
        (f"distance: {st['distance']}", (255, 255, 255)),
        (f"err yaw {st['yaw_err']}  pitch {st['pitch_err']}  tol {st['tol']}", (255, 255, 255)),
        (f"pitch comp {st['comp']}  trim yaw {st['yaw_trim']:+.1f} pitch {st['pitch_trim']:+.1f}",
         (255, 255, 255)),
        (f"shots: {st['shots']}   cards down: {st['downed']}   patrol: {st['patrol']}", (200, 200, 200)),
        (f"fps: {st['fps']:.0f}   keys: {st['keys']}   front: {st['front']} (min {st['min_front']:.0f} mm)",
         (0, 80, 255) if st["too_close"] else (200, 200, 200)),
        (f"chassis x{st['cx']:+.2f} y{st['cy']:+.2f} z{st['cz']:+.0f}  gimbal y{st['gy']:+.0f} p{st['gp']:+.0f}",
         (200, 200, 200)),
    ]
    for i, (text, color) in enumerate(lines):
        _text(frame, text, (18, 34 + i * 26), color, 0.55, 1)

    help_text = ("WASD move  QE rotate  IJKL gimbal  SPACE fire  T track  F auto-fire  G patrol  "
                 "X/Z target  N card size  CLICK learn colour  M mask  ESC quit")
    _text(frame, help_text, (12, h - 14), (220, 220, 220), 0.5, 1)

    if st["fire_flash"]:
        _text(frame, "FIRE!", (w // 2 - 70, 90), (0, 0, 255), 1.6, 4)


def parse_args():
    parser = argparse.ArgumentParser(description="WASD driving + auto-aim water ball shooter.")
    parser.add_argument("--connection", choices=("ap", "sta", "rndis"),
                        help="RoboMaster connection (default: robot.connection_type in settings.yaml)")
    parser.add_argument("--source", help="offline test source: webcam index, video or image path")
    parser.add_argument("--color", help="target colour (overrides config)")
    parser.add_argument("--shape", choices=TARGET_SHAPES, help="target shape (overrides config)")
    parser.add_argument("--size", type=float, help="real target size in metres (overrides config)")
    parser.add_argument("--auto-fire", action="store_true", help="start with auto-fire armed")
    parser.add_argument("--patrol", action="store_true", help="start patrolling along the wall")
    parser.add_argument("--headless", action="store_true", help="no window/keyboard (offline tests)")
    parser.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    parser.add_argument("--save", help="save annotated output (.png = last frame, .mp4/.avi = video)")
    return parser.parse_args()


def apply_overrides(config, args):
    ts = config.setdefault("target_shooting", {}) or {}
    config["target_shooting"] = ts
    target = ts.setdefault("target", {}) or {}
    ts["target"] = target
    if args.color:
        target["color"] = args.color
    if args.shape:
        target["shape"] = args.shape
    if args.size is not None:
        target["size_m"] = args.size
    if args.auto_fire:
        firing = ts.setdefault("firing", {}) or {}
        ts["firing"] = firing
        firing["auto_fire"] = True
    if args.patrol:
        patrol_cfg = ts.setdefault("patrol", {}) or {}
        ts["patrol"] = patrol_cfg
        patrol_cfg["enabled"] = True


def run(args):
    config = load_config()
    apply_overrides(config, args)
    settings = build_settings(config)
    if args.headless and not args.source:
        raise ValueError("--headless is only for offline --source tests")

    detector = ColorShapeDetector(settings)
    tracker = TargetTracker(settings)
    aim = AimController(settings)
    fire_ctl = FireControl(settings)
    engagement = Engagement(settings)
    patrol = PatrolController(settings)
    ctl = settings["controls"]
    tgt = settings["target"]
    colors = list(settings["detection"]["colors"])
    shapes = list(TARGET_SHAPES)
    auto_track = ctl["auto_track"]

    connection = args.connection or config.get("robot", {}).get("connection_type", "ap")
    io = OfflineIO(args.source) if args.source else RobotIO(connection, settings)
    keyboard = None if args.headless else make_keyboard(ctl["keyboard_backend"])
    chassis_limiter = CommandLimiter(ctl["command_rate_hz"])
    gimbal_limiter = CommandLimiter(ctl["command_rate_hz"])
    writer = None
    last_frame = None
    window = "RoboMaster Gimbal Shooter"
    clicks = []
    show_mask = False
    if keyboard:
        print(f"[keys] backend: {keyboard.name}")
    if not args.headless:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window, lambda event, x, y, *_: clicks.append((x, y))
                             if event == cv2.EVENT_LBUTTONDOWN else None)
        print("Tip: click the middle of a target card to learn its colour under this light.")

    frames = processed = 0
    fps, fps_time = 0.0, time.monotonic()
    fire_flash_until = 0.0
    last_close_warning = -1e9
    gimbal_cmd = (0.0, 0.0)
    chassis_cmd = (0.0, 0.0, 0.0)

    def stop_gimbal(now):
        nonlocal gimbal_cmd
        gimbal_cmd = (0.0, 0.0)
        io.drive_gimbal(0.0, 0.0)
        gimbal_limiter.force(gimbal_cmd, now)

    try:
        io.start()
        while True:
            frame = io.read_frame()
            now = time.monotonic()
            if frame is None:
                # Never leave the gimbal spinning on stale data.
                if gimbal_cmd != (0.0, 0.0):
                    stop_gimbal(now)
                if isinstance(io, OfflineIO) and io.image is None:
                    print("End of video source.")
                    break
                if keyboard:
                    keyboard.feed_cv_key(cv2.waitKey(1))
                    if "esc" in keyboard.pop_events():
                        break
                continue
            frames += 1
            capture_time = now - settings["aiming"]["camera_latency_s"]
            focal_px = detector.focal_length(frame.shape[1])

            # ---- detection + tracking
            detections, masks = detector.detect(frame)

            # Absolute direction of each candidate, so cards already knocked down
            # are not engaged again while the robot stands still.
            cap_pitch, cap_yaw = io.gimbal_angle(capture_time)
            def bearing_of(detection):
                d_yaw, d_pitch = pixel_to_angles(detection.center[0], detection.center[1],
                                                 frame.shape[1], frame.shape[0], focal_px)
                return cap_yaw + d_yaw, cap_pitch + d_pitch
            for detection in detections:
                if detection.is_target and engagement.is_blacklisted(*bearing_of(detection), now=now):
                    detection.is_target = False
                    detection.reject_reason = "already down"

            track = tracker.update(detections, frame.shape)
            visible = tracker.visible

            # ---- keyboard one-shot events
            fire_request = False
            if keyboard:
                for key in keyboard.pop_events():
                    if key == "esc":
                        raise KeyboardInterrupt
                    if key == "space":
                        fire_request = True
                    elif key == "t":
                        auto_track = not auto_track
                        aim.reset()
                        print(f"Auto-track {'ON' if auto_track else 'OFF'}")
                    elif key == "f":
                        fire_ctl.auto_fire = not fire_ctl.auto_fire
                        print(f"Auto-fire {'ARMED' if fire_ctl.auto_fire else 'off'}")
                    elif key in ("x", "z"):
                        if key == "x":
                            tgt["color"] = colors[(colors.index(tgt["color"]) + 1) % len(colors)]
                        else:
                            tgt["shape"] = shapes[(shapes.index(tgt["shape"]) + 1) % len(shapes)]
                        tracker.reset()
                        aim.reset()
                        print(f"Target -> {tgt['color']} {tgt['shape']}")
                    elif key == "c":
                        if visible:
                            d_cal = settings["camera"]["calibration_distance_m"]
                            new_f = d_cal * track.detection.size_px / tgt["size_m"]
                            detector.set_focal_length(new_f)
                            print(f"Calibrated focal_length_px = {new_f:.1f} (target at {d_cal} m). "
                                  "Put it in config target_shooting.camera.focal_length_px")
                        else:
                            print("Calibration needs a visible confirmed target at calibration_distance_m.")
                    elif key == "r":
                        stop_gimbal(now)
                        aim.reset()
                        io.recenter()
                    elif key in ("[", "]"):
                        settings["aiming"]["pitch_offset_deg"] += 0.2 if key == "]" else -0.2
                        print(f"pitch_offset_deg = {settings['aiming']['pitch_offset_deg']:+.1f}")
                    elif key in (",", "."):
                        settings["aiming"]["yaw_offset_deg"] += 0.2 if key == "." else -0.2
                        print(f"yaw_offset_deg = {settings['aiming']['yaw_offset_deg']:+.1f}")
                    elif key == "g":
                        running = patrol.toggle(now)
                        print(f"Patrol {'ON - driving along the wall' if running else 'OFF'}")
                    elif key == "n":
                        if visible and track.detection.size_px > 1:
                            d_cal = settings["camera"]["calibration_distance_m"]
                            measured = d_cal * track.detection.size_px / focal_px
                            tgt["size_m"] = round(measured, 3)
                            tracker.reset()
                            aim.reset()
                            print(f"Card size measured at {d_cal} m: size_m = {tgt['size_m']} "
                                  "(put it in config target_shooting.target.size_m)")
                        else:
                            print(f"Point at one card from {settings['camera']['calibration_distance_m']} m first.")
                    elif key == "m":
                        show_mask = not show_mask
                    elif key == "p":
                        capture_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                   "data", "captures")
                        os.makedirs(capture_dir, exist_ok=True)
                        path = os.path.join(capture_dir, datetime.now().strftime("frame_%Y%m%d_%H%M%S_%f.png"))
                        cv2.imwrite(path, frame)   # raw frame (nothing drawn yet)
                        print(f"Saved raw frame {path}")

            # ---- click-to-learn colour (uses the raw frame, before anything is drawn)
            while clicks:
                cx_click, cy_click = clicks.pop(0)
                try:
                    ranges = detector.learn_color(frame, cx_click, cy_click, tgt["color"])
                    tracker.reset()
                    aim.reset()
                    print(f"Learned '{tgt['color']}' at ({cx_click},{cy_click}). To keep it, put this in "
                          "config/settings.yaml under target_shooting.detection.colors:")
                    print(f"      {tgt['color']}:")
                    for r in ranges:
                        print(f"        - {{lower: {r['lower']}, upper: {r['upper']}}}")
                except ValueError as error:
                    print(f"Colour not learned: {error}")

            held = keyboard.held if keyboard else (lambda k: False)

            # ---- chassis (WASD + QE, or patrol when nothing is held)
            v, rot = ctl["chassis_speed_mps"], ctl["chassis_rotate_dps"]
            new_chassis = ((held("w") - held("s")) * v, (held("d") - held("a")) * v,
                           (held("e") - held("q")) * rot)
            if new_chassis == (0.0, 0.0, 0.0) and patrol.enabled:
                engaged = visible and track is not None and track.confirmed
                new_chassis = patrol.update(now, io.front_distance_mm(), engaged)
            new_chassis, too_close = limit_forward(new_chassis, io.front_distance_mm(),
                                                   settings["patrol"]["min_distance_mm"])
            if too_close and now - last_close_warning > 2.0:
                last_close_warning = now
                print(f"Too close to the wall ({io.front_distance_mm():.0f} mm) - forward blocked.")
            if new_chassis != (0.0, 0.0, 0.0):
                # The robot moved, so remembered directions of downed cards are stale.
                engagement.clear_blacklist()
            if chassis_limiter.due(new_chassis, now):
                chassis_cmd = new_chassis
                io.drive_chassis(*chassis_cmd)

            # ---- gimbal
            manual_yaw = (held("l") - held("j")) * ctl["gimbal_speed_dps"]
            manual_pitch = (held("i") - held("k")) * ctl["gimbal_speed_dps"]
            locked = False
            info = {}
            if manual_yaw or manual_pitch:
                new_gimbal = (float(manual_yaw), float(manual_pitch))
                aim.reset()
            elif auto_track and visible:
                yaw_s, pitch_s, locked, info = aim.update(
                    track, frame.shape, focal_px, io.gimbal_angle(capture_time), io.gimbal_angle(), now,
                    io.gimbal_rate())
                new_gimbal = (round(yaw_s, 1), round(pitch_s, 1))
            elif auto_track and track is not None and track.confirmed:
                # Short dropout: keep turning towards the last known target position.
                yaw_s, pitch_s = aim.coast(io.gimbal_angle(), now)
                new_gimbal = (round(yaw_s, 1), round(pitch_s, 1))
                info = aim.last
            else:
                new_gimbal = (0.0, 0.0)
                aim.reset()
            if new_gimbal == (0.0, 0.0) and gimbal_cmd != (0.0, 0.0):
                stop_gimbal(now)               # stops are never rate-limited
            elif gimbal_limiter.due(new_gimbal, now):
                gimbal_cmd = new_gimbal
                io.drive_gimbal(*gimbal_cmd)

            # ---- firing
            auto_shot = auto_track and visible and fire_ctl.should_auto_fire(track, locked, now)
            if (auto_shot or (fire_request and fire_ctl.can_manual_fire(now))) and \
                    io.fire(settings["firing"]["fire_times"]):
                fire_ctl.record_shot(now, auto=auto_shot)
                fire_flash_until = now + 0.3
                engagement.record_shot(now)
                dist = f"{track.distance_m:.2f} m" if visible and track.distance_m else "-"
                print(f"FIRE ({'auto' if auto_shot else 'manual'}) #{fire_ctl.total_shots} "
                      f"target={tgt['color']} {tgt['shape']} distance={dist}")

            # ---- did the card fall over? then stop shooting it and pick the next one
            bearing = bearing_of(track.detection) if (visible and track is not None) else None
            done = engagement.update(track, bearing, now, visible and track is not None)
            if (not done and settings["knockdown"]["skip_after_max_shots"] and visible
                    and fire_ctl.auto_fire and track is not None
                    and track.track_id == fire_ctl.track_id
                    and fire_ctl.shots_at_track >= settings["firing"]["max_shots_per_target"]
                    and now - fire_ctl.last_shot >= settings["knockdown"]["settle_seconds"]):
                engagement.give_up(now)
                done = True
            if done:
                tracker.reset()
                aim.reset()
                track, visible, locked = None, False, False
                stop_gimbal(now)

            # ---- draw
            if track is None:
                status, status_color = "SEARCHING", (255, 255, 255)
            elif not track.confirmed:
                status = f"ACQUIRING {track.hits}/{settings['tracking']['confirm_frames']}"
                status_color = (0, 200, 255)
            elif not visible:
                status, status_color = f"LOST ({track.misses})", (0, 200, 255)
            elif locked:
                status, status_color = "LOCKED", (0, 0, 255)
            else:
                status, status_color = "TRACKING", (0, 255, 0)

            draw_detections(frame, detections, track)
            if visible:
                # Where the barrel points when locked (aim point incl. compensation).
                comp = aim.pitch_compensation(track.distance_m)
                a = settings["aiming"]
                h, w = frame.shape[:2]
                px = w / 2 - focal_px * math.tan(math.radians(a["yaw_offset_deg"]))
                py = h / 2 + focal_px * math.tan(math.radians(a["pitch_offset_deg"] + comp))
                cv2.drawMarker(frame, (int(px), int(py)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)

            if now - fps_time >= 0.5:
                fps = frames / (now - fps_time) if frames else 0.0
                frames, fps_time = 0, now
            fmt = lambda k: f"{info[k]:+.1f}deg" if k in info else "-"  # noqa: E731
            draw_hud(frame, {
                "color": tgt["color"], "shape": tgt["shape"], "size_m": tgt["size_m"],
                "min_d": tgt["min_distance_m"], "max_d": tgt["max_distance_m"],
                "auto_track": auto_track, "auto_fire": fire_ctl.auto_fire,
                "status": status, "status_color": status_color,
                "distance": f"{track.distance_m:.2f} m" if track is not None and track.distance_m else "-",
                "yaw_err": fmt("yaw_err"), "pitch_err": fmt("pitch_err"), "tol": fmt("tol"),
                "comp": fmt("comp"), "yaw_trim": settings["aiming"]["yaw_offset_deg"],
                "pitch_trim": settings["aiming"]["pitch_offset_deg"],
                "shots": fire_ctl.total_shots, "fps": fps,
                "downed": engagement.downed, "patrol": patrol.state,
                "front": (f"{io.front_distance_mm():.0f} mm" if io.front_distance_mm() else "-"),
                "min_front": settings["patrol"]["min_distance_mm"], "too_close": too_close,
                "keys": keyboard.name if keyboard else "none",
                "cx": chassis_cmd[0], "cy": chassis_cmd[1], "cz": chassis_cmd[2],
                "gy": gimbal_cmd[0], "gp": gimbal_cmd[1],
                "fire_flash": now < fire_flash_until,
            })
            if show_mask and tgt["color"] in masks:
                h, w = frame.shape[:2]
                thumb = cv2.resize(masks[tgt["color"]], (w // 4, h // 4), interpolation=cv2.INTER_NEAREST)
                thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(thumb, (0, 0), (thumb.shape[1] - 1, thumb.shape[0] - 1), (0, 255, 255), 2)
                frame[10:10 + thumb.shape[0], w - thumb.shape[1] - 10:w - 10] = thumb
                _text(frame, f"mask: {tgt['color']}", (w - thumb.shape[1], 32 + thumb.shape[0]), (0, 255, 255), 0.5, 1)
            last_frame = frame

            if args.save and args.save.lower().endswith((".mp4", ".avi")):
                if writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if args.save.lower().endswith(".mp4") else "XVID"))
                    writer = cv2.VideoWriter(args.save, fourcc, 30, (frame.shape[1], frame.shape[0]))
                writer.write(frame)

            if not args.headless:
                cv2.imshow(window, frame)
                keyboard.feed_cv_key(cv2.waitKey(1))
            processed += 1
            if args.max_frames and processed >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        if keyboard:
            keyboard.release_all()
            keyboard.stop()
        io.close()
        if writer is not None:
            writer.release()
        if args.save and last_frame is not None and not args.save.lower().endswith((".mp4", ".avi")):
            cv2.imwrite(args.save, last_frame)
        if args.save:
            print(f"Saved {os.path.abspath(args.save)}")
        if not args.headless:
            cv2.destroyAllWindows()
        print(f"Total shots: {fire_ctl.total_shots}")


def main():
    args = parse_args()
    try:
        run(args)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    except Exception as error:
        print(f"Gimbal shooter error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
