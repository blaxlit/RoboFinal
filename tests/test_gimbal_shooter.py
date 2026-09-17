"""Offline tests for src/gimbal_shooter.py (no robot or robomaster SDK needed).

The closed-loop test simulates a gimbal with motor lag, friction and a delayed
camera, renders synthetic frames and runs the real detector, tracker, aim
controller and fire control against it.
"""

import collections
import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gimbal_shooter as gs  # noqa: E402
from config_loader import load_config  # noqa: E402
from target_detection import ColorShapeDetector, TargetTracker  # noqa: E402
from test_target_detection import H, W, background, draw_shape  # noqa: E402


def settings_with(**firing):
    config = load_config()
    config["target_shooting"].setdefault("firing", {}).update(firing)
    return gs.build_settings(config)


class SimGimbal:
    """First-order motor lag + static friction + delayed camera."""

    def __init__(self, yaw=0.0, pitch=0.0, tau=0.06, friction_dps=3.0, latency_s=0.10):
        self.yaw, self.pitch = yaw, pitch
        self.v_yaw = self.v_pitch = 0.0
        self.tau, self.friction, self.latency = tau, friction_dps, latency_s
        self.history = collections.deque(maxlen=400)

    def step(self, cmd_yaw, cmd_pitch, t, dt):
        k = dt / (self.tau + dt)
        self.v_yaw += k * ((cmd_yaw if abs(cmd_yaw) > self.friction else 0.0) - self.v_yaw)
        self.v_pitch += k * ((cmd_pitch if abs(cmd_pitch) > self.friction else 0.0) - self.v_pitch)
        self.yaw += self.v_yaw * dt
        self.pitch = max(-20.0, min(30.0, self.pitch + self.v_pitch * dt))
        self.history.append((t, self.pitch, self.yaw))

    def angle_at(self, t):
        for ht, p, y in reversed(self.history):
            if ht <= t:
                return p, y
        return (self.history[0][1], self.history[0][2]) if self.history else (self.pitch, self.yaw)


def run_closed_loop(target_yaw, target_pitch, distance, seconds=3.0, auto_fire=True, extra=None):
    settings = settings_with(auto_fire=auto_fire)
    detector = ColorShapeDetector(settings)
    tracker = TargetTracker(settings)
    aim = gs.AimController(settings)
    fire = gs.FireControl(settings)
    focal = detector.focal_length(W)
    size_px = focal * settings["target"]["size_m"] / distance
    gimbal = SimGimbal()
    gimbal.history.append((0.0, 0.0, 0.0))
    dt, t = 1.0 / 30.0, 0.0
    log = []
    cmd = (0.0, 0.0)
    shots = []
    while t < seconds:
        cap_p, cap_y = gimbal.angle_at(t - gimbal.latency)
        img = background(seed=int(t * 30) % 7)
        rel_yaw, rel_pitch = target_yaw - cap_y, target_pitch - cap_p
        if abs(rel_yaw) < 60 and abs(rel_pitch) < 40:
            cx = W / 2 + focal * math.tan(math.radians(rel_yaw))
            cy = H / 2 - focal * math.tan(math.radians(rel_pitch))
            draw_shape(img, "circle", "red", (cx, cy), size_px)
        if extra:
            extra(img)

        detections, _ = detector.detect(img)
        track = tracker.update(detections, img.shape)
        locked = False
        if tracker.visible:
            yaw_s, pitch_s, locked, info = aim.update(
                track, img.shape, focal, (cap_p, cap_y), (gimbal.pitch, gimbal.yaw), t)
            cmd = (yaw_s, pitch_s)
            if fire.should_auto_fire(track, locked, t):
                fire.record_shot(t, auto=True)
                shots.append((t, gimbal.yaw, gimbal.pitch, track.distance_m))
        else:
            aim.reset()
            cmd = (0.0, 0.0)
        gimbal.step(cmd[0], cmd[1], t, dt)
        log.append((t, gimbal.yaw, gimbal.pitch, locked))
        t += dt
    return settings, aim, log, shots


class ClosedLoopAimTests(unittest.TestCase):
    def check_run(self, target_yaw, target_pitch, distance):
        settings, aim, log, shots = run_closed_loop(target_yaw, target_pitch, distance)
        comp = aim.pitch_compensation(distance)
        want_pitch = target_pitch + comp
        final_t, final_yaw, final_pitch, _ = log[-1]

        self.assertAlmostEqual(final_yaw, target_yaw, delta=1.0)
        self.assertAlmostEqual(final_pitch, want_pitch, delta=1.0)
        # Overshoot limited (no wild swinging past the target).
        if abs(target_yaw) > 3:
            overshoot = max((y - target_yaw) * math.copysign(1, target_yaw) for _, y, _, _ in log)
            self.assertLess(overshoot, 3.0)
        # Settles and fires; every shot is aimed within the lock tolerance.
        self.assertTrue(shots, "never fired")
        self.assertLessEqual(len(shots), settings["firing"]["max_shots_per_target"])
        self.assertLess(shots[0][0], 2.0, "took too long to lock")
        for _, yaw, pitch, dist in shots:
            self.assertAlmostEqual(yaw, target_yaw, delta=settings["aiming"]["max_lock_tolerance_deg"] + 0.5)
            self.assertAlmostEqual(pitch, want_pitch, delta=settings["aiming"]["max_lock_tolerance_deg"] + 0.5)
            self.assertAlmostEqual(dist, distance, delta=distance * 0.1)
        # Shots respect the cooldown.
        gaps = [b[0] - a[0] for a, b in zip(shots, shots[1:])]
        self.assertTrue(all(g >= settings["firing"]["cooldown_s"] - 1e-6 for g in gaps))
        return shots

    def test_target_right_and_low(self):
        self.check_run(25.0, -6.0, 1.5)

    def test_target_left_and_high_close(self):
        self.check_run(-18.0, 8.0, 0.8)

    def test_target_far_small(self):
        self.check_run(10.0, 0.0, 2.8)

    def test_no_fire_when_disarmed(self):
        _, _, _, shots = run_closed_loop(10.0, 0.0, 1.5, seconds=2.0, auto_fire=False)
        self.assertEqual(shots, [])

    def test_no_fire_on_wrong_color_or_shape(self):
        def decoys(img):
            draw_shape(img, "circle", "blue", (300, 360), 120)
            draw_shape(img, "square", "red", (1000, 360), 120)
        # Target placed out of view: only decoys are visible.
        _, _, log, shots = run_closed_loop(170.0, 0.0, 1.5, seconds=2.0, extra=decoys)
        self.assertEqual(shots, [])
        self.assertAlmostEqual(log[-1][1], 0.0, delta=0.01)   # gimbal never moved

    def test_no_fire_out_of_range(self):
        _, _, _, shots = run_closed_loop(5.0, 0.0, 4.5, seconds=2.0)
        self.assertEqual(shots, [])


def run_realistic(true_latency, fps=22, noise_px=1.5, seconds=5.0, target=(20.0, -4.0), distance=1.5,
                  move_dps=0.0, dropout_every=0, seed=0):
    """Real-world conditions: camera delay that differs from the configured one,
    low frame rate, pixel noise, motor lag + friction, the same control flow as
    gimbal_shooter.run() (update / coast / reset)."""
    settings = settings_with()
    detector, tracker, aim = ColorShapeDetector(settings), TargetTracker(settings), gs.AimController(settings)
    focal = detector.focal_length(W)
    size_px = focal * settings["target"]["size_m"] / distance
    gimbal = SimGimbal(latency_s=true_latency)
    gimbal.history.append((0.0, 0.0, 0.0))
    rng = np.random.default_rng(seed)
    configured_latency = settings["aiming"]["camera_latency_s"]
    dt, t, n = 1.0 / fps, 0.0, 0
    log = []
    while t < seconds:
        target_yaw = target[0] + move_dps * t
        cap_p, cap_y = gimbal.angle_at(t - true_latency)
        img = background(seed=n % 7)
        n += 1
        if not (dropout_every and n % dropout_every == 0):
            cx = W / 2 + focal * math.tan(math.radians(target_yaw - cap_y)) + rng.normal(0, noise_px)
            cy = H / 2 - focal * math.tan(math.radians(target[1] - cap_p)) + rng.normal(0, noise_px)
            draw_shape(img, "circle", "red", (cx, cy), size_px)
        track = tracker.update(detector.detect(img)[0], img.shape)
        locked = False
        if tracker.visible:
            p0, y0 = gimbal.angle_at(t - 0.15)
            rate = max(abs(gimbal.yaw - y0), abs(gimbal.pitch - p0)) / 0.15
            yaw_s, pitch_s, locked, _ = aim.update(track, img.shape, focal, gimbal.angle_at(t - configured_latency),
                                                   (gimbal.pitch, gimbal.yaw), t, rate)
        elif track is not None and track.confirmed:
            yaw_s, pitch_s = aim.coast((gimbal.pitch, gimbal.yaw), t)
        else:
            aim.reset()
            yaw_s = pitch_s = 0.0
        for k in range(4):  # the command stays active until the next frame
            gimbal.step(yaw_s, pitch_s, t + k * dt / 4, dt / 4)
        log.append((t, gimbal.yaw, gimbal.pitch, yaw_s, pitch_s, locked, target_yaw, tracker.visible))
        t += dt
    want_pitch = target[1] + aim.pitch_compensation(distance)
    return log, want_pitch


class AntiShakeTests(unittest.TestCase):
    """The gimbal must come to rest on the target (no shaking), even when the real
    Wi-Fi video delay is not what the config says."""

    def assert_steady(self, log, want_pitch, target_yaw, label):
        tail = [r for r in log if r[0] >= log[-1][0] - 2.0]
        yaw = np.array([r[1] for r in tail])
        pitch = np.array([r[2] for r in tail])
        yaw_cmd = np.array([r[3] for r in tail])
        reversals = int(np.sum(yaw_cmd[1:] * yaw_cmd[:-1] < 0))
        self.assertLess(np.ptp(yaw), 0.3, f"{label}: yaw shakes {np.ptp(yaw):.2f} deg")
        self.assertLess(np.ptp(pitch), 0.3, f"{label}: pitch shakes {np.ptp(pitch):.2f} deg")
        self.assertEqual(reversals, 0, f"{label}: command reverses direction")
        self.assertLess(abs(yaw.mean() - target_yaw), 0.6, f"{label}: yaw off by {yaw.mean() - target_yaw:+.2f}")
        self.assertLess(abs(pitch.mean() - want_pitch), 0.6, f"{label}: pitch off by {pitch.mean() - want_pitch:+.2f}")
        seen = [r[5] for r in tail if r[7]]   # lock is only judged on frames where the target was seen
        self.assertGreater(np.mean(seen), 0.8, f"{label}: not locked")

    def test_steady_for_different_real_latencies(self):
        for latency in (0.10, 0.20, 0.30, 0.40):
            with self.subTest(latency=latency):
                log, want_pitch = run_realistic(latency)
                self.assert_steady(log, want_pitch, 20.0, f"latency {latency}s")

    def test_steady_at_low_fps_and_noisy_detection(self):
        log, want_pitch = run_realistic(0.3, fps=12, noise_px=3.0, seconds=6.0, target=(-30.0, 6.0), distance=0.8)
        self.assert_steady(log, want_pitch, -30.0, "12 fps, 3 px noise")

    def test_detection_dropouts_do_not_shake(self):
        log, want_pitch = run_realistic(0.25, dropout_every=2)
        self.assert_steady(log, want_pitch, 20.0, "every 2nd frame missed")

    def test_moving_target_followed_without_oscillation(self):
        log, _ = run_realistic(0.25, move_dps=5.0, seconds=6.0)
        tail = [r for r in log if r[0] >= 3.0]
        errors = np.array([r[1] - r[6] for r in tail])
        yaw_cmd = np.array([r[3] for r in tail])
        self.assertLess(np.abs(errors).max(), 3.0)
        self.assertEqual(int(np.sum(yaw_cmd[1:] * yaw_cmd[:-1] < 0)), 0)


class UnitTests(unittest.TestCase):
    def test_fire_control_rules(self):
        s = settings_with(auto_fire=True, cooldown_s=0.5, max_shots_per_target=2)
        fire = gs.FireControl(s)
        track = type("T", (), {"track_id": 1})()
        self.assertFalse(fire.should_auto_fire(None, True, 0.0))
        self.assertFalse(fire.should_auto_fire(track, False, 0.0))
        self.assertTrue(fire.should_auto_fire(track, True, 0.0))
        fire.record_shot(0.0, auto=True)
        self.assertFalse(fire.should_auto_fire(track, True, 0.3))      # cooldown
        self.assertTrue(fire.should_auto_fire(track, True, 0.6))
        fire.record_shot(0.6, auto=True)
        self.assertFalse(fire.should_auto_fire(track, True, 5.0))      # ammo budget used
        new_track = type("T", (), {"track_id": 2})()
        self.assertTrue(fire.should_auto_fire(new_track, True, 5.0))   # new target resets budget

    def test_command_limiter(self):
        lim = gs.CommandLimiter(rate_hz=20, keepalive_s=0.3)
        self.assertTrue(lim.due((1, 0), 0.0))
        self.assertFalse(lim.due((1, 0), 0.1))          # unchanged
        self.assertFalse(lim.due((2, 0), 0.02))         # changed but too soon
        self.assertTrue(lim.due((2, 0), 0.06))
        self.assertTrue(lim.due((2, 0), 0.40))          # keep-alive

    def test_pitch_compensation_interpolates_and_clamps(self):
        aim = gs.AimController(settings_with())
        self.assertAlmostEqual(aim.pitch_compensation(0.1), 5.0)
        self.assertAlmostEqual(aim.pitch_compensation(1.5), 2.5)
        self.assertAlmostEqual(aim.pitch_compensation(10), 2.2)
        self.assertEqual(aim.pitch_compensation(None), 0.0)

    def test_opencv_keyboard_hold_and_events(self):
        kb = gs.OpenCVKeyboard()
        kb.feed_cv_key(ord("W"))
        kb.feed_cv_key(ord(" "))
        kb.feed_cv_key(ord("f"))
        kb.feed_cv_key(-1)
        self.assertTrue(kb.held("w"))
        self.assertFalse(kb.held("s"))
        self.assertEqual(kb.pop_events(), ["space", "f"])
        self.assertEqual(kb.pop_events(), [])
        kb.release_all()
        self.assertFalse(kb.held("w"))

    def test_latency_angle_lookup(self):
        io = gs.RobotIO.__new__(gs.RobotIO)
        io.angles = collections.deque([(1.0, 0.0, 10.0), (1.1, 1.0, 12.0), (1.2, 2.0, 14.0)])
        self.assertEqual(io.gimbal_angle(), (2.0, 14.0))
        self.assertEqual(io.gimbal_angle(1.15), (1.0, 12.0))
        self.assertEqual(io.gimbal_angle(0.5), (0.0, 10.0))

    def test_invalid_shooter_config(self):
        for section, override in [
            ("aiming", {"pitch_compensation": [[2.0, 1.0], [1.0, 2.0]]}),
            ("aiming", {"pitch_limits_deg": [30, -20]}),
            ("aiming", {"min_yaw_speed": 500}),
            ("aiming", {"target_smoothing": 0}),
            ("aiming", {"hold_enter_ratio": 2}),
            ("firing", {"auto_fire": "yes"}),
            ("firing", {"cooldown_s": 0}),
            ("controls", {"keyboard_backend": "joystick"}),
            ("controls", {"robot_mode": "sport"}),
        ]:
            with self.subTest(section=section, override=override):
                config = load_config()
                config["target_shooting"][section].update(override)
                with self.assertRaises(ValueError):
                    gs.build_settings(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
