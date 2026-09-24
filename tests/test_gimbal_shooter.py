"""Offline tests for src/gimbal_shooter.py (no robot or robomaster SDK needed).

The closed-loop test simulates a gimbal with motor lag, friction and a delayed
camera, renders synthetic frames and runs the real detector, tracker, aim
controller and fire control against it.
"""

import collections
import contextlib
import io
import math
import os
import shutil
import sys
import tempfile
import types
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


class CardVsWallTests(unittest.TestCase):
    """Cards on stands are engaged; walls, doors, floors and fallen cards are not."""

    def settings(self, **target):
        base = {"color": "green", "shape": "rectangle", "size_m": 0.07}
        base.update(target)
        return gs.build_settings({"target_shooting": {"target": base}})

    def test_wall_sized_colour_is_never_a_target(self):
        settings = self.settings()
        detector = ColorShapeDetector(settings)
        img = background(seed=1)
        cv2.rectangle(img, (0, 0), (W - 1, 260), (60, 170, 60), -1)      # green wall behind
        draw_shape(img, "rectangle", "green", (640, 430), 60, angle=90)  # a card on a stand
        targets = [d for d in detector.detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0].center[1], 430, delta=10)

    def test_landscape_card_is_still_a_target(self):
        """Standing cards come in both orientations; a wide card must not be
        mistaken for a fallen one (upright_only is off by default)."""
        detector = ColorShapeDetector(self.settings())
        img = background(seed=7)
        draw_shape(img, "rectangle", "green", (500, 380), 60, angle=0)    # landscape
        draw_shape(img, "rectangle", "green", (800, 380), 60, angle=90)   # portrait
        targets = [d for d in detector.detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 2)

    def test_fallen_card_is_rejected_and_leaning_card_accepted(self):
        detector = ColorShapeDetector(self.settings(upright_only=True))
        img = background(seed=2)
        draw_shape(img, "rectangle", "green", (400, 360), 60, angle=90)   # standing
        draw_shape(img, "rectangle", "green", (700, 360), 60, angle=65)   # leaning 25 deg
        draw_shape(img, "rectangle", "green", (1000, 360), 60, angle=0)   # fallen flat
        by_x = sorted(detector.detect(img)[0], key=lambda d: d.center[0])
        self.assertEqual([d.is_target for d in by_x], [True, True, False])
        self.assertEqual(by_x[2].reject_reason, "lying down")

    def test_card_touching_frame_edge_is_rejected(self):
        detector = ColorShapeDetector(self.settings())
        img = background(seed=3)
        draw_shape(img, "rectangle", "green", (8, 360), 60, angle=90)
        detections, _ = detector.detect(img)
        self.assertTrue(detections)
        self.assertFalse(detections[0].is_target)
        self.assertEqual(detections[0].reject_reason, "cut off by frame")


class KnockdownTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings_with()
        self.engagement = gs.Engagement(self.settings)

    def track(self, track_id=1):
        return type("T", (), {"track_id": track_id})()

    def test_card_that_vanishes_after_a_shot_counts_as_down(self):
        eng, track = self.engagement, self.track()
        for t in (0.0, 0.1, 0.2):
            self.assertFalse(eng.update(track, (10.0, -5.0), t, visible=True))
        eng.record_shot(0.3)
        # Still visible right after the shot: not down yet, and not judged during settle.
        self.assertFalse(eng.update(track, (10.0, -5.0), 0.5, visible=True))
        settle = self.settings["knockdown"]["settle_seconds"]
        for i in range(self.settings["knockdown"]["fall_missing_frames"]):
            down = eng.update(track, (10.0, -5.0), 0.3 + settle + 0.1 + i * 0.05, visible=False)
        self.assertTrue(down)
        self.assertEqual(eng.downed, 1)

    def test_card_that_drops_counts_as_down(self):
        eng, track = self.engagement, self.track()
        eng.update(track, (10.0, -5.0), 0.0, visible=True)
        eng.record_shot(0.1)
        settle = self.settings["knockdown"]["settle_seconds"]
        self.assertFalse(eng.update(track, (10.0, -5.2), 0.1 + settle - 0.1, visible=True))
        self.assertTrue(eng.update(track, (10.0, -8.5), 0.1 + settle + 0.1, visible=True))  # fell 3.5 deg

    def test_downed_spot_is_blacklisted_then_expires(self):
        eng, track = self.engagement, self.track()
        eng.update(track, (10.0, -5.0), 0.0, visible=True)
        eng.record_shot(0.1)
        for i in range(30):
            eng.update(track, (10.0, -5.0), 0.1 + self.settings["knockdown"]["settle_seconds"] + i * 0.05,
                       visible=False)
        self.assertTrue(eng.is_blacklisted(10.5, -5.5, now=5.0))     # same spot: skip it
        self.assertFalse(eng.is_blacklisted(25.0, -5.0, now=5.0))    # another card: fine
        later = 5.0 + self.settings["knockdown"]["blacklist_seconds"]
        self.assertFalse(eng.is_blacklisted(10.5, -5.5, now=later))

    def test_driving_clears_the_blacklist(self):
        eng = self.engagement
        eng.last_bearing = (10.0, -5.0)
        eng.mark_down(0.0)
        self.assertTrue(eng.is_blacklisted(10.0, -5.0, now=1.0))
        eng.clear_blacklist()
        self.assertFalse(eng.is_blacklisted(10.0, -5.0, now=1.0))

    def test_disabled_knockdown_never_declares_down(self):
        settings = gs.build_settings({"target_shooting": {"knockdown": {"enabled": False}}})
        eng = gs.Engagement(settings)
        track = self.track()
        eng.record_shot(0.0)
        for i in range(30):
            self.assertFalse(eng.update(track, (10.0, -5.0), 2.0 + i * 0.05, visible=False))


    def test_recoil_shake_right_after_a_shot_is_not_a_knockdown(self):
        """Firing blurs/shakes the picture: frames missed straight after the shot
        must not blacklist a card that is still standing."""
        eng, track = self.engagement, self.track()
        eng.update(track, (10.0, -5.0), 0.0, visible=True)
        eng.record_shot(0.1)
        for i in range(20):        # 20 missed frames, all inside the settle window
            self.assertFalse(eng.update(track, None, 0.15 + i * 0.04, visible=False))
        self.assertEqual(eng.downed, 0)
        # The card is seen again afterwards: still no knockdown, nothing blacklisted.
        settle = self.settings["knockdown"]["settle_seconds"]
        self.assertFalse(eng.update(track, (10.0, -5.0), 0.1 + settle + 0.1, visible=True))
        self.assertFalse(eng.is_blacklisted(10.0, -5.0, now=5.0))


    def test_give_up_blacklists_a_card_that_will_not_fall(self):
        eng = self.engagement
        eng.update(self.track(), (10.0, -5.0), 0.0, visible=True)
        eng.give_up(1.0)
        self.assertTrue(eng.is_blacklisted(10.0, -5.0, now=2.0))
        self.assertEqual(eng.downed, 0)       # not counted as a knockdown
        self.assertIsNone(eng.track_id)


class PatrolTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings_with()
        self.patrol = gs.PatrolController(self.settings)
        self.cfg = self.settings["patrol"]

    def test_off_by_default_and_toggles(self):
        self.assertFalse(self.patrol.enabled)
        self.assertEqual(self.patrol.update(0.0, 900, False), (0.0, 0.0, 0.0))
        self.assertTrue(self.patrol.toggle(0.0))
        x, y, z = self.patrol.update(0.1, 900, False)
        self.assertAlmostEqual(y, self.cfg["speed_mps"])
        self.assertEqual(z, 0.0)

    def test_stops_while_engaging_a_card(self):
        self.patrol.toggle(0.0)
        self.assertEqual(self.patrol.update(0.1, 900, engaged=True), (0.0, 0.0, 0.0))
        self.assertEqual(self.patrol.state, "holding (target)")

    def test_holds_distance_to_the_wall(self):
        self.patrol.toggle(0.0)
        far_x = self.patrol.update(0.1, self.cfg["wall_distance_mm"] + 400, False)[0]
        near_x = self.patrol.update(0.2, self.cfg["wall_distance_mm"] - 300, False)[0]
        self.assertGreater(far_x, 0)        # too far -> approach
        self.assertLess(near_x, 0)          # too close -> back away
        self.assertLessEqual(abs(far_x), self.cfg["max_approach_mps"])
        self.assertAlmostEqual(self.patrol.update(0.3, self.cfg["wall_distance_mm"], False)[0], 0.0)

    def test_never_closer_than_one_metre(self):
        self.patrol.toggle(0.0)
        self.assertGreaterEqual(self.cfg["min_distance_mm"], 1000)
        # Just inside the margin: creep backwards, keep patrolling sideways.
        x, y, _ = self.patrol.update(0.1, self.cfg["min_distance_mm"] - 50, False)
        self.assertLess(x, 0)
        self.assertNotEqual(y, 0.0)
        self.assertEqual(self.patrol.state, "keeping 1 m")
        # Much too close: back off at full speed.
        x, _, _ = self.patrol.update(0.2, self.cfg["min_distance_mm"] * 0.5, False)
        self.assertAlmostEqual(x, -self.cfg["max_approach_mps"])
        self.assertEqual(self.patrol.state, "backing off")

    def test_patrol_never_commands_forward_inside_the_margin(self):
        self.patrol.toggle(0.0)
        for front in range(200, 1001, 100):
            x = self.patrol.update(0.1, front, False)[0]
            self.assertLessEqual(x, 0.0, f"drove forward at {front} mm")

    def test_manual_forward_is_blocked_near_the_wall(self):
        limit = self.cfg["min_distance_mm"]
        # Too close: forward is cut, backing up and strafing still work.
        (x, y, z), blocked = gs.limit_forward((0.5, 0.3, 90.0), limit - 200, limit)
        self.assertEqual((x, y, z), (0.0, 0.3, 90.0))
        self.assertTrue(blocked)
        self.assertEqual(gs.limit_forward((-0.5, 0.0, 0.0), limit - 200, limit)[0][0], -0.5)
        # Far enough away, or no ToF reading: nothing is changed.
        self.assertEqual(gs.limit_forward((0.5, 0.0, 0.0), limit + 200, limit), ((0.5, 0.0, 0.0), False))
        self.assertEqual(gs.limit_forward((0.5, 0.0, 0.0), None, limit), ((0.5, 0.0, 0.0), False))
        self.assertEqual(gs.limit_forward((0.5, 0.0, 0.0), 0, limit), ((0.5, 0.0, 0.0), False))

    def test_reverses_direction_at_the_end_of_a_leg(self):
        self.patrol.toggle(0.0)
        first = self.patrol.update(0.1, 900, False)[1]
        later = self.patrol.update(self.cfg["leg_seconds"] + 0.2, 900, False)[1]
        self.assertAlmostEqual(first, -later)

    def test_missing_tof_still_patrols_without_approaching(self):
        self.patrol.toggle(0.0)
        x, y, _ = self.patrol.update(0.1, None, False)
        self.assertEqual(x, 0.0)
        self.assertNotEqual(y, 0.0)
        self.assertIn("no ToF", self.patrol.state)


class EndToEndKnockdownTests(unittest.TestCase):
    """Full program on a synthetic clip: fire at a card, the card goes down,
    and that spot is not shot again."""

    def make_clip(self, path, aim_x, aim_y, size_px, frames_present, frames_gone, frames_again):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 22, (W, H))
        def card(img):
            draw_shape(img, "rectangle", "green", (aim_x, aim_y), size_px, angle=90)
        for _ in range(frames_present):
            img = background(seed=1); card(img); writer.write(img)
        for _ in range(frames_gone):
            writer.write(background(seed=1))
        for _ in range(frames_again):          # card lying there / another one in the same spot
            img = background(seed=1); card(img); writer.write(img)
        writer.release()

    def test_stops_shooting_a_downed_card(self):
        settings = settings_with()
        aim = gs.AimController(settings)
        detector = ColorShapeDetector(settings)
        focal = detector.focal_length(W)
        distance = 1.2
        # Put the card exactly where the barrel points, so the (stationary, offline)
        # gimbal locks on immediately.
        comp = aim.pitch_compensation(distance)
        aim_y = H / 2 + focal * math.tan(math.radians(comp))
        size_px = focal * 0.07 / distance

        tmp = tempfile.mkdtemp()
        clip = os.path.join(tmp, "cards.mp4")
        try:
            self.make_clip(clip, W / 2, aim_y, size_px, 40, 60, 60)
            args = types.SimpleNamespace(connection=None, source=clip, color="green", shape="rectangle",
                                         size=0.07, auto_fire=True, patrol=False, headless=True,
                                         max_frames=0, save=None)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                gs.run(args)
            text = out.getvalue()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        before_down, _, after_down = text.partition("Target down")
        self.assertTrue(after_down, f"card was never declared down:\n{text}")
        self.assertIn("FIRE (auto)", before_down)
        self.assertNotIn("FIRE", after_down, "kept shooting a card that was already down")
        self.assertLessEqual(text.count("FIRE (auto)"), settings["firing"]["max_shots_per_target"])


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
            ("patrol", {"speed_mps": 0}),
            ("patrol", {"min_distance_mm": 5000}),
            ("knockdown", {"fall_missing_frames": 0}),
            ("knockdown", {"skip_after_max_shots": "sometimes"}),
        ]:
            with self.subTest(section=section, override=override):
                config = load_config()
                config["target_shooting"][section].update(override)
                with self.assertRaises(ValueError):
                    gs.build_settings(config)


if __name__ == "__main__":
    unittest.main(verbosity=2)
