"""Offline tests for src/target_detection.py using synthetic camera frames.

Run:  python3 -m unittest discover -s tests -v
"""

import math
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from target_detection import (  # noqa: E402
    ColorShapeDetector,
    TargetTracker,
    build_detection_settings,
    focal_length_from_fov,
    pixel_to_angles,
)

W, H = 1280, 720
BGR = {"red": (30, 30, 220), "green": (40, 180, 40), "blue": (200, 90, 20), "yellow": (30, 220, 230)}


def make_settings(**target):
    return build_detection_settings({"target_shooting": {"target": target}})


def background(seed=0, brightness=1.0, noise=8):
    rng = np.random.default_rng(seed)
    # Grey gradient "room" with a wooden-ish floor so it is not a flat colour.
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.linspace(150, 110, W)[None, :, None]
    img[int(H * 0.65):] = (70, 100, 140)
    img += rng.normal(0, noise, img.shape)
    return np.clip(img * brightness, 0, 255).astype(np.uint8)


def draw_shape(img, shape, color, center, size_px, tilt_x=1.0, angle=0.0):
    """size_px is the largest outer dimension; tilt_x squashes horizontally (perspective)."""
    cx, cy = center
    col = BGR[color] if isinstance(color, str) else color
    if shape == "circle":
        axes = (max(1, int(size_px / 2 * tilt_x)), int(size_px / 2))
        cv2.ellipse(img, (int(cx), int(cy)), axes, angle, 0, 360, col, -1, cv2.LINE_AA)
        return
    if shape == "square":
        pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float32) * size_px / 2
    elif shape == "rectangle":
        pts = np.array([[-1, -0.5], [1, -0.5], [1, 0.5], [-1, 0.5]], np.float32) * size_px / 2
    else:  # triangle, side = size_px
        r = size_px / math.sqrt(3)
        pts = np.array([[r * math.cos(math.radians(a)), r * math.sin(math.radians(a))]
                        for a in (-90, 30, 150)], np.float32)
    rot = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
    pts = pts @ rot.T
    pts[:, 0] *= tilt_x
    pts += (cx, cy)
    cv2.fillPoly(img, [np.round(pts).astype(np.int32)], col, cv2.LINE_AA)


def size_for_distance(settings, distance_m):
    focal = focal_length_from_fov(W, settings["camera"]["horizontal_fov_deg"])
    return focal * settings["target"]["size_m"] / distance_m


class ShapeAndColorTests(unittest.TestCase):
    def test_each_shape_and_color_is_classified(self):
        settings = make_settings(color="red", shape="any")
        detector = ColorShapeDetector(settings)
        for color in BGR:
            for shape in ("circle", "square", "rectangle", "triangle"):
                with self.subTest(color=color, shape=shape):
                    img = background(seed=1)
                    draw_shape(img, shape, color, (640, 360), 160, angle=12 if shape != "circle" else 0)
                    found = [d for d in detector.detect(img)[0] if d.color == color]
                    self.assertEqual(len(found), 1, [(d.color, d.shape) for d in found])
                    self.assertEqual(found[0].shape, shape)

    def test_target_rules_color_and_shape(self):
        settings = make_settings(color="red", shape="circle", size_m=0.2)
        img = background(seed=2)
        draw_shape(img, "circle", "red", (400, 360), 120)       # the target
        draw_shape(img, "square", "red", (700, 360), 120)       # wrong shape
        draw_shape(img, "circle", "blue", (1000, 360), 120)     # wrong colour
        detections, _ = ColorShapeDetector(settings).detect(img)
        targets = [d for d in detections if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0].center[0], 400, delta=4)
        reasons = sorted(d.reject_reason for d in detections if not d.is_target)
        self.assertEqual(reasons, ["wrong color", "wrong shape"])

    def test_no_false_positive_on_empty_or_bad_frames(self):
        detector = ColorShapeDetector(make_settings())
        for seed in range(5):
            self.assertEqual(detector.detect(background(seed=seed, noise=20))[0], [])
        self.assertEqual(detector.detect(None)[0], [])
        self.assertEqual(detector.detect(np.zeros((8, 8, 3), np.uint8))[0], [])
        self.assertEqual(detector.detect(np.zeros((H, W), np.uint8))[0], [])

    def test_thin_red_line_and_skin_tone_rejected(self):
        img = background(seed=3)
        cv2.line(img, (100, 100), (1100, 140), BGR["red"], 6)             # thin line
        draw_shape(img, "circle", (120, 160, 210), (640, 400), 200)        # skin/pale orange
        detections, _ = ColorShapeDetector(make_settings(shape="any")).detect(img)
        self.assertFalse([d for d in detections if d.is_target])


class DistanceTests(unittest.TestCase):
    def test_distance_accuracy_all_shapes(self):
        for shape in ("circle", "square", "triangle"):
            settings = make_settings(color="red", shape=shape, size_m=0.2, max_distance_m=5.0)
            detector = ColorShapeDetector(settings)
            for distance in (0.5, 1.0, 1.5, 2.0, 3.0):
                with self.subTest(shape=shape, distance=distance):
                    img = background(seed=4)
                    draw_shape(img, shape, "red", (640, 360), size_for_distance(settings, distance))
                    targets = [d for d in detector.detect(img)[0] if d.is_target]
                    self.assertEqual(len(targets), 1)
                    error = abs(targets[0].distance_m - distance) / distance
                    self.assertLess(error, 0.08, f"measured {targets[0].distance_m:.3f}")

    def test_tilted_circle_distance_uses_major_axis(self):
        settings = make_settings(color="red", shape="circle", size_m=0.2)
        img = background(seed=5)
        draw_shape(img, "circle", "red", (640, 360), size_for_distance(settings, 1.2), tilt_x=0.75)
        targets = [d for d in ColorShapeDetector(settings).detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0].distance_m, 1.2, delta=0.1)

    def test_out_of_range_is_not_a_target(self):
        settings = make_settings(color="red", shape="circle", size_m=0.2, max_distance_m=2.0)
        img = background(seed=6)
        draw_shape(img, "circle", "red", (640, 360), size_for_distance(settings, 2.8))
        detections, _ = ColorShapeDetector(settings).detect(img)
        self.assertEqual(len(detections), 1)
        self.assertFalse(detections[0].is_target)
        self.assertEqual(detections[0].reject_reason, "out of range")

    def test_calibrated_focal_length_overrides_fov(self):
        settings = make_settings(color="red", shape="circle", size_m=0.2)
        detector = ColorShapeDetector(settings)
        detector.set_focal_length(1000.0)
        img = background(seed=7)
        draw_shape(img, "circle", "red", (640, 360), 100)   # 1000 * 0.2 / 100 = 2.0 m
        target = [d for d in detector.detect(img)[0] if d.is_target][0]
        self.assertAlmostEqual(target.distance_m, 2.0, delta=0.08)


class RobustnessTests(unittest.TestCase):
    def test_dim_light_noise_and_blur(self):
        settings = make_settings(color="red", shape="circle", size_m=0.2)
        detector = ColorShapeDetector(settings)
        img = background(seed=8, brightness=0.55, noise=14)
        draw_shape(img, "circle", (25, 25, 150), (500, 300), size_for_distance(settings, 1.5))
        img = cv2.GaussianBlur(img, (9, 9), 0)
        noise = np.random.default_rng(8).normal(0, 10, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        targets = [d for d in detector.detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertAlmostEqual(targets[0].distance_m, 1.5, delta=0.15)

    def test_edge_clipped_target_is_flagged(self):
        img = background(seed=9)
        draw_shape(img, "circle", "red", (10, 360), 200)
        detections, _ = ColorShapeDetector(make_settings(shape="any")).detect(img)
        self.assertTrue(detections and detections[0].clipped)

    def test_angles(self):
        focal = focal_length_from_fov(W, 96.0)
        self.assertEqual(pixel_to_angles(640, 360, W, H, focal), (0.0, 0.0))
        yaw, pitch = pixel_to_angles(1280, 0, W, H, focal)
        self.assertAlmostEqual(yaw, 48.0, places=3)
        self.assertGreater(pitch, 0)


class TrackerTests(unittest.TestCase):
    def setUp(self):
        self.settings = make_settings(color="red", shape="circle", size_m=0.2)
        self.detector = ColorShapeDetector(self.settings)
        self.tracker = TargetTracker(self.settings)

    def frame(self, centers):
        img = background(seed=10)
        for c in centers:
            draw_shape(img, "circle", "red", c, 120)
        return img

    def step(self, centers):
        img = self.frame(centers)
        return self.tracker.update(self.detector.detect(img)[0], img.shape)

    def test_confirmation_needs_consecutive_hits(self):
        n = self.settings["tracking"]["confirm_frames"]
        for i in range(n - 1):
            self.assertFalse(self.step([(640, 360)]).confirmed)
        self.assertTrue(self.step([(640, 360)]).confirmed)
        self.assertTrue(self.tracker.visible)

    def test_survives_short_dropout_then_drops(self):
        for _ in range(4):
            self.step([(640, 360)])
        self.assertIsNotNone(self.step([]))
        self.assertFalse(self.tracker.visible)          # never shoot on a missed frame
        self.assertTrue(self.step([(650, 362)]).confirmed)
        for _ in range(self.settings["tracking"]["lost_frames"] + 1):
            track = self.step([])
        self.assertIsNone(track)

    def test_does_not_hop_to_second_target(self):
        for x in range(500, 560, 10):
            track = self.step([(x, 360), (1000, 360)])
        self.assertLess(track.center[0], 600)

    def test_ignores_single_frame_glitch_far_away(self):
        for _ in range(4):
            self.step([(400, 360)])
        track = self.step([(1100, 200)])            # sudden far blob only
        self.assertLess(track.center[0], 450)
        self.assertFalse(self.tracker.visible)


def rgb(r, g, b):
    return (b, g, r)


LIGHTS = {
    "normal": (1.0, None), "dim": (0.6, None), "dark": (0.35, None), "bright": (1.35, None),
    "warm": (0.8, [0.7, 0.95, 1.25]), "dim+warm": (0.6, [0.7, 0.95, 1.25]),
    "dark+warm": (0.4, [0.7, 0.95, 1.25]), "cool": (0.8, [1.25, 1.0, 0.75]),
}


def relight(img, light):
    gain, cast = LIGHTS[light]
    out = img.astype(np.float32) * gain
    if cast:
        out *= np.array(cast, np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


class LightingTests(unittest.TestCase):
    """Real cards under real rooms: dark/tinted colours, dim and warm/cool light."""

    CARDS = {
        "navy (violet tint)": ("blue", rgb(55, 25, 125)),
        "navy": ("blue", rgb(25, 25, 110)),
        "bright blue": ("blue", rgb(30, 90, 220)),
        "dark green": ("green", rgb(10, 90, 45)),
        "dark red": ("red", rgb(120, 5, 10)),
        "bright red": ("red", rgb(220, 30, 30)),
        "mustard": ("yellow", rgb(190, 155, 10)),
    }

    def test_cards_detected_in_every_lighting(self):
        for name, (want, card) in self.CARDS.items():
            for light in LIGHTS:
                with self.subTest(card=name, light=light):
                    img = background(seed=1)
                    draw_shape(img, "square", card, (640, 360), 150)
                    detections, _ = ColorShapeDetector(make_settings(shape="any")).detect(relight(img, light))
                    self.assertEqual([(d.color, d.shape) for d in detections], [(want, "square")])

    def test_no_false_positives_in_any_lighting(self):
        for light in LIGHTS:
            for seed in range(3):
                with self.subTest(light=light, seed=seed):
                    img = relight(background(seed=seed, noise=14), light)
                    self.assertEqual(ColorShapeDetector(make_settings(shape="any")).detect(img)[0], [])

    def test_all_four_cards_in_warm_dim_room(self):
        """The screenshot case: yellow circle, navy, red and dark green cards side by side."""
        img = background(seed=2)
        draw_shape(img, "circle", rgb(190, 155, 10), (250, 360), 150)
        draw_shape(img, "rectangle", rgb(40, 25, 120), (520, 360), 170, angle=90)
        draw_shape(img, "rectangle", rgb(120, 5, 10), (800, 360), 170, angle=90)
        draw_shape(img, "square", rgb(10, 90, 45), (1080, 360), 150)
        detections, _ = ColorShapeDetector(make_settings(shape="any")).detect(relight(img, "dim+warm"))
        found = sorted((d.color, d.shape) for d in detections)
        self.assertEqual(found, [("blue", "rectangle"), ("green", "square"), ("red", "rectangle"),
                                 ("yellow", "circle")])


class ShapeMatchingTests(unittest.TestCase):
    def test_rectangle_target_accepts_square_cards(self):
        from target_detection import shape_matches
        self.assertTrue(shape_matches("rectangle", "square"))
        self.assertTrue(shape_matches("rectangle", "rectangle"))
        self.assertFalse(shape_matches("square", "rectangle"))
        self.assertTrue(shape_matches("any", "triangle"))
        self.assertFalse(shape_matches("circle", "square"))

    def test_rectangle_orientation_matching(self):
        from target_detection import shape_matches
        self.assertTrue(shape_matches("rectangle_v", "rectangle", "vertical"))
        self.assertFalse(shape_matches("rectangle_v", "rectangle", "horizontal"))
        self.assertTrue(shape_matches("rectangle_h", "rectangle", "horizontal"))
        self.assertFalse(shape_matches("rectangle_h", "rectangle", "vertical"))
        self.assertFalse(shape_matches("rectangle_v", "square", ""))     # a square is neither
        self.assertTrue(shape_matches("rectangle", "rectangle", "horizontal"))

    def test_portrait_and_landscape_cards_are_separated(self):
        img = background(seed=11)
        draw_shape(img, "rectangle", "green", (300, 360), 60, angle=90)   # portrait
        draw_shape(img, "rectangle", "green", (650, 360), 60, angle=0)    # landscape
        draw_shape(img, "square", "green", (1000, 360), 55)               # square
        wanted = {"rectangle_v": [300], "rectangle_h": [650],
                  "rectangle": [300, 650, 1000], "square": [1000]}
        for shape, xs in wanted.items():
            with self.subTest(shape=shape):
                settings = make_settings(color="green", shape=shape, size_m=0.07)
                targets = [d for d in ColorShapeDetector(settings).detect(img)[0] if d.is_target]
                self.assertEqual(sorted(round(d.center[0] / 50) * 50 for d in targets), xs)

    def test_leaning_card_still_counts_as_portrait(self):
        settings = make_settings(color="green", shape="rectangle_v", size_m=0.07)
        for lean in (0, 15, 30):
            with self.subTest(lean=lean):
                img = background(seed=12)
                draw_shape(img, "rectangle", "green", (640, 360), 60, angle=90 - lean)
                targets = [d for d in ColorShapeDetector(settings).detect(img)[0] if d.is_target]
                self.assertEqual(len(targets), 1, f"leaning {lean} deg was not seen as portrait")
                self.assertEqual(targets[0].orientation, "vertical")

    def test_near_square_card_is_engaged_for_a_rectangle_target(self):
        settings = make_settings(color="yellow", shape="rectangle", size_m=0.07)
        detector = ColorShapeDetector(settings)
        img = background(seed=6)
        # A card whose sides are nearly equal (classified "square").
        draw_shape(img, "square", "yellow", (640, 360), 60)
        targets = [d for d in detector.detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].shape, "square")


class LearnColorTests(unittest.TestCase):
    def test_click_learns_unlisted_colour(self):
        # Purple is not in the default ranges; clicking it teaches "blue" to mean this card.
        settings = make_settings(color="blue", shape="any", size_m=0.2)
        detector = ColorShapeDetector(settings)
        img = background(seed=3)
        draw_shape(img, "square", rgb(120, 40, 150), (640, 360), 150)
        img = relight(img, "warm")
        self.assertFalse([d for d in detector.detect(img)[0] if d.color == "blue"])
        ranges = detector.learn_color(img, 640, 360, "blue")
        self.assertTrue(ranges)
        targets = [d for d in detector.detect(img)[0] if d.is_target]
        self.assertEqual(len(targets), 1)
        self.assertEqual(settings["detection"]["colors"]["blue"], ranges)

    def test_learning_red_wraps_hue(self):
        detector = ColorShapeDetector(make_settings(color="red", shape="any", size_m=0.2))
        img = background(seed=4)
        draw_shape(img, "circle", rgb(150, 10, 25), (640, 360), 150)   # hue ~ 177, next to the wrap
        ranges = detector.learn_color(img, 640, 360)
        self.assertEqual(len(ranges), 2)
        self.assertEqual(len([d for d in detector.detect(img)[0] if d.is_target]), 1)

    def test_clicking_background_is_refused(self):
        detector = ColorShapeDetector(make_settings())
        img = background(seed=5)
        before = [list(r["lower"]) for r in detector.settings["detection"]["colors"]["red"]]
        with self.assertRaises(ValueError):
            detector.learn_color(img, 640, 200)
        self.assertEqual([list(r["lower"]) for r in detector.settings["detection"]["colors"]["red"]], before)


class ConfigTests(unittest.TestCase):
    def test_repo_config_is_valid(self):
        from config_loader import load_config
        settings = build_detection_settings(load_config())
        self.assertIn(settings["target"]["color"], settings["detection"]["colors"])

    def test_bad_values_raise_clear_errors(self):
        bad = [
            {"target": {"color": "purple"}},
            {"target": {"shape": "star"}},
            {"target": {"min_distance_m": 3, "max_distance_m": 1}},
            {"detection": {"morph_kernel": 4}},
            {"detection": {"colors": {"red": [{"lower": [170, 0, 0], "upper": [10, 255, 255]}]}}},
            {"tracking": {"smoothing": 0}},
            {"lighting": {"clahe": "yes"}},
            {"lighting": {"max_brightness_gain": 20}},
            {"camera": {"focal_length_px": -5}},
        ]
        for override in bad:
            with self.subTest(override=override):
                with self.assertRaises(ValueError):
                    build_detection_settings({"target_shooting": override})


if __name__ == "__main__":
    unittest.main(verbosity=2)
