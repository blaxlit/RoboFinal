"""Colour + shape + distance target detection for the RoboMaster camera.

This module only depends on OpenCV/NumPy (no robomaster SDK), so it can be
tested on images, videos or a laptop webcam:

    python3 src/target_detection.py --source 0            # webcam preview
    python3 src/target_detection.py --source photo.jpg    # single image

Pipeline for every frame:
    1. Find candidates fast: resize to ``process_width``, blur, HSV threshold
       every configured colour, morphology clean-up.
    2. Refine each candidate on a full-resolution ROI so small / far targets
       keep their corners (accurate shape and size).
    3. Contour filtering (area, solidity) and shape classification
       (circle / square / rectangle / triangle).
    4. Distance from the pinhole model: distance = focal_px * size_m / size_px.
    5. Target selection + temporal tracking so one noisy frame never flips
       the target or triggers a shot.
"""

import argparse
import copy
import itertools
import math
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

SHAPES = ("circle", "square", "rectangle", "triangle")
# Rectangles can be picked by orientation: rectangle_v = standing/portrait,
# rectangle_h = lying/landscape, rectangle = either.
TARGET_SHAPES = ("circle", "square", "rectangle", "rectangle_v", "rectangle_h", "triangle", "any")
VERTICAL, HORIZONTAL = "vertical", "horizontal"

# ==============================================================================
# 1. Settings (defaults + validation)
# ==============================================================================
DEFAULT_SETTINGS = {
    "camera": {
        "resolution": "720p",
        # Used only when focal_length_px is null. Press C in the shooter to
        # calibrate the real focal length from a target at a known distance.
        "horizontal_fov_deg": 96.0,
        "focal_length_px": None,
        "calibration_distance_m": 1.0,
        "swap_red_blue": False,
    },
    "detection": {
        "process_width": 960,
        "min_area_px": 40,         # contour area at process_width scale
        "max_area_ratio": 0.25,    # a blob bigger than this is a wall/door, not a card
        "min_solidity": 0.85,      # area / convex hull area
        "blur_kernel": 5,
        "morph_kernel": 5,
        # Hue ranges are applied AFTER lighting normalisation and cover the hue
        # wheel without gaps (only orange/brown 10-17 and purple 136-159 are left
        # out), so dark or tinted cards (navy, teal-green) still get a colour.
        "colors": {
            "red": [
                {"lower": [0, 90, 45], "upper": [9, 255, 255]},
                {"lower": [160, 90, 45], "upper": [179, 255, 255]},
            ],
            "yellow": [{"lower": [18, 90, 70], "upper": [38, 255, 255]}],
            "green": [{"lower": [39, 60, 35], "upper": [88, 255, 255]}],
            "blue": [{"lower": [89, 70, 35], "upper": [135, 255, 255]}],
        },
    },
    "lighting": {
        "white_balance": True,     # cancel warm/cool light colour casts
        "max_wb_gain": 2.0,
        "auto_brightness": True,   # brighten dark frames before colour thresholding
        "target_brightness": 120,  # mean grey level aimed for (0-255)
        "max_brightness_gain": 3.0,
        "wb_smoothing": 0.2,       # per-frame EMA of the gains (video)
        "clahe": True,             # local contrast/brightness equalisation (dim light, shadows)
        "clahe_clip": 2.0,
    },
    "target": {
        "color": "red",
        "shape": "circle",
        "size_m": 0.07,            # largest outer dimension of the real card
        "min_distance_m": 0.3,
        "max_distance_m": 3.0,
        # Walls, doors and floors are rejected by these: a card is small, fully
        # visible, and standing (a knocked-over card lies flat).
        "require_fully_visible": True,   # ignore anything touching the frame edge
        # Only for card sets that are ALL portrait: it rejects wide (landscape)
        # cards too, so it is off by default. Knocked-over cards are handled by
        # the knockdown watcher in gimbal_shooter.py instead.
        "upright_only": False,
        "upright_tolerance_deg": 45.0,   # cards may lean; 90 deg = lying on its side
    },
    "tracking": {
        "confirm_frames": 3,       # hits needed before the target is trusted
        "lost_frames": 6,          # misses tolerated before the track is dropped
        "smoothing": 0.5,          # EMA weight of the newest measurement (0-1]
        "max_jump_ratio": 2.0,     # max centre jump, in target sizes, per frame
    },
}


def _deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict) and key != "colors":
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _require(condition, message):
    if not condition:
        raise ValueError(f"Invalid target_shooting config: {message}")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_detection_settings(settings):
    """Raise ValueError with a readable message for any bad setting."""
    cam, det, light = settings["camera"], settings["detection"], settings["lighting"]
    tgt, trk = settings["target"], settings["tracking"]

    _require(cam["resolution"] in ("360p", "540p", "720p"),
             "camera.resolution must be 360p, 540p or 720p")
    _require(_is_number(cam["horizontal_fov_deg"]) and 10 < cam["horizontal_fov_deg"] < 170,
             "camera.horizontal_fov_deg must be between 10 and 170")
    _require(cam["focal_length_px"] is None
             or (_is_number(cam["focal_length_px"]) and cam["focal_length_px"] > 0),
             "camera.focal_length_px must be null or > 0")
    _require(_is_number(cam["calibration_distance_m"]) and cam["calibration_distance_m"] > 0,
             "camera.calibration_distance_m must be > 0")

    _require(isinstance(det["process_width"], int) and 160 <= det["process_width"] <= 1920,
             "detection.process_width must be an integer 160-1920")
    _require(_is_number(det["min_area_px"]) and det["min_area_px"] > 0,
             "detection.min_area_px must be > 0")
    _require(_is_number(det["max_area_ratio"]) and 0 < det["max_area_ratio"] <= 1,
             "detection.max_area_ratio must be in (0, 1]")
    _require(_is_number(det["min_solidity"]) and 0 <= det["min_solidity"] <= 1,
             "detection.min_solidity must be in [0, 1]")
    for name in ("blur_kernel", "morph_kernel"):
        k = det[name]
        _require(isinstance(k, int) and k >= 1 and k % 2 == 1,
                 f"detection.{name} must be a positive odd integer")

    _require(all(isinstance(light[k], bool) for k in ("white_balance", "clahe", "auto_brightness")),
             "lighting.white_balance, lighting.auto_brightness and lighting.clahe must be true/false")
    _require(_is_number(light["target_brightness"]) and 40 <= light["target_brightness"] <= 200,
             "lighting.target_brightness must be in [40, 200]")
    _require(_is_number(light["max_brightness_gain"]) and 1.0 <= light["max_brightness_gain"] <= 6.0,
             "lighting.max_brightness_gain must be in [1, 6]")
    _require(_is_number(light["max_wb_gain"]) and 1.0 <= light["max_wb_gain"] <= 4.0,
             "lighting.max_wb_gain must be in [1, 4]")
    _require(_is_number(light["wb_smoothing"]) and 0 < light["wb_smoothing"] <= 1,
             "lighting.wb_smoothing must be in (0, 1]")
    _require(_is_number(light["clahe_clip"]) and 0.5 <= light["clahe_clip"] <= 10,
             "lighting.clahe_clip must be in [0.5, 10]")

    colors = det["colors"]
    _require(isinstance(colors, dict) and colors, "detection.colors must be a non-empty mapping")
    for color, ranges in colors.items():
        _require(isinstance(ranges, list) and ranges,
                 f"detection.colors.{color} must be a list of ranges")
        for rng in ranges:
            _require(isinstance(rng, dict) and "lower" in rng and "upper" in rng,
                     f"detection.colors.{color} ranges need 'lower' and 'upper'")
            for bound in ("lower", "upper"):
                vals = rng[bound]
                _require(isinstance(vals, list) and len(vals) == 3
                         and all(isinstance(v, int) for v in vals),
                         f"detection.colors.{color}.{bound} must be 3 integers [H, S, V]")
                _require(0 <= vals[0] <= 179 and 0 <= vals[1] <= 255 and 0 <= vals[2] <= 255,
                         f"detection.colors.{color}.{bound} out of range (H 0-179, S/V 0-255)")
            _require(all(lo <= hi for lo, hi in zip(rng["lower"], rng["upper"])),
                     f"detection.colors.{color}: lower must be <= upper "
                     "(split red into two ranges instead of wrapping hue)")

    _require(tgt["color"] in colors,
             f"target.color '{tgt['color']}' is not in detection.colors {sorted(colors)}")
    _require(tgt["shape"] in TARGET_SHAPES, f"target.shape must be one of {TARGET_SHAPES}")
    _require(_is_number(tgt["size_m"]) and tgt["size_m"] > 0, "target.size_m must be > 0")
    _require(_is_number(tgt["min_distance_m"]) and _is_number(tgt["max_distance_m"])
             and 0 <= tgt["min_distance_m"] < tgt["max_distance_m"],
             "target.min_distance_m must be >= 0 and < target.max_distance_m")
    _require(isinstance(tgt["require_fully_visible"], bool) and isinstance(tgt["upright_only"], bool),
             "target.require_fully_visible and target.upright_only must be true/false")
    _require(_is_number(tgt["upright_tolerance_deg"]) and 0 < tgt["upright_tolerance_deg"] <= 90,
             "target.upright_tolerance_deg must be in (0, 90]")

    _require(isinstance(trk["confirm_frames"], int) and trk["confirm_frames"] >= 1,
             "tracking.confirm_frames must be an integer >= 1")
    _require(isinstance(trk["lost_frames"], int) and trk["lost_frames"] >= 0,
             "tracking.lost_frames must be an integer >= 0")
    _require(_is_number(trk["smoothing"]) and 0 < trk["smoothing"] <= 1,
             "tracking.smoothing must be in (0, 1]")
    _require(_is_number(trk["max_jump_ratio"]) and trk["max_jump_ratio"] > 0,
             "tracking.max_jump_ratio must be > 0")


def build_detection_settings(config=None):
    """Merge ``config['target_shooting']`` over defaults and validate it."""
    user = (config or {}).get("target_shooting", {}) or {}
    settings = {key: _deep_merge(DEFAULT_SETTINGS[key], user.get(key)) for key in DEFAULT_SETTINGS}
    validate_detection_settings(settings)
    return settings


# ==============================================================================
# 2. Geometry helpers
# ==============================================================================
def shape_matches(wanted, found, orientation=""):
    """Does a detected shape match the selected target shape?

    'rectangle' accepts squares too (a card whose sides are nearly equal would
    otherwise flip in and out), while 'rectangle_v' / 'rectangle_h' pick only
    portrait / landscape rectangles - squares are neither.
    """
    if wanted == "any":
        return True
    if wanted == "rectangle":
        return found in ("rectangle", "square")
    if wanted == "rectangle_v":
        return found == "rectangle" and orientation == VERTICAL
    if wanted == "rectangle_h":
        return found == "rectangle" and orientation == HORIZONTAL
    return found == wanted


def focal_length_from_fov(image_width, horizontal_fov_deg):
    return (image_width / 2.0) / math.tan(math.radians(horizontal_fov_deg) / 2.0)


def pixel_to_angles(x, y, image_width, image_height, focal_px):
    """Angle (deg) from the optical axis: +yaw = right, +pitch = up."""
    yaw = math.degrees(math.atan2(x - image_width / 2.0, focal_px))
    pitch = math.degrees(math.atan2(image_height / 2.0 - y, focal_px))
    return yaw, pitch


def estimate_distance(size_px, size_m, focal_px):
    if size_px <= 1 or size_m <= 0 or focal_px <= 0:
        return None
    return focal_px * size_m / size_px


# ==============================================================================
# 3. Detector
# ==============================================================================
@dataclass
class Detection:
    color: str
    shape: str                       # circle/square/rectangle/triangle/unknown
    center: Tuple[float, float]      # full-resolution pixels
    bbox: Tuple[int, int, int, int]  # x, y, w, h (full resolution)
    contour: np.ndarray              # full-resolution contour
    area_px: float                   # full-resolution area
    size_px: float                   # largest outer dimension in pixels
    circularity: float
    clipped: bool                    # touches the image border (partial view)
    tilt_deg: float = 0.0            # long axis away from vertical (90 = lying down)
    orientation: str = ""            # "vertical" / "horizontal" for rectangles
    distance_m: Optional[float] = None
    is_target: bool = False
    reject_reason: str = ""


class ColorShapeDetector:
    def __init__(self, settings):
        self.settings = settings
        det = settings["detection"]
        self.colors = {
            name: [(np.array(r["lower"], np.uint8), np.array(r["upper"], np.uint8)) for r in ranges]
            for name, ranges in det["colors"].items()
        }
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (det["morph_kernel"], det["morph_kernel"])
        )
        self.focal_px_override = settings["camera"]["focal_length_px"]
        light = settings["lighting"]
        self.clahe = cv2.createCLAHE(clipLimit=light["clahe_clip"], tileGridSize=(8, 8))
        self.wb_gains = None  # smoothed per-channel gains (B, G, R)

    # -------------------------------------------------------------- lighting
    def _white_balance_gains(self, image_bgr):
        """Grey-world gains estimated from near-neutral pixels (walls, floor, boards),
        so a big coloured target does not bias the correction."""
        small = cv2.resize(image_bgr, (160, 90), interpolation=cv2.INTER_AREA).reshape(-1, 3).astype(np.float32)
        vmax, vmin = small.max(axis=1), small.min(axis=1)
        neutral = (vmax > 30) & (vmax < 250) & ((vmax - vmin) < 0.45 * vmax)
        ref = small[neutral] if neutral.sum() > 0.05 * len(small) else small
        means = np.maximum(ref.mean(axis=0), 1.0)
        light = self.settings["lighting"]
        gains = np.ones(3, np.float32)
        if light["white_balance"]:
            limit = light["max_wb_gain"]
            gains = np.clip(means.mean() / means, 1.0 / limit, limit)
        if light["auto_brightness"]:
            level = float((small * gains).mean())
            gains = gains * float(np.clip(light["target_brightness"] / max(level, 1.0),
                                          1.0, light["max_brightness_gain"]))
        return gains

    def normalize_lighting(self, frame_bgr, update=True):
        """Return a colour-cast-free, brightness- and contrast-equalised copy of the frame."""
        light = self.settings["lighting"]
        out = frame_bgr
        if light["white_balance"] or light["auto_brightness"]:
            gains = self._white_balance_gains(frame_bgr)
            if update:
                a = light["wb_smoothing"]
                self.wb_gains = gains if self.wb_gains is None else a * gains + (1 - a) * self.wb_gains
            use = self.wb_gains if self.wb_gains is not None else gains
            out = cv2.convertScaleAbs(out.astype(np.float32) * use.reshape(1, 1, 3))
        if light["clahe"]:
            lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            out = cv2.cvtColor(cv2.merge((self.clahe.apply(l_channel), a_channel, b_channel)), cv2.COLOR_LAB2BGR)
        return out

    def learn_color(self, frame_bgr, x, y, color=None, radius=7):
        """Sample the colour under (x, y) and replace ``color``'s HSV ranges with it.

        Returns the new ranges. Raises ValueError when the spot is not colourful.
        """
        color = color or self.settings["target"]["color"]
        h, w = frame_bgr.shape[:2]
        x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
        y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
        if x1 <= x0 or y1 <= y0:
            raise ValueError("click inside the image")
        patch = self.normalize_lighting(frame_bgr, update=False)[y0:y1, x0:x1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
        sat, val = float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))
        if sat < 45 or val < 25:
            raise ValueError(f"that spot is not colourful enough (S={sat:.0f}, V={val:.0f}); click the middle of the card")
        # Circular hue statistics (red wraps around 0/179).
        angles = hsv[:, 0] * (2 * np.pi / 180.0)
        mean_angle = math.atan2(float(np.sin(angles).mean()), float(np.cos(angles).mean()))
        hue = (mean_angle * 180.0 / (2 * np.pi)) % 180.0
        diffs = np.abs(((hsv[:, 0] - hue) + 90) % 180 - 90)
        spread = max(8.0, 2.5 * float(np.percentile(diffs, 90)) + 4.0)
        lo_s, lo_v = int(max(40, 0.5 * sat)), int(max(25, 0.4 * val))
        lo_h, hi_h = int(math.floor(hue - spread)), int(math.ceil(hue + spread))
        ranges = []
        if lo_h < 0:
            ranges += [{"lower": [0, lo_s, lo_v], "upper": [hi_h, 255, 255]},
                       {"lower": [180 + lo_h, lo_s, lo_v], "upper": [179, 255, 255]}]
        elif hi_h > 179:
            ranges += [{"lower": [lo_h, lo_s, lo_v], "upper": [179, 255, 255]},
                       {"lower": [0, lo_s, lo_v], "upper": [hi_h - 180, 255, 255]}]
        else:
            ranges.append({"lower": [lo_h, lo_s, lo_v], "upper": [hi_h, 255, 255]})
        self.settings["detection"]["colors"][color] = ranges
        self.colors[color] = [(np.array(r["lower"], np.uint8), np.array(r["upper"], np.uint8)) for r in ranges]
        return ranges

    # -------------------------------------------------------------- focal
    def focal_length(self, image_width):
        if self.focal_px_override:
            return float(self.focal_px_override)
        return focal_length_from_fov(image_width, self.settings["camera"]["horizontal_fov_deg"])

    def set_focal_length(self, focal_px):
        self.focal_px_override = float(focal_px)

    # -------------------------------------------------------------- masks
    def color_mask(self, hsv, color):
        mask = None
        for lower, upper in self.colors[color]:
            part = cv2.inRange(hsv, lower, upper)
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        return mask

    # -------------------------------------------------------------- shape
    @staticmethod
    def classify_shape(contour, area):
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or area <= 0:
            return "unknown", 0.0
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        vertices = len(cv2.approxPolyDP(contour, 0.035 * perimeter, True))

        (_, (rw, rh), _) = cv2.minAreaRect(contour)
        rect_fill = area / (rw * rh) if rw * rh > 0 else 0.0
        aspect = max(rw, rh) / min(rw, rh) if min(rw, rh) > 0 else 99.0

        ellipse_fill = 0.0
        if len(contour) >= 5:
            (_, (ea, eb), _) = cv2.fitEllipse(contour)
            ellipse_area = math.pi * ea * eb / 4.0
            ellipse_fill = area / ellipse_area if ellipse_area > 0 else 0.0

        tri_area, _ = cv2.minEnclosingTriangle(contour)
        tri_fill = area / tri_area if tri_area > 0 else 0.0

        if vertices == 3 or (tri_fill > 0.88 and rect_fill < 0.7 and vertices <= 4):
            return "triangle", circularity
        if vertices == 4 and rect_fill > 0.80:
            return ("square" if aspect <= 1.25 else "rectangle"), circularity
        if vertices >= 6 and circularity > 0.75 and 0.93 <= ellipse_fill <= 1.07 and rect_fill < 0.88:
            return "circle", circularity
        if rect_fill > 0.90 and vertices <= 6:
            return ("square" if aspect <= 1.25 else "rectangle"), circularity
        return "unknown", circularity

    # -------------------------------------------------------------- detect
    def _prepare(self, image_bgr):
        k = self.settings["detection"]["blur_kernel"]
        if k > 1:
            image_bgr = cv2.GaussianBlur(image_bgr, (k, k), 0)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    def _refine(self, frame_bgr, color, approx_bbox, approx_center):
        """Re-extract a candidate on a full-resolution ROI for accurate shape/size."""
        full_h, full_w = frame_bgr.shape[:2]
        x, y, w, h = approx_bbox
        pad = int(max(w, h) * 0.3) + 8
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(full_w, x + w + pad), min(full_h, y + h + pad)
        mask = self.color_mask(self._prepare(frame_bgr[y0:y1, x0:x1]), color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        point = (float(approx_center[0] - x0), float(approx_center[1] - y0))
        inside = [c for c in contours if cv2.pointPolygonTest(c, point, False) >= 0]
        contour = max(inside or contours, key=cv2.contourArea)
        return contour + np.array([[x0, y0]], dtype=contour.dtype)

    def detect(self, frame_bgr):
        """Return (detections, masks) for a BGR frame. Never raises on odd frames."""
        if (frame_bgr is None or not hasattr(frame_bgr, "ndim") or frame_bgr.ndim != 3
                or frame_bgr.shape[2] != 3 or frame_bgr.shape[0] < 16 or frame_bgr.shape[1] < 16):
            return [], {}
        det = self.settings["detection"]
        full_h, full_w = frame_bgr.shape[:2]

        if self.settings["camera"]["swap_red_blue"]:
            frame_bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
        frame_bgr = self.normalize_lighting(frame_bgr)

        # Stage 1: find candidates quickly on a downscaled frame.
        scale = min(1.0, det["process_width"] / float(full_w))
        small = frame_bgr if scale == 1.0 else cv2.resize(
            frame_bgr, (int(round(full_w * scale)), int(round(full_h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        hsv = self._prepare(small)
        inv = 1.0 / scale
        max_area_full = det["max_area_ratio"] * full_h * full_w
        min_area_full = det["min_area_px"] * inv * inv
        border = 2

        tgt = self.settings["target"]
        focal_px = self.focal_length(full_w)
        detections, masks = [], {}

        for color in self.colors:
            mask = self.color_mask(hsv, color)
            masks[color] = mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for coarse in contours:
                if cv2.contourArea(coarse) < det["min_area_px"] * 0.5:
                    continue
                cm = cv2.moments(coarse)
                if cm["m00"] <= 0:
                    continue
                bx, by, bw, bh = cv2.boundingRect(coarse)
                approx_bbox = (int(bx * inv), int(by * inv), int(math.ceil(bw * inv)), int(math.ceil(bh * inv)))
                approx_center = (cm["m10"] / cm["m00"] * inv, cm["m01"] / cm["m00"] * inv)

                # Stage 2: accurate contour at full resolution.
                contour = self._refine(frame_bgr, color, approx_bbox, approx_center) if scale < 1.0 else coarse
                if contour is None:
                    continue
                area = cv2.contourArea(contour)
                if area < min_area_full or area > max_area_full:
                    continue
                hull_area = cv2.contourArea(cv2.convexHull(contour))
                if hull_area <= 0 or area / hull_area < det["min_solidity"]:
                    continue
                moments = cv2.moments(contour)
                if moments["m00"] <= 0:
                    continue

                shape, circularity = self.classify_shape(contour, area)
                x, y, w, h = cv2.boundingRect(contour)
                clipped = x <= border or y <= border or x + w >= full_w - border or y + h >= full_h - border
                (_, (rect_w, rect_h), rect_angle) = cv2.minAreaRect(contour)  # noqa: E501
                if shape == "circle" and len(contour) >= 5:
                    size_px = max(cv2.fitEllipse(contour)[1])
                else:
                    size_px = max(rect_w, rect_h)
                # Angle of the long side away from vertical (0 = standing, 90 = lying flat).
                # Taken from the box corners: minAreaRect's own angle is ambiguous.
                box = cv2.boxPoints(((0, 0), (rect_w, rect_h), rect_angle))
                edges = [(box[i], box[(i + 1) % 4]) for i in range(4)]
                (p0, p1) = max(edges, key=lambda e: (e[1][0] - e[0][0]) ** 2 + (e[1][1] - e[0][1]) ** 2)
                long_axis = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
                short, long_side = min(rect_w, rect_h), max(rect_w, rect_h)
                # Only an elongated shape has a meaningful tilt; a square card that
                # falls over looks squashed (elongated) from the robot's view.
                elongated = long_side > 1.15 * max(short, 1e-6)
                tilt = abs((long_axis % 180.0) - 90.0) if elongated else 0.0
                orientation = ("" if not elongated else VERTICAL if tilt < 45.0 else HORIZONTAL)

                detection = Detection(
                    color=color,
                    shape=shape,
                    center=(moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]),
                    bbox=(x, y, w, h),
                    contour=contour,
                    area_px=area,
                    size_px=size_px,
                    circularity=circularity,
                    clipped=clipped,
                    tilt_deg=tilt,
                    orientation=orientation,
                )

                color_ok = color == tgt["color"]
                shape_ok = shape_matches(tgt["shape"], shape, orientation)
                if color_ok and shape_ok:
                    detection.distance_m = estimate_distance(detection.size_px, tgt["size_m"], focal_px)
                    lying = shape in ("rectangle", "square") and tilt > tgt["upright_tolerance_deg"]
                    if detection.distance_m is None:
                        detection.reject_reason = "no distance"
                    elif not tgt["min_distance_m"] <= detection.distance_m <= tgt["max_distance_m"]:
                        # A wall or door fills the frame -> "distance" comes out far too small.
                        detection.reject_reason = "out of range"
                    elif tgt["require_fully_visible"] and clipped:
                        detection.reject_reason = "cut off by frame"
                    elif tgt["upright_only"] and lying:
                        detection.reject_reason = "lying down"
                    else:
                        detection.is_target = True
                elif color_ok:
                    detection.reject_reason = "wrong shape"
                else:
                    detection.reject_reason = "wrong color"
                detections.append(detection)

        detections.sort(key=lambda d: d.area_px, reverse=True)
        return detections, masks


# ==============================================================================
# 4. Temporal tracker
# ==============================================================================
@dataclass
class Track:
    detection: Detection
    center: Tuple[float, float]
    distance_m: Optional[float]
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    track_id: int = 0
    history: List[Tuple[float, float]] = field(default_factory=list)


class TargetTracker:
    """Keeps one stable target across frames (no flicker, no target hopping)."""

    def __init__(self, settings):
        self.cfg = settings["tracking"]
        self.track = None  # type: Optional[Track]
        self._ids = itertools.count(1)

    def reset(self):
        self.track = None

    def _select(self, candidates, frame_shape):
        h, w = frame_shape[:2]
        if self.track is not None:
            px, py = self.track.center
            limit = max(self.track.detection.size_px, 20.0) * self.cfg["max_jump_ratio"]
            near = [d for d in candidates if math.hypot(d.center[0] - px, d.center[1] - py) <= limit]
            if near:
                return min(near, key=lambda d: math.hypot(d.center[0] - px, d.center[1] - py)), True
            return None, False

        def score(d):
            off_center = math.hypot(d.center[0] - w / 2.0, d.center[1] - h / 2.0) / math.hypot(w / 2.0, h / 2.0)
            return d.area_px * (1.0 - 0.5 * off_center) * (0.5 if d.clipped else 1.0)

        return (max(candidates, key=score), False) if candidates else (None, False)

    def update(self, detections, frame_shape):
        candidates = [d for d in detections if d.is_target]
        chosen, continued = self._select(candidates, frame_shape)

        if chosen is None:
            if self.track is not None:
                self.track.misses += 1
                if self.track.misses > self.cfg["lost_frames"]:
                    self.track = None
                    # A different target may be visible: start tracking it next frame.
                    if candidates:
                        chosen, continued = self._select(candidates, frame_shape)
            if chosen is None:
                return self.track

        if self.track is None or not continued:
            self.track = Track(detection=chosen, center=chosen.center, distance_m=chosen.distance_m,
                               track_id=next(self._ids))
        else:
            a = self.cfg["smoothing"]
            t = self.track
            t.center = (a * chosen.center[0] + (1 - a) * t.center[0],
                        a * chosen.center[1] + (1 - a) * t.center[1])
            if chosen.distance_m is not None:
                t.distance_m = (chosen.distance_m if t.distance_m is None
                                else a * chosen.distance_m + (1 - a) * t.distance_m)
            t.detection = chosen
            t.hits += 1
            t.misses = 0

        self.track.history = (self.track.history + [self.track.center])[-30:]
        if self.track.hits >= self.cfg["confirm_frames"]:
            self.track.confirmed = True
        return self.track

    @property
    def visible(self):
        """True when the confirmed target was seen in the latest frame."""
        return self.track is not None and self.track.confirmed and self.track.misses == 0


# ==============================================================================
# 5. Drawing
# ==============================================================================
DRAW_COLORS = {
    "red": (60, 60, 255), "green": (80, 220, 80), "blue": (255, 140, 40),
    "yellow": (40, 230, 240),
}


def _text(frame, text, org, color=(255, 255, 255), scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame, detections, track=None):
    """Draw every detection; the tracked target is highlighted."""
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 28, 1)

    for d in detections:
        base = DRAW_COLORS.get(d.color, (200, 200, 200))
        thickness = 3 if d.is_target else 1
        cv2.drawContours(frame, [d.contour], -1, (0, 255, 0) if d.is_target else base, thickness)
        x, y, bw, bh = d.bbox
        label = f"{d.color} {d.shape}"
        if d.orientation:
            label += "-v" if d.orientation == VERTICAL else "-h"
        if d.distance_m is not None:
            label += f" {d.distance_m:.2f}m"
        if d.clipped:
            label += " (edge)"
        if d.reject_reason:
            label += f" x {d.reject_reason}"
        _text(frame, label, (x, max(18, y - 8)), (0, 255, 0) if d.is_target else base, 0.5, 1)

    if track is not None:
        cx, cy = int(track.center[0]), int(track.center[1])
        color = (0, 255, 0) if track.confirmed else (0, 200, 255)
        radius = int(max(12, track.detection.size_px * 0.6))
        cv2.circle(frame, (cx, cy), radius, color, 2)
        cv2.line(frame, (cx - radius - 8, cy), (cx + radius + 8, cy), color, 1)
        cv2.line(frame, (cx, cy - radius - 8), (cx, cy + radius + 8), color, 1)
        pts = [(int(px), int(py)) for px, py in track.history]
        for p0, p1 in zip(pts, pts[1:]):
            cv2.line(frame, p0, p1, color, 1)


# ==============================================================================
# 6. Stand-alone preview (no robot needed)
# ==============================================================================
def _open_source(source):
    if source.isdigit():
        return cv2.VideoCapture(int(source)), False
    image = cv2.imread(source)
    if image is not None:
        return image, True
    return cv2.VideoCapture(source), False


def main():
    parser = argparse.ArgumentParser(description="Preview colour/shape/distance detection.")
    parser.add_argument("--source", default="0", help="webcam index, image or video path")
    parser.add_argument("--color", help="override target.color")
    parser.add_argument("--shape", choices=TARGET_SHAPES, help="override target.shape")
    parser.add_argument("--output", help="save the annotated image (image sources only)")
    args = parser.parse_args()

    try:
        from config_loader import load_config
        config = load_config()
    except Exception as error:  # config is optional for the preview
        print(f"Using default detection settings ({error})")
        config = {}
    config.setdefault("target_shooting", {}).setdefault("target", {})
    if args.color:
        config["target_shooting"]["target"]["color"] = args.color
    if args.shape:
        config["target_shooting"]["target"]["shape"] = args.shape

    try:
        settings = build_detection_settings(config)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    detector = ColorShapeDetector(settings)
    tracker = TargetTracker(settings)

    source, is_image = _open_source(args.source)
    if is_image:
        detections, _ = detector.detect(source)
        draw_detections(source, detections)
        for d in detections:
            dist = f"{d.distance_m:.2f} m" if d.distance_m is not None else "-"
            print(f"{d.color:>7} {d.shape:<9} center=({d.center[0]:.0f},{d.center[1]:.0f}) "
                  f"dist={dist:<8} target={d.is_target} {d.reject_reason}")
        if args.output:
            cv2.imwrite(args.output, source)
        else:
            cv2.imshow("Target detection", source)
            cv2.waitKey(0)
        return 0

    if not source.isOpened():
        print(f"Cannot open source {args.source}", file=sys.stderr)
        return 1
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                break
            detections, _ = detector.detect(frame)
            track = tracker.update(detections, frame.shape)
            draw_detections(frame, detections, track)
            cv2.imshow("Target detection (Esc to quit)", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        source.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
