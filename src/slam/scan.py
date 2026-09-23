"""Turn raw ToF readings into clean beams, and match a scan to the map.

Noise reduction, in order:
1. range gate (tof_min_m / tof_max_m) and scale/bias calibration,
2. per-angle-bin median with MAD outlier rejection,
3. angular spike removal (a beam that disagrees with two agreeing neighbours),
4. log-odds fusion in the grid (repeated scans average out what is left),
5. isolated-cell removal when the map is shown or scored.
"""

import math

import cv2
import numpy as np


def correct_range(raw_m, params):
    """Apply the wall-calibration scale/bias and the lens offset."""
    return raw_m * params["tof_scale"] + params["tof_bias_m"] + params["tof_offset_m"]


def bin_samples(samples, params):
    """samples: iterable of (robot_angle_rad, raw_range_m).

    Returns (angles, ranges, hits) numpy arrays in the robot frame, where the
    range is measured from the gimbal yaw axis.
    """
    step = math.radians(params["scan_step_deg"])
    bins = {}
    no_return = {}
    for ang, rng in samples:
        key = int(round(ang / step))
        if rng is None or not math.isfinite(rng) or rng > params["tof_max_m"]:
            no_return.setdefault(key, []).append(ang)
            continue
        if rng < params["tof_min_m"]:
            continue
        bins.setdefault(key, []).append((ang, rng))

    # each beam keeps the mean angle of its own readings, not the bin centre,
    # so the gimbal's real angle is used (no rounding bias)
    angles, ranges, hits = [], [], []
    for key in sorted(set(bins) | set(no_return)):
        got = np.asarray(bins.get(key, []), float).reshape(-1, 2)
        misses = no_return.get(key, [])
        if len(got) == 0 or len(misses) > len(got):
            if misses and params["no_hit_free_m"] > 0:
                angles.append(float(np.mean(misses)))
                ranges.append(params["no_hit_free_m"])
                hits.append(False)
            continue
        values = got[:, 1]
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med))) * 1.4826
        keep = np.abs(values - med) <= max(params["outlier_mad_k"] * mad, params["outlier_abs_m"])
        if keep.sum() < params["min_bin_samples"]:
            continue
        angles.append(float(np.mean(got[keep, 0])))
        ranges.append(correct_range(float(np.median(values[keep])), params))
        hits.append(True)
    angles, ranges, hits = np.asarray(angles), np.asarray(ranges), np.asarray(hits, bool)
    angles, ranges, hits = remove_spikes(angles, ranges, hits, params["spike_tol_m"], step)
    return remove_isolated_hits(angles, ranges, hits, params.get("isolated_join_m", 0.0), step)


def remove_isolated_hits(angles, ranges, hits, join, step):
    """Drop wall points with no other wall point nearby among the two beams on
    each side. The ToF's wide beam makes such lone points at wall ends and
    corners; real walls always give runs of neighbouring points."""
    n = len(angles)
    if join <= 0 or n < 3:
        return angles, ranges, hits
    x, y = ranges * np.cos(angles), ranges * np.sin(angles)
    keep = np.ones(n, bool)
    for i in range(n):
        if not hits[i]:
            continue
        limit = join + ranges[i] * step * 1.5
        ok = False
        for d in (-2, -1, 1, 2):
            j = (i + d) % n
            if j != i and hits[j] and abs(_wrap(angles[j] - angles[i])) < 2.6 * step and \
                    math.hypot(x[j] - x[i], y[j] - y[i]) < limit * abs(d):
                ok = True
                break
        keep[i] = ok
    return angles[keep], ranges[keep], hits[keep]


def remove_spikes(angles, ranges, hits, tol, step):
    """Drop a hit whose neighbours agree with each other but not with it."""
    n = len(angles)
    if n < 3:
        return angles, ranges, hits
    keep = np.ones(n, bool)
    for i in range(n):
        j, k = (i - 1) % n, (i + 1) % n
        # neighbours must be adjacent bins (the scan can have gaps)
        if abs(_wrap(angles[j] - angles[i])) > 1.5 * step or abs(_wrap(angles[k] - angles[i])) > 1.5 * step:
            continue
        if not (hits[j] and hits[k]):
            continue
        a, b = ranges[j], ranges[k]
        if not hits[i]:
            # a lone "nothing there" between two nearby walls is a dropout
            if max(a, b) < ranges[i]:
                keep[i] = False
        elif abs(a - b) < tol and abs(ranges[i] - (a + b) / 2) > tol:
            keep[i] = False
    return angles[keep], ranges[keep], hits[keep]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def sensor_origin(pose, params):
    x, y, th = pose
    off = params["gimbal_offset_x_m"]
    return x + off * math.cos(th), y + off * math.sin(th)


def beams_to_world(pose, angles, ranges, params):
    """World endpoints of beams given in the robot frame."""
    ox, oy = sensor_origin(pose, params)
    wa = angles + pose[2]
    return ox + ranges * np.cos(wa), oy + ranges * np.sin(wa), wa


class ScanMatcher:
    """Correlative scan matcher on a likelihood field.

    It tries every pose in a small window around the odometry guess and keeps
    the one whose wall points land closest to walls already in the map.
    """

    def __init__(self):
        self.last_score = None
        self.last_gain = 0.0

    def match(self, grid, pose, angles, ranges, hits, params):
        self.last_gain = 0.0
        occ = grid.occupied_mask(params["occ_prob"])
        pts = hits & (ranges < params["tof_max_m"])
        if occ.sum() < params["match_min_points"] or pts.sum() < params["match_min_points"]:
            return pose, False
        dist = cv2.distanceTransform((~occ).astype(np.uint8), cv2.DIST_L2, 3) * grid.res
        sigma = params["match_sigma_m"]
        field = np.exp(-(dist ** 2) / (2 * sigma ** 2)).astype(np.float32)

        a = angles[pts]
        r = ranges[pts]
        off = params["gimbal_offset_x_m"]
        # robot-frame points
        px = off + r * np.cos(a)
        py = r * np.sin(a)

        win = params["match_window_m"]
        wdeg = params["match_window_deg"]
        best = (self._score(field, grid, pose, px, py), pose)  # prior is 0 at the guess
        base_score = best[0]
        # coarse then fine search
        for xy_step, deg_step, span_xy, span_deg in (
                (max(grid.res, win / 4), max(1.0, wdeg / 4), win, wdeg),
                (grid.res / 2, 0.5, max(grid.res, win / 4), max(1.0, wdeg / 4))):
            cx, cy, cth = best[1]
            dths = np.radians(np.arange(-span_deg, span_deg + 1e-9, deg_step)) if span_deg > 0 else [0.0]
            ds = np.arange(-span_xy, span_xy + 1e-9, xy_step) if span_xy > 0 else np.array([0.0])
            for dth in dths:
                th = cth + dth
                c, s = math.cos(th), math.sin(th)
                wx = c * px - s * py
                wy = s * px + c * py
                for dx in ds:
                    for dy in ds:
                        cand = (cx + dx, cy + dy, th)
                        score = self._score_rot(field, grid, cand, wx, wy) - self._prior(cand, pose, params)
                        if score > best[0]:
                            best = (score, cand)
        refined = self._refine(field, grid, best[1], px, py)
        if refined is not None:
            score = self._score(field, grid, refined, px, py) - self._prior(refined, pose, params)
            if score >= best[0]:
                best = (score, refined)
        self.last_score = best[0]
        self.last_gain = best[0] - base_score
        return best[1], best[1] != pose

    @staticmethod
    def _sample(field, grid, x, y):
        """Bilinear field value and gradient (per metre) at world points."""
        fx = (x - grid.origin_x) / grid.res - 0.5
        fy = (y - grid.origin_y) / grid.res - 0.5
        c0 = np.floor(fx).astype(int)
        r0 = np.floor(fy).astype(int)
        ok = (r0 >= 0) & (r0 < grid.rows - 1) & (c0 >= 0) & (c0 < grid.cols - 1)
        r0c, c0c = np.where(ok, r0, 0), np.where(ok, c0, 0)
        ax, ay = fx - c0, fy - r0
        f00, f01 = field[r0c, c0c], field[r0c, c0c + 1]
        f10, f11 = field[r0c + 1, c0c], field[r0c + 1, c0c + 1]
        val = (f00 * (1 - ax) + f01 * ax) * (1 - ay) + (f10 * (1 - ax) + f11 * ax) * ay
        gx = ((f01 - f00) * (1 - ay) + (f11 - f10) * ay) / grid.res
        gy = ((f10 - f00) * (1 - ax) + (f11 - f01) * ax) / grid.res
        return np.where(ok, val, 0), np.where(ok, gx, 0), np.where(ok, gy, 0)

    def _refine(self, field, grid, pose, px, py, iterations=8):
        """Gauss-Newton on sum (1 - field)^2, for sub-cell accuracy."""
        x, y, th = pose
        start = pose
        for _ in range(iterations):
            c, s = math.cos(th), math.sin(th)
            wx, wy = x + c * px - s * py, y + s * px + c * py
            val, gx, gy = self._sample(field, grid, wx, wy)
            dth_x = -s * px - c * py
            dth_y = c * px - s * py
            J = np.stack([gx, gy, gx * dth_x + gy * dth_y], axis=1)
            r = 1.0 - val
            H = J.T @ J + np.diag([1e-3, 1e-3, 1e-3])
            g = J.T @ r
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                return None
            step[:2] = np.clip(step[:2], -grid.res, grid.res)
            step[2] = float(np.clip(step[2], -math.radians(1.0), math.radians(1.0)))
            x, y, th = x + step[0], y + step[1], th + step[2]
            if abs(step[0]) + abs(step[1]) < 1e-4 and abs(step[2]) < 1e-4:
                break
        if math.hypot(x - start[0], y - start[1]) > 3 * grid.res:
            return None
        return (x, y, _wrap(th))

    @staticmethod
    def _prior(cand, guess, params):
        """Gaussian pull toward the odometry/IMU guess (sigma 10 cm, 2 deg).

        Close walls pin the pose only to about a cell, so without this small,
        ambiguous corrections would random-walk the heading.
        """
        dxy = math.hypot(cand[0] - guess[0], cand[1] - guess[1]) / 0.10
        dth = abs(_wrap(cand[2] - guess[2])) / math.radians(2.0)
        return params["match_prior_weight"] * (dxy * dxy + dth * dth)

    @staticmethod
    def _score_rot(field, grid, pose, wx, wy):
        r, c = grid.world_to_cell(pose[0] + wx, pose[1] + wy)
        ok = grid.inside(r, c)
        if not ok.any():
            return 0.0
        return float(field[r[ok], c[ok]].sum()) / len(wx)

    def _score(self, field, grid, pose, px, py):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        return self._score_rot(field, grid, pose, c * px - s * py, s * px + c * py)
