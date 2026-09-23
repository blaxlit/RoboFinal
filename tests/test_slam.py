"""Offline tests for the SLAM package (no robot or robomaster SDK needed)."""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.request

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from slam import params as params_mod  # noqa: E402
from slam.calibration import Calibrator, best_shift, profile  # noqa: E402
from slam.driver import Driver  # noqa: E402
from slam.evaluation import evaluate, load_ground_truth, rasterize_gt  # noqa: E402
from slam.grid import FREE, OCCUPIED, UNKNOWN, OccupancyGrid  # noqa: E402
from slam.io_sim import SimRobotIO  # noqa: E402
from slam.planner import choose_frontier_goal  # noqa: E402
from slam.scan import ScanMatcher, bin_samples  # noqa: E402

GT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "slam", "ground_truth_example.json")


def base_params(**over):
    p = params_mod.defaults()
    p.update(over)
    return p


def make_grid(p):
    return OccupancyGrid(p["resolution_m"], p["width_m"], p["height_m"], p["origin_x_m"], p["origin_y_m"])


def room_scan(pose, half=1.0, step_deg=2.0, offset=0.0):
    """Beams from ``pose`` in a square room [-half, half]^2 (robot frame angles)."""
    x, y, th = pose
    angles, ranges = [], []
    for a in np.radians(np.arange(-180, 180, step_deg)):
        wa = th + a
        dx, dy = math.cos(wa), math.sin(wa)
        ts = []
        if abs(dx) > 1e-9:
            ts += [(half - x) / dx, (-half - x) / dx]
        if abs(dy) > 1e-9:
            ts += [(half - y) / dy, (-half - y) / dy]
        angles.append(a)
        ranges.append(min(t for t in ts if t > 0) + offset)
    return np.array(angles), np.array(ranges), np.ones(len(angles), bool)


class GridTest(unittest.TestCase):
    def test_scan_marks_walls_and_free_space(self):
        p = base_params()
        g = make_grid(p)
        a, r, h = room_scan((0, 0, 0))
        g.integrate((0, 0), a, r, h, p["l_occ"], p["l_free"], p["l_clamp"], max_gap=math.radians(3.5))
        cls = g.classify(p["occ_prob"], p["free_prob"])
        self.assertEqual(cls[g.world_to_cell(0.0, 0.0)], FREE)
        self.assertEqual(cls[g.world_to_cell(0.5, 0.5)], FREE)
        self.assertEqual(cls[g.world_to_cell(1.0, 0.0)], OCCUPIED)
        self.assertEqual(cls[g.world_to_cell(1.5, 0.0)], UNKNOWN)

    def test_outlier_beam_cannot_erase_a_known_wall(self):
        p = base_params()
        g = make_grid(p)
        a, r, h = room_scan((0, 0, 0))
        for _ in range(3):
            g.integrate((0, 0), a, r, h, p["l_occ"], p["l_free"], p["l_clamp"])
        wall = g.world_to_cell(1.0, 0.0)
        before = g.logodds[wall]
        g.integrate((0, 0), np.array([0.0]), np.array([2.5]), np.array([True]), p["l_occ"], p["l_free"], p["l_clamp"])
        self.assertEqual(g.logodds[wall], before)


class FilterTest(unittest.TestCase):
    def test_median_rejects_outliers_and_spikes(self):
        p = base_params(scan_step_deg=2.0, tof_offset_m=0.0)
        samples = []
        for deg in range(-40, 42, 2):
            for k in range(5):
                samples.append((math.radians(deg), 1.0 + (0.5 if k == 0 else 0.0)))  # one bad reading per bin
        samples = [(a, 3.0 if abs(a - math.radians(10)) < 1e-6 else r) for a, r in samples]  # a whole bad beam
        angles, ranges, hits = bin_samples(samples, p)
        self.assertTrue(np.allclose(ranges, 1.0))
        self.assertNotIn(round(math.radians(10), 4), [round(v, 4) for v in angles])

    def test_dropout_between_walls_is_removed(self):
        p = base_params(scan_step_deg=2.0, tof_offset_m=0.0)
        samples = [(math.radians(d), 0.5) for d in range(-10, 12, 2)]
        samples[5] = (samples[5][0], math.inf)
        angles, ranges, hits = bin_samples(samples, p)
        self.assertTrue(hits.all())


class EvaluationTest(unittest.TestCase):
    def test_perfect_and_empty_maps(self):
        p = base_params(wall_tolerance_cells=0)
        gt = load_ground_truth(GT_PATH)
        g = make_grid(p)
        truth, arena = rasterize_gt(gt, g, p)
        perfect = np.where(arena, truth, UNKNOWN).astype(np.uint8)
        m = evaluate(perfect, g, p, gt)
        self.assertEqual(m["accuracy_pct"], 100.0)
        self.assertEqual(m["coverage_pct"], 100.0)
        empty = np.zeros_like(perfect)
        m = evaluate(empty, g, p, gt)
        self.assertEqual(m["accuracy_pct"], 0.0)
        self.assertEqual(m["coverage_pct"], 0.0)

    def test_tolerance_accepts_walls_one_cell_off(self):
        p = base_params(wall_tolerance_cells=1)
        gt = load_ground_truth(GT_PATH)
        g = make_grid(p)
        truth, arena = rasterize_gt(gt, g, p)
        shifted = np.roll(np.where(arena, truth, UNKNOWN).astype(np.uint8), 1, axis=1)
        self.assertGreater(evaluate(shifted, g, p, gt)["accuracy_pct"], evaluate(shifted, g, dict(p, wall_tolerance_cells=0), gt)["accuracy_pct"])


class PlannerTest(unittest.TestCase):
    def test_goes_toward_open_side(self):
        p = base_params()
        g = make_grid(p)
        cls = np.full((g.rows, g.cols), UNKNOWN, np.uint8)
        r0, c0 = g.world_to_cell(-0.3, -0.3)
        r1, c1 = g.world_to_cell(1.5, 0.3)
        cls[r0:r1 + 1, c0:c1 + 1] = FREE
        cls[r0 - 1, c0:c1 + 1] = OCCUPIED
        cls[r1 + 1, c0:c1 + 1] = OCCUPIED
        cls[r0:r1 + 1, c0 - 1] = OCCUPIED  # corridor closed behind, open ahead
        info = choose_frontier_goal(g, cls, (0.0, 0.0, 0.0), p)
        self.assertIsNotNone(info["goal"])
        gx, gy = g.cell_to_world(*info["goal"])
        self.assertGreater(gx, 0.8)


class MatcherTest(unittest.TestCase):
    def test_recovers_pose_offset(self):
        p = base_params()
        g = make_grid(p)
        # walls between cell centres, so cell rounding does not bias the map
        a, r, h = room_scan((0, 0, 0), half=1.225)
        for _ in range(3):
            g.integrate((0, 0), a, r, h, p["l_occ"], p["l_free"], p["l_clamp"], max_gap=math.radians(3.5))
        true = (0.2, -0.1, math.radians(3.0))
        a, r, h = room_scan(true, half=1.225)
        guess = (0.28, -0.03, 0.0)
        est, _ = ScanMatcher().match(g, guess, a, r, h, p)
        self.assertLess(math.hypot(est[0] - true[0], est[1] - true[1]), 0.03)
        self.assertLess(abs(est[2] - true[2]), math.radians(1.0))


class CalibrationTest(unittest.TestCase):
    def test_profile_shift(self):
        p = base_params(tof_max_m=5.0)
        a, r, _ = room_scan((0.3, 0.1, 0.0), half=1.5, step_deg=1.0)
        a2, r2, _ = room_scan((0.3, 0.1, math.radians(25)), half=1.5, step_deg=1.0)
        shift, _ = best_shift(profile(list(zip(a, r)), p), profile(list(zip(a2, r2)), p))
        self.assertAlmostEqual(shift, 25.0, delta=1.0)

    def test_detects_flipped_yaw_sign_in_simulator(self):
        gt = load_ground_truth(GT_PATH)
        io = SimRobotIO(gt, start=(0.9, 1.5, 0.0), time_scale=60, flip={"yaw": True})
        try:
            p = base_params()
            d = Driver(io, p)
            res = Calibrator(d, p, lambda m: None).turn_signs()
            self.assertEqual(p["chassis_yaw_sign"], 1)
            self.assertEqual(p["cmd_z_sign"], 1)
            self.assertIn("turn_scan_deg", res)
        finally:
            io.close()


class MissionTest(unittest.TestCase):
    def test_short_simulated_mission_writes_results(self):
        from slam.console import serve
        from slam.explorer import Explorer
        out = tempfile.mkdtemp()
        gt = load_ground_truth(GT_PATH)
        p = base_params(auto_calibrate=False, max_iterations=3)
        io = SimRobotIO(gt, start=(p["gt_start_x"], p["gt_start_y"], 0.0), time_scale=60)
        ex = Explorer(io, p, out, gt, log_print=False)
        server, url = serve(ex, "127.0.0.1", 0)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        try:
            ex.command("start")
            end = time.time() + 180
            while ex.state != "DONE" and time.time() < end:
                time.sleep(0.2)
            self.assertEqual(ex.state, "DONE")
            for name in ("map.png", "trajectory.csv", "report.json", "report.md", "comparison.png",
                         "log_events.csv", "log_scans_raw.csv", "map_grid.csv"):
                self.assertTrue(os.path.exists(os.path.join(ex.run_dir, name)), name)
            with open(os.path.join(ex.run_dir, "report.json")) as f:
                rep = json.load(f)
            self.assertEqual(rep["start"]["x"], 0.0)
            self.assertIn("gt", rep["end"])
            self.assertGreater(rep["metrics"]["coverage_pct"], 5)
            self.assertLess(rep["end_error_m"], 0.15)
            with urllib.request.urlopen(url + "api/state") as resp:
                state = json.load(resp)
            self.assertEqual(state["state"], "DONE")
            req = urllib.request.Request(url + "api/params", data=json.dumps({"changes": {"scan_speed_dps": 30}}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as resp:
                json.load(resp)
            self.assertEqual(ex.p["scan_speed_dps"], 30.0)
        finally:
            server.shutdown()
            ex.shutdown()
            io.close()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
