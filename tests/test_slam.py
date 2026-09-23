"""Offline tests for the SLAM package (no robot or robomaster SDK needed)."""

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest

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

    def test_lone_point_is_removed(self):
        p = base_params(scan_step_deg=2.0, tof_offset_m=0.0)
        samples = [(math.radians(d), 1.0) for d in range(-20, 22, 2)]
        samples += [(math.radians(60), 0.7)]  # a stray point on its own
        angles, ranges, hits = bin_samples(samples, p)
        self.assertNotIn(60, [round(math.degrees(a)) for a in angles])
        self.assertEqual(len(angles), 21)

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


class GridModelTest(unittest.TestCase):
    def maze_points(self, theta_deg=-5.5, cell=0.63, offset=(0.2, 0.25)):
        """Oriented wall points of a 5x4-cell maze, rotated and shifted."""
        from slam import gridmodel as G
        segs = [(0, 0, 5, 0), (5, 0, 5, 4), (5, 4, 0, 4), (0, 4, 0, 0), (1, 0, 1, 2), (2, 1, 3, 1), (3, 1, 3, 3),
                (4, 2, 4, 4), (1, 3, 2, 3)]
        th = math.radians(theta_deg)
        pts, ang = [], []
        rng = np.random.default_rng(0)
        for x1, y1, x2, y2 in segs:
            n = int(math.hypot(x2 - x1, y2 - y1) * cell / 0.02)
            t = np.linspace(0.05, 0.95, n)
            u = (x1 + (x2 - x1) * t) * cell + offset[0] + rng.normal(0, 0.008, n)
            v = (y1 + (y2 - y1) * t) * cell + offset[1] + rng.normal(0, 0.008, n)
            xs, ys = math.cos(th) * u - math.sin(th) * v, math.sin(th) * u + math.cos(th) * v
            p, a = G.oriented_points(xs, ys)
            pts.append(p)
            ang.append(a)
        return np.concatenate(pts), np.concatenate(ang)

    def test_detects_angle_cell_and_offset(self):
        from slam import gridmodel as G
        pts, ang = self.maze_points()
        history, m = [], None
        for frac in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):  # walls seen bit by bit, as during a mission
            n = int(len(pts) * frac)
            m = G.detect(pts[:n], ang[:n], base_params(), m, history)
            history.append(m)
        self.assertTrue(m.cell_trusted)
        self.assertAlmostEqual(m.cell, 0.63, delta=0.01)
        self.assertAlmostEqual(math.degrees(m.theta), -5.5, delta=0.5)
        self.assertAlmostEqual(m.ou % m.cell, 0.2, delta=0.02)

    def test_snap_pose_removes_small_heading_and_shift(self):
        from slam import gridmodel as G
        pts, ang = self.maze_points(theta_deg=0.0, offset=(0.0, 0.0))
        m = G.GridModel(0.0, 0.63, 0.0, 0.0, 1.0, len(pts), True, True)
        # the same walls seen from a pose that is 3 deg and 4 cm off
        th = math.radians(3)
        c, s = math.cos(th), math.sin(th)
        moved = np.stack([c * pts[:, 0] - s * pts[:, 1] + 0.04, s * pts[:, 0] + c * pts[:, 1]], axis=1)
        dx, dy, dth = G.snap_pose(m, (0.04, 0.0, th), moved, ang + th)
        self.assertAlmostEqual(math.degrees(dth), -3.0, delta=0.5)
        self.assertLess(abs(0.04 + dx), 0.02)

    def test_edges_fill_gaps_and_drop_blobs(self):
        from slam import gridmodel as G
        p = base_params()
        g = make_grid(p)
        cls = np.full((g.rows, g.cols), UNKNOWN, np.uint8)
        # one 0.6 m cell seen free, walls on three sides, the right wall only half seen
        r0, c0 = g.world_to_cell(0.0, 0.0)
        n = 12
        cls[r0:r0 + n, c0:c0 + n] = FREE
        cls[r0 - 1, c0:c0 + n] = OCCUPIED
        cls[r0 + n, c0:c0 + n] = OCCUPIED
        cls[r0:r0 + n, c0 - 1] = OCCUPIED
        cls[r0:r0 + n // 2, c0 + n] = OCCUPIED
        cls[r0 + 5, c0 + 5] = OCCUPIED  # a blob in the middle
        model = G.GridModel(0.0, 0.6, -0.025, -0.025, 1.0, 500, True)
        e = G.classify_edges(model, g, cls, p)
        a, b = 0 - e.i0, 0 - e.j0
        self.assertEqual(e.vert[a, b], G.EDGE_WALL)          # left
        self.assertEqual(e.vert[a + 1, b], G.EDGE_WALL)      # right, half seen -> filled
        self.assertEqual(e.horz[a, b], G.EDGE_WALL)          # bottom
        out = G.render(model, e, g, p)
        self.assertEqual(out[r0 + 5, c0 + 5], FREE)          # blob gone


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
        from slam.explorer import Explorer
        out = tempfile.mkdtemp()
        gt = load_ground_truth(GT_PATH)
        p = base_params(calibration="off", max_iterations=3)
        io = SimRobotIO(gt, start=(p["gt_start_x"], p["gt_start_y"], 0.0), time_scale=60)
        ex = Explorer(io, p, out, gt, log_print=False)
        try:
            ex.command("start")
            end = time.time() + 180
            while ex.state != "DONE" and time.time() < end:
                time.sleep(0.2)
            self.assertEqual(ex.state, "DONE")
            for name in ("map.png", "trajectory.csv", "report.json", "report.md", "comparison.png",
                         "log_events.csv", "log_scans_raw.csv", "map_cells.csv"):
                self.assertTrue(os.path.exists(os.path.join(ex.run_dir, name)), name)
            with open(os.path.join(ex.run_dir, "report.json")) as f:
                rep = json.load(f)
            self.assertEqual(rep["start"]["x"], 0.0)
            self.assertIn("gt", rep["end"])
            self.assertGreater(rep["metrics"]["coverage_pct"], 5)
            self.assertLess(rep["end_error_m"], 0.15)
            self.assertEqual(io.collisions, 0, "the robot body touched a wall")
            ex.set_params({"scan_speed_dps": 30})
            self.assertEqual(ex.p["scan_speed_dps"], 30.0)
        finally:
            ex.shutdown()
            io.close()
            shutil.rmtree(out, ignore_errors=True)


class PygameConsoleTest(unittest.TestCase):
    def test_every_tab_renders(self):
        try:
            import pygame
        except ImportError:
            self.skipTest("pygame not installed")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from slam.explorer import Explorer
        from slam.ui_pygame import TABS, Console
        out = tempfile.mkdtemp()
        gt = load_ground_truth(GT_PATH)
        p = base_params(calibration="off")
        io = SimRobotIO(gt, start=(p["gt_start_x"], p["gt_start_y"], 0.0), time_scale=60)
        ex = Explorer(io, p, out, gt, log_print=False)
        try:
            ex.command("scan")
            end = time.time() + 30
            while ex.stats["scans"] == 0 and time.time() < end:
                time.sleep(0.1)
            con = Console(ex, out)
            for theme in ("dark", "light"):
                con.ui.set_theme(theme)
                for tab in TABS:
                    con.tab = tab
                    con.ui.begin(con.screen, [])
                    con.poll()
                    con.draw([])
            self.assertIsNotNone(con.map)
            self.assertGreater(len(con.events), 0)
        finally:
            pygame.quit()
            ex.shutdown()
            io.close()
            shutil.rmtree(out, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
