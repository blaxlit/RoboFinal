"""The exploration mission: scan -> localise -> map -> pick a frontier -> drive.

One worker thread runs every robot action (so the robot only ever does one
thing at a time) and a telemetry thread records the pose at 10 Hz. The web
console talks to this class through ``command()``, ``set_params()`` and
``snapshot()``.
"""

import base64
import collections
import csv
import json
import math
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

from . import gridmodel as gm
from . import params as params_mod
from .calibration import Calibrator
from .driver import Abort, Driver, wrap
from .evaluation import comparison_image, evaluate, gt_to_world, gt_walls_world, world_to_gt
from .grid import OccupancyGrid, border_mask
from .planner import Costmap, choose_frontier_goal, plan_to, simplify_path
from .results import render_map, write_grid_csv, write_report
from .scan import ScanMatcher, beams_to_world, bin_samples, correct_range, sensor_origin


def compose(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-c * a[0] - s * a[1], s * a[0] - c * a[1], -a[2])


def pose_dict(p):
    return {"x": round(p[0], 4), "y": round(p[1], 4), "deg": round(math.degrees(p[2]), 2)}


class Finish(Exception):
    """Operator pressed Finish during a mission."""


class Explorer:
    def __init__(self, io, params, out_root, ground_truth=None, log_print=True):
        self.io = io
        self.p = params
        self.out_root = out_root
        self.gt = ground_truth
        self.gt_version = 1
        self.log_print = log_print
        self.lock = threading.RLock()
        self.state = "IDLE"
        self.detail = ""
        self._stop = False
        self._pause = False
        self._finish = False
        self._running = True
        self.driver = Driver(io, params, self._checkpoint)
        self.calibrator = Calibrator(self.driver, params, self.log)
        self.matcher = ScanMatcher()
        self.cmd_q = queue.Queue()
        self.events = collections.deque(maxlen=400)
        self._event_id = 0
        self.series = {k: collections.deque(maxlen=4000) for k in ("tof", "speed", "coverage", "accuracy", "match",
                                                                   "loc_error")}
        self.manual_cmd = (0.0, 0.0, 0.0, -1.0)
        self._manual_active = False
        self.param_version = 1
        self.mission = None
        self.last_report = None
        self.calibration = {}
        self._new_session()
        self.update_metrics()
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.telemetry = threading.Thread(target=self._telemetry, daemon=True)
        self.worker.start()
        self.telemetry.start()

    # ---- session / files --------------------------------------------------------------------
    def _new_session(self):
        with self.lock:
            self.t0 = self.io.now()
            self.corr = inverse(self.driver.odom_pose())
            self.grid = self._make_grid()
            self.blacklist = np.zeros((self.grid.rows, self.grid.cols), bool)
            self.traj = []
            self.stats = {"scans": 0, "distance_m": 0.0, "blocked_moves": 0, "iterations": 0}
            self.metrics = {}
            self.plan = None
            self.last_scan = None
            self.start_pose = (0.0, 0.0, 0.0)
            self._reset_grid_model()
            self._scan_id = 0
            self._last_beam_t = 0.0
            for s in self.series.values():
                s.clear()
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = os.path.join(self.out_root, f"run_{stamp}")
            os.makedirs(self.run_dir, exist_ok=True)
            self._files = {}
            for name, header in (
                    ("log_events.csv", ["t_s", "level", "message"]),
                    ("log_pose.csv", ["t_s", "state", "x_m", "y_m", "heading_deg", "odom_x_m", "odom_y_m",
                                      "odom_heading_deg", "tof_m", "gimbal_deg", "true_x_m", "true_y_m",
                                      "true_heading_deg"]),
                    ("log_scans_raw.csv", ["scan_id", "robot_angle_deg", "raw_range_m"]),
                    ("log_scans_filtered.csv", ["scan_id", "pose_x_m", "pose_y_m", "pose_deg", "beam_deg",
                                                "range_m", "hit", "world_x_m", "world_y_m"])):
                f = open(os.path.join(self.run_dir, name), "w", newline="", encoding="utf-8")
                w = csv.writer(f)
                w.writerow(header)
                self._files[name] = (f, w)
        self.log(f"New session - logs in {self.run_dir}")

    def _reset_grid_model(self):
        self.grid_model = None
        self.grid_edges = None
        self.snapped = None
        self.grid_version = 0
        self.wall_pts = []            # (points Nx2, wall angles N) per scan
        self.grid_history = []        # earlier detections, to confirm the cell size
        self.scanned_cells = set()
        self.cell_tries = collections.Counter()
        self.blocked_edges = set()
        self._view_aligned = False

    def _make_grid(self):
        p = self.p
        return OccupancyGrid(p["resolution_m"], p["width_m"], p["height_m"], p["origin_x_m"], p["origin_y_m"])

    def _write(self, name, row):
        f, w = self._files[name]
        w.writerow(row)

    def elapsed(self):
        return self.io.now() - self.t0

    def log(self, msg, level="info"):
        t = self.elapsed() if hasattr(self, "t0") else 0.0
        with self.lock:
            self._event_id += 1
            self.events.append((self._event_id, round(t, 2), level, msg))
            if hasattr(self, "_files"):
                self._write("log_events.csv", [round(t, 3), level, msg])
                self._files["log_events.csv"][0].flush()
        if self.log_print:
            print(f"[{t:7.1f}s] {level.upper():5s} {msg}", flush=True)

    # ---- poses ---------------------------------------------------------------------------------
    def pose(self):
        return compose(self.corr, self.driver.odom_pose())

    def set_pose(self, x, y, deg):
        with self.lock:
            self.corr = compose((x, y, math.radians(deg)), inverse(self.driver.odom_pose()))
        self.log(f"Pose set to ({x:.2f}, {y:.2f}, {deg:.0f} deg)")

    def true_pose_world(self):
        if not hasattr(self.io, "true_pose") or self.gt is None:
            return None
        x, y, th = self.io.true_pose()
        wx, wy = gt_to_world(self.p)(x, y)
        return float(wx), float(wy), wrap(th - math.radians(self.p["gt_start_deg"]))

    # ---- control from the console ---------------------------------------------------------------
    BLOCKING = {"start", "scan", "calibrate", "calibrate_wall", "turn", "move", "goto", "home",
                "gimbal_center", "gimbal_to", "save"}

    def command(self, name, args=None):
        args = args or {}
        if name == "stop":
            self._stop = True
            self._pause = False
            with self.cmd_q.mutex:
                self.cmd_q.queue.clear()
            self.driver.stop()
            self.log("STOP pressed")
        elif name == "pause":
            self._pause = True
            self.log("Paused")
        elif name == "resume":
            self._pause = False
            self.log("Resumed")
        elif name == "finish":
            if self.state in ("IDLE", "DONE"):
                self.cmd_q.put(("finish_now", args))
            else:
                self._finish = True
                self._pause = False
                self.log("Finish requested - wrapping up")
        elif name == "manual":
            self.manual_cmd = (float(args.get("vx", 0)), float(args.get("vy", 0)), float(args.get("wz", 0)),
                               time.monotonic())
        elif name == "reset_map":
            with self.lock:
                self.grid.clear()
                self.blacklist[:] = False
                self.plan = None
                self.last_scan = None
                self._reset_grid_model()
            self.log("Map cleared (pose kept)")
        elif name == "reset_all":
            if self.state not in ("IDLE", "DONE"):
                raise ValueError("stop the robot before starting a new session")
            self._close_files()
            self._new_session()
            self.mission = None
            self.last_report = None
            self.update_metrics()
        elif name == "redetect_grid":
            with self.lock:
                self.grid_model = None
            self.update_grid_model(log=True)
        elif name == "clear_blacklist":
            with self.lock:
                self.blacklist[:] = False
            self.log("Frontier blacklist cleared")
        elif name == "set_pose":
            self.set_pose(float(args["x"]), float(args["y"]), float(args["deg"]))
        elif name == "rotate_pose":
            x, y, th = self.pose()
            self.set_pose(x, y, math.degrees(th) + float(args["deg"]))
        elif name in self.BLOCKING:
            if name != "save" and self.state not in ("IDLE", "DONE"):
                raise ValueError(f"robot is busy ({self.state}) - press Stop first")
            self.cmd_q.put((name, args))
        else:
            raise ValueError(f"unknown command '{name}'")

    def set_params(self, changes):
        reset = False
        applied = {}
        for key, value in changes.items():
            value = params_mod.coerce(key, value)
            if self.p.get(key) != value:
                self.p[key] = value
                applied[key] = value
                reset |= params_mod.PARAMS[key][7]
        if applied:
            self.param_version += 1
            self.log("Settings: " + ", ".join(f"{k}={v}" for k, v in applied.items()))
        if reset:
            with self.lock:
                self.grid = self._make_grid()
                self.blacklist = np.zeros((self.grid.rows, self.grid.cols), bool)
                self.plan = None
                self._reset_grid_model()
            self.log("Map size/resolution changed - map cleared", "warn")
        if applied:
            if any(k.startswith("grid_") for k in applied) and self.wall_pts:
                self.update_grid_model(log=True)
            self.update_metrics()
        return applied

    def set_ground_truth(self, gt):
        with self.lock:
            self.gt = gt
            self.gt_version += 1
        self.log(f"Ground truth loaded: {gt['name']} ({len(gt['walls'])} walls)" if gt else "Ground truth removed")
        self.update_metrics()

    # ---- threads --------------------------------------------------------------------------------------
    def _checkpoint(self):
        if self._stop:
            raise Abort()
        if self._pause:
            self.driver.stop()
            before = self.state
            self.state = "PAUSED"
            while self._pause and not self._stop and self._running:
                self._manual_tick()
                time.sleep(0.05)
            self.state = before
            if self._stop:
                raise Abort()

    def _manual_tick(self):
        vx, vy, wz, t = self.manual_cmd
        if time.monotonic() - t < 0.4:  # dead-man: stops when the console stops sending
            self.driver.manual(vx, vy, wz)
            self._manual_active = True
        elif self._manual_active:
            self.driver.manual(0, 0, 0)
            self._manual_active = False

    def _worker(self):
        while self._running:
            try:
                name, args = self.cmd_q.get(timeout=0.05)
            except queue.Empty:
                if self.state in ("IDLE", "DONE"):
                    self._manual_tick()
                continue
            self._stop = False
            self._finish = False
            prev = self.state
            try:
                self._run_command(name, args)
                if name not in ("start", "finish_now"):
                    self.state = "IDLE"
            except Abort:
                self.driver.stop()
                self.log("Stopped by operator", "warn")
                self.state = "IDLE"
            except Exception as exc:  # keep the console alive whatever happens
                self.driver.stop()
                self.log(f"{name} failed: {exc!r}", "error")
                self.state = "IDLE" if prev != "DONE" else "DONE"
            finally:
                self.detail = ""
                self._stop = False

    def _telemetry(self):
        last = None
        while self._running:
            try:
                odom = self.driver.odom_pose()
                pose = compose(self.corr, odom)
                t = self.elapsed()
                tof = self.driver.tof_m()
                gim = self.driver.gimbal_deg()
                true = self.true_pose_world()
                with self.lock:
                    if last is None or math.hypot(pose[0] - last[0], pose[1] - last[1]) > 0.01 or \
                            abs(wrap(pose[2] - last[2])) > math.radians(2):
                        if last is not None:
                            step = math.hypot(pose[0] - last[0], pose[1] - last[1])
                            self.stats["distance_m"] += step
                        self.traj.append((round(t, 2), pose[0], pose[1], pose[2]))
                        speed = 0.0 if last is None else math.hypot(pose[0] - last[0], pose[1] - last[1]) / max(
                            t - last[3], 1e-3)
                        self.series["speed"].append((round(t, 2), round(speed, 3)))
                        last = (pose[0], pose[1], pose[2], t)
                    self.series["tof"].append((round(t, 2), round(tof, 3) if math.isfinite(tof) and tof < 9.9 else None))
                    self._write("log_pose.csv", [
                        round(t, 3), self.state, round(pose[0], 4), round(pose[1], 4), round(math.degrees(pose[2]), 2),
                        round(odom[0], 4), round(odom[1], 4), round(math.degrees(odom[2]), 2),
                        round(tof, 4) if math.isfinite(tof) else "", round(gim, 2),
                        *(("", "", "") if true is None else
                          (round(true[0], 4), round(true[1], 4), round(math.degrees(true[2]), 2)))])
            except Exception as exc:
                if self.log_print:
                    print(f"telemetry error: {exc!r}")
            time.sleep(0.1 / getattr(self.io, "time_scale", 1.0))

    # ---- commands -------------------------------------------------------------------------------------------
    def _run_command(self, name, args):
        if name == "start":
            self.run_mission()
        elif name == "scan":
            self.state = "SCANNING"
            self.do_scan()
        elif name == "calibrate":
            self.state = "CALIBRATING"
            self.calibrate()
        elif name == "calibrate_wall":
            self.state = "CALIBRATING"
            self.calibration.update(self.calibrator.wall(float(args["distance"])))
            self.param_version += 1
        elif name == "turn":
            self.state = "MOVING"
            self.driver.turn_by(math.radians(float(args["deg"])))
        elif name == "move":
            self.state = "MOVING"
            moved, blocked = self.driver.forward(float(args["m"]), on_tof=self._moving_tof)
            if blocked:
                self.log(f"Emergency stop after {moved:.2f} m (wall at {self.driver.last_blocked_range:.2f} m)", "warn")
        elif name == "goto":
            self.state = "MOVING"
            self.go_to(float(args["x"]), float(args["y"]))
        elif name == "home":
            self.state = "MOVING"
            self.go_home()
        elif name == "gimbal_center":
            self.driver.gimbal_goto(0.0, 0.0)
        elif name == "gimbal_to":
            self.driver.gimbal_goto(float(args["deg"]))
        elif name == "save":
            self.save_outputs()
        elif name == "finish_now":
            self.finish("operator finished")

    def calibrate(self):
        self.log("Auto calibration started")
        before = self.pose()
        res = self.calibrator.run(with_motion=self.p["calibrate_with_motion"])
        self.calibration.update(res)
        self.param_version += 1
        with self.lock:
            # the robot ends where it started, but the signs and drift may have
            # changed what odometry means, so re-anchor the pose
            self.corr = compose(before, inverse(self.driver.odom_pose()))
        self.log("Auto calibration done")

    def run_mission(self):
        p = self.p
        self.state = "RUNNING"
        started = self.io.now()
        if self.stats["scans"] == 0:
            # nothing mapped yet: the start pose defines the map frame
            with self.lock:
                self.corr = inverse(self.driver.odom_pose())
                self.traj.clear()
                self.stats["distance_m"] = 0.0
        self.start_pose = self.pose()
        self.mission = {"started": started, "started_at": datetime.now().isoformat(timespec="seconds"),
                        "start": self.start_pose}
        self.log("Mission started at map (%.2f, %.2f, %.0f deg)" % (self.start_pose[0], self.start_pose[1],
                                                                  math.degrees(self.start_pose[2])))
        reason = "stopped"
        try:
            if p["auto_calibrate"]:
                self.state = "CALIBRATING"
                self.calibrate()
            attempts = collections.Counter()
            skip_scan = False
            while True:
                self._checkpoint()
                if self._finish:
                    raise Finish()
                if self.stats["iterations"] >= p["max_iterations"]:
                    reason = f"reached max_iterations ({p['max_iterations']})"
                    break
                if self.io.now() - started > p["max_time_s"]:
                    reason = f"reached max_time_s ({p['max_time_s']:.0f} s)"
                    break
                if not skip_scan:
                    self.state = "SCANNING"
                    self.do_scan()
                skip_scan = False
                cov = self.metrics.get("coverage_pct")
                if cov is not None and (p["border_enabled"] or self.gt) and cov >= p["coverage_goal_pct"]:
                    reason = f"coverage goal reached ({cov:.1f}%)"
                    break
                self.state = "PLANNING"
                if self.grid_ready():
                    step = self.grid_step()
                    self.stats["iterations"] += 1
                    if step == "done":
                        reason = "every reachable grid cell explored"
                        break
                    continue
                info = self.plan_frontier()
                if info is None:
                    reason = "no reachable frontier left - exploration complete"
                    break
                key = tuple(int(v) // max(1, int(0.3 / self.grid.res)) for v in info["goal"])
                attempts[key] += 1
                if len(info["path"]) <= 2 or attempts[key] > 3:
                    self._blacklist(info["frontier"])
                    self.log(f"Frontier at ({info['goal_xy'][0]:.2f}, {info['goal_xy'][1]:.2f}) cannot be cleared "
                             f"from here - ignoring it", "warn")
                    skip_scan = True
                    continue
                self.state = "MOVING"
                moved, blocked = self.follow(info["path"], p["scan_every_m"])
                self.stats["iterations"] += 1
                if blocked:
                    attempts[key] += 1
                if moved < 0.02 and not blocked:
                    attempts[key] += 1
            if p["return_home"]:
                self.state = "MOVING"
                self.log("Returning to the start")
                self.go_home()
        except Finish:
            reason = "operator finished"
        except Abort:
            reason = "operator stopped"
            self.driver.stop()
            self._stop = False
        self.finish(reason)

    def finish(self, reason):
        self.driver.stop()
        end = self.pose()
        with self.lock:
            self.plan = None
        self.state = "SAVING"
        self.update_metrics()
        report = self.build_report(reason, end)
        s, e = report["start"], report["end"]
        self.log(f"Mission finished ({reason}). Started at ({s['x']:.2f}, {s['y']:.2f}, {s['deg']:.0f} deg), "
                 f"ended at ({e['x']:.2f}, {e['y']:.2f}, {e['deg']:.0f} deg)")
        if "gt" in s:
            self.log(f"In the arena frame: start ({s['gt']['x']:.2f}, {s['gt']['y']:.2f}), "
                     f"end ({e['gt']['x']:.2f}, {e['gt']['y']:.2f})")
        m = self.metrics
        if "accuracy_pct" in m:
            self.log(f"Map Accuracy {m['accuracy_pct']:.2f}%  Coverage {m['coverage_pct']:.2f}%")
        elif "coverage_pct" in m:
            self.log(f"Coverage {m['coverage_pct']:.2f}%")
        try:
            self.save_outputs(report)
        finally:
            self.state = "DONE"

    # ---- scanning ----------------------------------------------------------------------------------------------
    def do_scan(self):
        p = self.p
        self.detail = "scanning"
        samples = self.driver.scan()
        odom = self.driver.odom_pose()
        angles, ranges, hits = bin_samples(samples, p)
        guess = compose(self.corr, odom)
        pose = guess
        self._scan_id += 1
        sid = self._scan_id
        if p["scan_matching"] and self.stats["scans"] > 0:
            with self.lock:
                pose, _ = self.matcher.match(self.grid, guess, angles, ranges, hits, p)
            dx, dy = pose[0] - guess[0], pose[1] - guess[1]
            dth = math.degrees(wrap(pose[2] - guess[2]))
            self.series["match"].append((round(self.elapsed(), 2), round(math.hypot(dx, dy) * 100, 2)))
            if math.hypot(dx, dy) > 0.005 or abs(dth) > 0.3:
                self.log(f"Scan {sid}: localisation corrected by ({dx * 100:+.1f}, {dy * 100:+.1f}) cm, {dth:+.1f} deg")
        pose = self._snap_to_grid(pose, angles, ranges, hits, sid)
        true = self.true_pose_world()
        if true is not None:
            e_before = math.hypot(true[0] - guess[0], true[1] - guess[1])
            e_after = math.hypot(true[0] - pose[0], true[1] - pose[1])
            self.series["loc_error"].append((round(self.elapsed(), 2), round(e_after * 100, 2)))
            self.log(f"Scan {sid}: simulator truth error {e_before * 100:.1f} cm -> {e_after * 100:.1f} cm, "
                     f"{abs(math.degrees(wrap(true[2] - guess[2]))):.1f} -> "
                     f"{abs(math.degrees(wrap(true[2] - pose[2]))):.1f} deg", "debug")
        with self.lock:
            self.corr = compose(pose, inverse(odom))
            origin = sensor_origin(pose, p)
            self.grid.integrate(origin, angles + pose[2], ranges, hits, p["l_occ"], p["l_free"], p["l_clamp"],
                                max_gap=math.radians(p["scan_step_deg"] * 1.6))
            wx, wy, _ = beams_to_world(pose, angles, ranges, p)
            self.last_scan = {"pose": pose, "pts": [[round(float(a), 3), round(float(b), 3), int(h)]
                                                    for a, b, h in zip(wx, wy, hits)]}
            self.stats["scans"] += 1
            for a, r in samples:
                self._write("log_scans_raw.csv", [sid, round(math.degrees(a), 2),
                                                  round(r, 4) if math.isfinite(r) else "inf"])
            for a, r, h, x, y in zip(angles, ranges, hits, wx, wy):
                self._write("log_scans_filtered.csv", [sid, round(pose[0], 4), round(pose[1], 4),
                                                       round(math.degrees(pose[2]), 2), round(math.degrees(a), 2),
                                                       round(float(r), 4), int(h), round(float(x), 4),
                                                       round(float(y), 4)])
        n_hit = int(hits.sum())
        self.log(f"Scan {sid}: {len(samples)} readings -> {len(angles)} beams ({n_hit} walls) "
                 f"at ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f} deg)")
        if p["grid_mode"] != "off":
            wx, wy, _ = beams_to_world(pose, angles[hits], ranges[hits], p)
            pts, ang = gm.oriented_points(wx, wy)
            if len(pts):
                self.wall_pts.append((pts, ang))
            self.update_grid_model()
            if self.grid_model is not None and self.grid_model.cell_trusted:
                self.scanned_cells.add(gm.cell_of(self.grid_model, pose[0], pose[1]))
        self.update_metrics()

    # ---- auto grid -------------------------------------------------------------------------------------------------
    def _snap_to_grid(self, pose, angles, ranges, hits, sid):
        """Nudge the scan pose so its walls sit on the grid (heading, then position)."""
        p, model = self.p, self.grid_model
        if p["grid_mode"] == "off" or not p["grid_snap_pose"] or model is None or not model.theta_locked \
                or not hits.any():
            return pose
        wx, wy, _ = beams_to_world(pose, angles[hits], ranges[hits], p)
        pts, ang = gm.oriented_points(wx, wy)
        dx, dy, dth = gm.snap_pose(model, pose, pts, ang)
        if abs(dx) + abs(dy) < 1e-4 and abs(dth) < 1e-5:
            return pose
        self.log(f"Scan {sid}: grid snap ({dx * 100:+.1f}, {dy * 100:+.1f}) cm, {math.degrees(dth):+.1f} deg")
        return pose[0] + dx, pose[1] + dy, wrap(pose[2] + dth)

    def update_grid_model(self, log=False):
        p = self.p
        if p["grid_mode"] == "off" or not self.wall_pts:
            with self.lock:
                self.grid_model = self.grid_edges = self.snapped = None
            return
        prev = self.grid_model
        if prev is not None and prev.cell_trusted and prev.theta_locked and not log:
            model = prev  # locked: re-fitting to snapped points would let the grid drift with them
        else:
            pts = np.concatenate([a for a, _ in self.wall_pts[-80:]])
            ang = np.concatenate([b for _, b in self.wall_pts[-80:]])
            model = gm.detect(pts, ang, p, prev, self.grid_history)
            if model is None:
                return
            self.grid_history = (self.grid_history + [model])[-5:]
        if model is not prev and (prev is None or model.cell_trusted != prev.cell_trusted
                                  or abs(model.cell - prev.cell) > 0.01 or log):
            self.log(f"Grid: {'found' if model.cell_trusted else 'angle found'} - cell {model.cell:.3f} m, "
                     f"angle {math.degrees(model.theta):+.1f} deg (score {model.score:.2f}"
                     f"{'' if model.cell_trusted else ', cell size not trusted yet'})")
        with self.lock:
            classes = self.classes()
            edges = gm.classify_edges(model, self.grid, classes, p) if model.cell_trusted else None
            snapped = gm.render(model, edges, self.grid, p) if edges is not None else None
            self.grid_model, self.grid_edges, self.snapped = model, edges, snapped
            self.grid_version += 1
        if model.cell_trusted and p["grid_align_view"] and not self._view_aligned:
            self._view_aligned = True
            self.set_params({"map_rotation_deg": round(-math.degrees(model.theta), 1)})

    def grid_ready(self):
        return (self.p["grid_mode"] != "off" and self.p["grid_drive"] and self.grid_model is not None
                and self.grid_model.cell_trusted and self.grid_edges is not None)

    def grid_plan(self):
        """Nearest cell (through open edges) that still needs looking at."""
        model, edges, p = self.grid_model, self.grid_edges, self.p
        x, y, _ = self.pose()
        start = gm.cell_of(model, x, y)
        prev = {start: None}
        queue_ = collections.deque([start])
        inside = border_mask(self.grid, p) if p["border_enabled"] else None
        while queue_:
            c = queue_.popleft()
            a, b = c[0] - edges.i0, c[1] - edges.j0
            if not (0 <= a < edges.cell_free.shape[0] and 0 <= b < edges.cell_free.shape[1]):
                continue
            unseen = edges.cell_free[a, b] < p["grid_cell_seen_frac"]
            needs = unseen or gm.unknown_count(edges, c) > 0
            if c != start and needs and c not in self.scanned_cells and self.cell_tries[c] < 3:
                if inside is not None:
                    r, cc = self.grid.world_to_cell(*gm.cell_center(model, c))
                    if not (self.grid.inside(r, cc) and inside[r, cc]):
                        needs = False
                if needs:
                    path = [c]
                    while prev[path[-1]] is not None:
                        path.append(prev[path[-1]])
                    return path[::-1]
            for n in gm.neighbours(edges, c, self.blocked_edges):
                if n not in prev:
                    prev[n] = c
                    queue_.append(n)
        return None

    def grid_step(self):
        path = self.grid_plan()
        if path is None:
            return "done"
        model = self.grid_model
        target = path[-1]
        # one cell at a time (grid_cells_per_step), then scan again and re-plan
        path = path[:1 + max(1, int(self.p["grid_cells_per_step"]))]
        if path[-1] == target:
            self.cell_tries[target] += 1
        centers = [gm.cell_center(model, c) for c in path]
        with self.lock:
            self.plan = {"frontiers": [], "goal": [round(v, 3) for v in centers[-1]],
                         "path": [[round(x, 3), round(y, 3)] for x, y in [self.pose()[:2]] + centers[1:]]}
        self.log(f"Grid: cell {path[0]} -> {path[-1]}" + ("" if path[-1] == target else f" (heading for {target})"))
        self.state = "MOVING"
        self.grid_follow(path)
        return "moved"

    def grid_follow(self, path):
        """Drive cell centre to cell centre along grid axes, centring on the way."""
        p, model = self.p, self.grid_model
        runs = []
        for a, b in zip(path, path[1:]):
            d = (b[0] - a[0], b[1] - a[1])
            if runs and runs[-1][0] == d and not p["grid_stop_each_cell"]:
                runs[-1][1].append(b)
            else:
                runs.append((d, [b]))
        cur = path[0]
        for d, cells in runs:
            self._checkpoint()
            if self._finish:
                raise Finish()
            heading = model.theta + math.atan2(d[1], d[0])
            x, y, th = self.pose()
            err = wrap(heading - th)
            if abs(err) > math.radians(p["turn_tolerance_deg"]):
                self.detail = f"turning {math.degrees(err):+.0f} deg"
                self.driver.turn_by(err)
            # centre on the line through the cell centres
            x, y, th = self.pose()
            cx, cy = gm.cell_center(model, cur)
            lx, ly = -math.sin(heading), math.cos(heading)
            off = (cx - x) * lx + (cy - y) * ly
            if abs(off) > p["grid_center_tol_m"]:
                self.detail = f"centring {off * 100:+.0f} cm"
                self.driver.strafe(off)
            # look before driving: the ToF must see past the edge into the next cell
            front = self.driver.front_range()
            need = model.cell / 2 - p["tof_offset_m"] - p["gimbal_offset_x_m"] + 0.12
            if front is not None and front < need:
                self.blocked_edges.add(frozenset((cur, cells[0])))
                self.log(f"Wall ahead at {front:.2f} m between cells {cur} and {cells[0]} - not moving, "
                         f"marking that edge as a wall", "warn")
                return False
            ex, ey = gm.cell_center(model, cells[-1])
            x, y, _ = self.pose()
            dist = (ex - x) * math.cos(heading) + (ey - y) * math.sin(heading)
            self.detail = f"to cell {cells[-1]} ({dist:.2f} m)"
            moved, blocked = self.driver.forward(dist, on_tof=self._moving_tof) if dist > 0.02 else (0.0, False)
            if blocked:
                # something is in the way: treat the next edge as closed
                self.stats["blocked_moves"] += 1
                nxt = cells[min(len(cells) - 1, int(moved / model.cell + 0.5))]
                prv = cur if nxt == cells[0] else cells[cells.index(nxt) - 1]
                self.blocked_edges.add(frozenset((prv, nxt)))
                self.log(f"Emergency stop between cells {prv} and {nxt} - treating that edge as a wall", "warn")
                return False
            cur = cells[-1]
            if p["grid_scan_each_cell"] and cur != path[-1]:
                self.state_scan_between_legs()
        return True

    def _edge_near_seen(self, sg):
        """Only show unknown edges next to explored space (not the empty ring)."""
        x, y = (sg[0] + sg[2]) / 2, (sg[1] + sg[3]) / 2
        r, c = self.grid.world_to_cell(x, y)
        r0, r1 = max(0, r - 3), min(self.grid.rows, r + 4)
        c0, c1 = max(0, c - 3), min(self.grid.cols, c + 4)
        return bool((self.grid.logodds[r0:r1, c0:c1] < 0).any())

    def output_classes(self):
        """The map that is saved and scored: the clean grid map when there is one."""
        if self.p["grid_mode"] != "off" and self.p["grid_snap_output"] and self.snapped is not None:
            return self.snapped
        return self.classes()

    def _moving_tof(self, t, raw_m):
        p = self.p
        if not p["map_while_moving"] or t - self._last_beam_t < 0.1:
            return
        self._last_beam_t = t
        if raw_m <= 0 or raw_m > 9.9:
            rng, hit = p["no_hit_free_m"], False
        else:
            rng = correct_range(raw_m, p)
            hit = p["tof_min_m"] < raw_m < p["tof_max_m"]
            if not hit:
                rng = min(rng, p["no_hit_free_m"])
        if rng <= 0:
            return
        pose = self.pose()
        with self.lock:
            self.grid.integrate(sensor_origin(pose, p), [pose[2]], [rng], [hit],
                                p["l_occ"], p["l_free"], p["l_clamp"], weight=0.5)

    # ---- planning / driving -------------------------------------------------------------------------------------
    def classes(self):
        return self.grid.classify(self.p["occ_prob"], self.p["free_prob"], self.p["clean_isolated"],
                                  self.p["min_wall_blob_cells"])

    def plan_frontier(self):
        pose = self.pose()
        with self.lock:
            classes = self.classes()
            blacklist = self.blacklist.copy()
            grid = self.grid
        info = choose_frontier_goal(grid, classes, pose, self.p, blacklist)
        fr = [[round(float(v), 3) for v in grid.cell_to_world(c[0], c[1])] + [len(cells)]
              for c, cells in info["frontiers"]]
        if info["goal"] is None:
            with self.lock:
                self.plan = {"frontiers": fr, "path": [], "goal": None}
            return None
        path_w = [grid.cell_to_world(r, c) for r, c in info["path"]]
        gx, gy = grid.cell_to_world(*info["goal"])
        info["goal_xy"] = (float(gx), float(gy))
        with self.lock:
            self.plan = {"frontiers": fr, "path": [[round(float(x), 3), round(float(y), 3)] for x, y in path_w],
                         "goal": [round(float(gx), 3), round(float(gy), 3)]}
        self.log(f"Target frontier ({gx:.2f}, {gy:.2f}), path {len(info['path']) * grid.res:.2f} m, "
                 f"{len(info['frontiers'])} frontier(s) left")
        return info

    def _blacklist(self, cells):
        mask = np.zeros(self.blacklist.shape, np.uint8)
        mask[cells[:, 0], cells[:, 1]] = 1
        k = max(1, int(round(0.15 / self.grid.res)))
        mask = cv2.dilate(mask, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
        with self.lock:
            self.blacklist |= mask

    def follow(self, path_cells, budget):
        """Drive along a planned path for at most ``budget`` metres."""
        with self.lock:
            cm = Costmap(self.grid, self.classes(), self.p)
            grid = self.grid
        wps = simplify_path(path_cells, cm.lethal | cm.body)
        travelled = 0.0
        for r, c in wps[1:]:
            self._checkpoint()
            if self._finish:
                raise Finish()
            wx, wy = grid.cell_to_world(r, c)
            x, y, th = self.pose()
            dist = math.hypot(wx - x, wy - y)
            if dist < 0.03:
                continue
            err = wrap(math.atan2(wy - y, wx - x) - th)
            if abs(err) > math.radians(self.p["turn_tolerance_deg"]):
                self.detail = f"turning {math.degrees(err):+.0f} deg"
                self.driver.turn_by(err)
            seg = min(dist, budget - travelled)
            if seg < 0.03:
                break
            self.detail = f"driving {seg:.2f} m"
            moved, blocked = self.driver.forward(seg, on_tof=self._moving_tof)
            travelled += moved
            if blocked:
                self.stats["blocked_moves"] += 1
                self.log(f"Emergency stop: wall {self.driver.last_blocked_range:.2f} m ahead", "warn")
                return travelled, True
            if travelled >= budget - 0.02:
                break
        return travelled, False

    def go_to(self, x, y, max_legs=30):
        self.log(f"Going to ({x:.2f}, {y:.2f})")
        for _ in range(max_legs):
            pose = self.pose()
            if math.hypot(x - pose[0], y - pose[1]) < 0.1:
                self.log("Arrived")
                return True
            with self.lock:
                classes = self.classes()
                grid = self.grid
            path = plan_to(grid, classes, pose, (x, y), self.p)
            if not path or len(path) < 2:
                self.log("No known path there - scan or explore more first", "warn")
                return False
            with self.lock:
                self.plan = {"frontiers": [], "goal": [x, y],
                             "path": [[round(float(a), 3), round(float(b), 3)]
                                      for a, b in (grid.cell_to_world(r, c) for r, c in path)]}
            self.follow(path, self.p["scan_every_m"])
            self.state_scan_between_legs()
        return False

    def state_scan_between_legs(self):
        prev = self.state
        self.state = "SCANNING"
        self.do_scan()
        self.state = prev

    def go_home(self):
        ok = self.go_to(self.start_pose[0], self.start_pose[1])
        if ok:
            err = wrap(self.start_pose[2] - self.pose()[2])
            self.driver.turn_by(err)
        return ok

    # ---- metrics / outputs ---------------------------------------------------------------------------------
    def update_metrics(self):
        with self.lock:
            classes = self.output_classes()
            raw = self.classes() if classes is self.snapped else None
            grid = self.grid
            border = border_mask(grid, self.p) if self.p["border_enabled"] else None
            gt = self.gt
        try:
            m = evaluate(classes, grid, self.p, gt, border)
            m["map"] = "grid" if raw is not None else "raw"
            if raw is not None:
                r = evaluate(raw, grid, self.p, gt, border)
                r.pop("_images", None)
                m["raw"] = {k: r[k] for k in ("accuracy_pct", "coverage_pct", "strict_accuracy_pct") if k in r}
            if self.grid_model is not None:
                m["grid"] = self.grid_model.as_dict()
                if self.grid_edges is not None:
                    e = self.grid_edges
                    m["grid"].update(
                        walls=int((e.vert == gm.EDGE_WALL).sum() + (e.horz == gm.EDGE_WALL).sum()),
                        open=int((e.vert == gm.EDGE_OPEN).sum() + (e.horz == gm.EDGE_OPEN).sum()),
                        cells_seen=int((e.cell_free >= self.p["grid_cell_seen_frac"]).sum()))
        except Exception as exc:
            self.log(f"Scoring failed: {exc!r}", "error")
            return
        images = m.pop("_images", None)
        t = round(self.elapsed(), 2)
        with self.lock:
            self.metrics = m
            self._eval_images = images
            if "coverage_pct" in m:
                self.series["coverage"].append((t, m["coverage_pct"]))
            if "accuracy_pct" in m:
                self.series["accuracy"].append((t, m["accuracy_pct"]))

    def build_report(self, reason, end_pose):
        start = self.start_pose
        started = self.mission["started"] if self.mission else self.t0
        rep = {
            "robot": self.io.name,
            "started_at": self.mission["started_at"] if self.mission else "",
            "duration_s": round(self.io.now() - started, 2),
            "finish_reason": reason,
            "scans": self.stats["scans"],
            "distance_m": round(self.stats["distance_m"], 3),
            "blocked_moves": self.stats["blocked_moves"],
            "start": pose_dict(start),
            "end": pose_dict(end_pose),
            "metrics": dict(self.metrics),
            "calibration": dict(self.calibration),
            "run_dir": self.run_dir,
        }
        if self.gt is not None:
            tf = world_to_gt(self.p)
            for key, pz in (("start", start), ("end", end_pose)):
                gx, gy = tf(pz[0], pz[1])
                rep[key]["gt"] = {"x": round(float(gx), 4), "y": round(float(gy), 4),
                                  "deg": round(math.degrees(wrap(pz[2] + math.radians(self.p["gt_start_deg"]))), 2)}
        true = self.true_pose_world()
        if true is not None:
            rep["true_end"] = pose_dict(true)
            rep["end_error_m"] = round(math.hypot(true[0] - end_pose[0], true[1] - end_pose[1]), 4)
            rep["end_error_deg"] = round(abs(math.degrees(wrap(true[2] - end_pose[2]))), 2)
        self.last_report = rep
        return rep

    def save_outputs(self, report=None):
        d = self.run_dir
        with self.lock:
            classes = self.classes()
            out_classes = self.output_classes()
            grid = self.grid
            traj = list(self.traj)
            prob = grid.probability()
            images = getattr(self, "_eval_images", None)
            model, edges = self.grid_model, self.grid_edges
        if self.p["grid_mode"] != "off" and self.wall_pts and (model is None or not model.cell_trusted):
            # not confirmed during the run: best fit over all the data, for the saved maps
            pts = np.concatenate([a for a, _ in self.wall_pts])
            ang = np.concatenate([b for _, b in self.wall_pts])
            final = gm.detect(pts, ang, self.p, model, final=True)
            if final is not None and final.cell_trusted:
                with self.lock:
                    model, edges = final, gm.classify_edges(final, grid, classes, self.p)
                    self.snapped = gm.render(model, edges, grid, self.p)
                    self.grid_model, self.grid_edges = model, edges
                    self.grid_version += 1
                    out_classes = self.output_classes()
                self.log(f"Grid (fit over the whole run): cell {model.cell:.3f} m, angle "
                         f"{math.degrees(model.theta):+.1f} deg")
        if report is None:
            report = self.build_report("saved by operator", self.pose())
        walls = gt_walls_world(self.gt, self.p) if self.gt else None
        end = (traj[-1][1], traj[-1][2], traj[-1][3]) if traj else self.pose()
        xy = [(x, y) for _, x, y, _ in traj]
        cv2.imwrite(os.path.join(d, "map.png"), render_map(grid, out_classes, self.p, xy, self.start_pose, end))
        if out_classes is not classes:
            cv2.imwrite(os.path.join(d, "map_raw.png"), render_map(grid, classes, self.p, xy, self.start_pose, end))
        if walls:
            cv2.imwrite(os.path.join(d, "map_with_ground_truth.png"),
                        render_map(grid, out_classes, self.p, xy, self.start_pose, end, walls))
        if model is not None and edges is not None:
            cv2.imwrite(os.path.join(d, "map_grid.png"),
                        gm.render_grid_image(model, edges, self.p, xy, self.start_pose, end))
            with open(os.path.join(d, "grid_model.json"), "w", encoding="utf-8") as f:
                json.dump({**model.as_dict(), "cells_i": [edges.i0, edges.i1], "cells_j": [edges.j0, edges.j1],
                           "edge_states": "0 unknown, 1 open, 2 wall",
                           "vertical_edges": edges.vert.tolist(), "horizontal_edges": edges.horz.tolist(),
                           "wall_segments_m": [list(map(lambda v: round(v, 4), sgm[:4]))
                                               for sgm in edges.segments(model, gm.EDGE_WALL)]}, f, indent=1)
        if images is not None:
            cv2.imwrite(os.path.join(d, "comparison.png"), comparison_image(images))
        write_grid_csv(os.path.join(d, "map_cells.csv"), out_classes, grid)
        np.save(os.path.join(d, "map_prob.npy"), prob)
        with open(os.path.join(d, "map_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"resolution_m": grid.res, "origin_x_m": grid.origin_x, "origin_y_m": grid.origin_y,
                       "rows": grid.rows, "cols": grid.cols,
                       "frame": "x forward at start, y left, heading counter-clockwise"}, f, indent=2)
        with open(os.path.join(d, "trajectory.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "x_m", "y_m", "heading_deg"])
            for t, x, y, th in traj:
                w.writerow([t, round(x, 4), round(y, 4), round(math.degrees(th), 2)])
        with open(os.path.join(d, "settings_used.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(self.p), f, sort_keys=True)
        if self.gt:
            with open(os.path.join(d, "ground_truth.json"), "w", encoding="utf-8") as f:
                json.dump(self.gt, f, indent=2)
        write_report(d, report)
        for f, _ in self._files.values():
            f.flush()
        self.log(f"Saved map, trajectory, logs and report to {d}")
        return d

    # ---- console snapshot -------------------------------------------------------------------------------------------
    def snapshot(self, map_version=-1, event_id=0, gt_version=0, param_version=0, traj_from=0):
        pose = self.pose()
        odom = self.driver.odom_pose()
        tof = self.driver.tof_m()
        out = {
            "state": self.state, "detail": self.detail, "robot": self.io.name, "paused": self._pause,
            "t": round(self.elapsed(), 2), "pose": pose_dict(pose), "odom": pose_dict(odom),
            "corr": pose_dict(self.corr), "start": pose_dict(self.start_pose),
            "tof_m": round(tof, 3) if math.isfinite(tof) and tof < 9.9 else None,
            "gimbal_deg": round(self.driver.gimbal_deg(), 1), "battery": self.io.battery(),
            "run_dir": self.run_dir, "param_version": self.param_version, "gt_version": self.gt_version,
        }
        true = self.true_pose_world()
        if true is not None:
            out["true_pose"] = pose_dict(true)
        with self.lock:
            out["stats"] = dict(self.stats, distance_m=round(self.stats["distance_m"], 3))
            out["metrics"] = dict(self.metrics)
            out["plan"] = self.plan
            out["scan"] = self.last_scan and {"pose": pose_dict(self.last_scan["pose"]), "pts": self.last_scan["pts"]}
            out["events"] = [e for e in self.events if e[0] > event_id]
            n = len(self.traj)
            start = traj_from if 0 <= traj_from <= n else 0
            out["traj"] = {"from": start, "len": n,
                           "pts": [[round(x, 3), round(y, 3)] for _, x, y, _ in self.traj[start:]]}
            out["series"] = {k: _downsample(list(v), 400) for k, v in self.series.items()}
            g = self.grid
            if g.version != map_version:
                cls = self.classes()
                prob = (g.probability() * 255).astype(np.uint8)
                out["map"] = {"version": g.version, "rows": g.rows, "cols": g.cols, "res": g.res,
                              "origin": [g.origin_x, g.origin_y],
                              "cls": base64.b64encode(cls.tobytes()).decode(),
                              "prob": base64.b64encode(prob.tobytes()).decode()}
            if gt_version != self.gt_version:
                out["gt"] = None if self.gt is None else {"name": self.gt["name"],
                                                          "walls": gt_walls_world(self.gt, self.p), "raw": self.gt}
        with self.lock:
            model, edges = self.grid_model, self.grid_edges
            out["grid_version"] = self.grid_version
            if model is not None:
                g = model.as_dict()
                if edges is not None:
                    g["walls"] = [[round(v, 3) for v in sg[:4]] for sg in edges.segments(model, gm.EDGE_WALL)]
                    g["unknown"] = [[round(v, 3) for v in sg[:4]] for sg in edges.segments(model, gm.EDGE_UNKNOWN)
                                    if self._edge_near_seen(sg)]
                out["grid"] = g
        if param_version != self.param_version:
            out["params"] = dict(self.p)
        if self.last_report:
            out["report"] = {k: self.last_report[k] for k in ("start", "end", "finish_reason", "duration_s")
                             if k in self.last_report}
        return out

    def _close_files(self):
        for f, _ in self._files.values():
            try:
                f.close()
            except Exception:
                pass

    def shutdown(self):
        self._stop = True
        self._running = False
        self.driver.stop()
        self._close_files()


def _downsample(points, n):
    if len(points) <= n:
        return points
    step = len(points) / n
    return [points[int(i * step)] for i in range(n)] + [points[-1]]
