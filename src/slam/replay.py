"""Rebuild a map from a recorded run with the current settings (no robot).

Uses the run's raw ToF readings (log_scans_raw.csv) and the pose each scan was
taken from (log_scans_filtered.csv), then runs the current filters, scan
matching, grid snapping and grid model again. Handy for tuning settings on
real data.
"""

import collections
import csv
import math
import os

import cv2
import numpy as np

from . import gridmodel as gm
from .driver import wrap
from .grid import OccupancyGrid
from .results import render_map
from .scan import ScanMatcher, beams_to_world, bin_samples, sensor_origin


def load_run(run_dir):
    raw = collections.defaultdict(list)
    with open(os.path.join(run_dir, "log_scans_raw.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rng = float("inf") if r["raw_range_m"] == "inf" else float(r["raw_range_m"])
            raw[int(r["scan_id"])].append((math.radians(float(r["robot_angle_deg"])), rng))
    poses = {}
    with open(os.path.join(run_dir, "log_scans_filtered.csv"), encoding="utf-8") as f:
        for r in csv.DictReader(f):
            poses.setdefault(int(r["scan_id"]), (float(r["pose_x_m"]), float(r["pose_y_m"]),
                                                  math.radians(float(r["pose_deg"]))))
    return [(sid, poses[sid], raw[sid]) for sid in sorted(raw) if sid in poses]


def replay(run_dir, params, log=print):
    """Returns (grid, raw classes, clean classes or None, model, edges, poses)."""
    scans = load_run(run_dir)
    p = params
    grid = OccupancyGrid(p["resolution_m"], p["width_m"], p["height_m"], p["origin_x_m"], p["origin_y_m"])
    matcher = ScanMatcher()
    model = None
    wall_pts = []
    history = []
    poses = []
    corr = (0.0, 0.0, 0.0)  # correction carried from scan to scan, on top of the logged poses
    for k, (sid, logged, samples) in enumerate(scans):
        angles, ranges, hits = bin_samples(samples, p)
        c, s = math.cos(corr[2]), math.sin(corr[2])
        guess = (corr[0] + c * logged[0] - s * logged[1], corr[1] + s * logged[0] + c * logged[1],
                 wrap(logged[2] + corr[2]))
        pose = guess
        if p["scan_matching"] and k > 0:
            pose, _ = matcher.match(grid, guess, angles, ranges, hits, p)
        if model is not None and p["grid_mode"] != "off" and p["grid_snap_pose"] and hits.any() and model.theta_locked:
            wx, wy, _ = beams_to_world(pose, angles[hits], ranges[hits], p)
            pts, ang = gm.oriented_points(wx, wy)
            dx, dy, dth = gm.snap_pose(model, pose, pts, ang)
            pose = (pose[0] + dx, pose[1] + dy, wrap(pose[2] + dth))
        # keep the extra correction for the next scans (like the live corr)
        dth = wrap(pose[2] - logged[2])
        c, s = math.cos(dth), math.sin(dth)
        corr = (pose[0] - (c * logged[0] - s * logged[1]), pose[1] - (s * logged[0] + c * logged[1]), dth)
        grid.integrate(sensor_origin(pose, p), angles + pose[2], ranges, hits, p["l_occ"], p["l_free"],
                       p["l_clamp"], max_gap=math.radians(p["scan_step_deg"] * 1.6))
        poses.append(pose)
        if p["grid_mode"] != "off" and hits.any():
            wx, wy, _ = beams_to_world(pose, angles[hits], ranges[hits], p)
            pts, ang = gm.oriented_points(wx, wy)
            if len(pts):
                wall_pts.append((pts, ang))
            if model is None or not model.cell_trusted:  # locked once trusted, like the live explorer
                new = gm.detect(np.concatenate([a for a, _ in wall_pts]), np.concatenate([b for _, b in wall_pts]),
                                p, model, history, anchor=poses[0][:2] if p["grid_start_centered"] else None)
                if new is not None:
                    history.append(new)
                    model = new
        log(f"scan {sid}: pose ({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.1f}) "
            f"moved {math.hypot(pose[0] - logged[0], pose[1] - logged[1]) * 100:.1f} cm vs the recorded pose")
    raw = grid.classify(p["occ_prob"], p["free_prob"], p["clean_isolated"], p["min_wall_blob_cells"])
    edges = clean = None
    if wall_pts and (model is None or not model.cell_trusted):
        model = gm.detect(np.concatenate([a for a, _ in wall_pts]), np.concatenate([b for _, b in wall_pts]),
                          p, model, final=True, anchor=poses[0][:2] if p["grid_start_centered"] and poses else None)
    if model is not None and model.cell_trusted:
        edges = gm.classify_edges(model, grid, raw, p)
        clean = gm.render(model, edges, grid, p)
        log(f"grid: {model.as_dict()}")
    return grid, raw, clean, model, edges, poses


def save_replay(run_dir, out_dir, params, log=print):
    os.makedirs(out_dir, exist_ok=True)
    grid, raw, clean, model, edges, poses = replay(run_dir, params, log)
    traj = [(x, y) for x, y, _ in poses]
    start, end = (poses[0] if poses else None), (poses[-1] if poses else None)
    cv2.imwrite(os.path.join(out_dir, "replay_map_raw.png"), render_map(grid, raw, params, traj, start, end))
    if clean is not None:
        cv2.imwrite(os.path.join(out_dir, "replay_map_clean.png"), render_map(grid, clean, params, traj, start, end))
        cv2.imwrite(os.path.join(out_dir, "replay_map_grid.png"),
                    gm.render_grid_image(model, edges, params, traj, start, end))
    return out_dir
