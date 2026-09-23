"""Every tunable SLAM value, with its type, range and help text.

The console builds its settings panel from PARAMS, so a value added here shows
up in the browser with no other change. Values load in this order (later wins):
the defaults below, the ``slam:`` section of config/settings.yaml, then
config/slam_settings.yaml (written by the console's "Save settings" button).
"""

import copy
import os

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SETTINGS_PATH = os.path.join(BASE_DIR, "config", "settings.yaml")
OVERRIDE_PATH = os.path.join(BASE_DIR, "config", "slam_settings.yaml")

# key: (default, type, min, max, step, group, help, needs_map_reset)
# type is "float", "int", "bool" or a list of allowed strings.
PARAMS = {
    # ---- map ---------------------------------------------------------------
    "resolution_m": (0.05, "float", 0.02, 0.2, 0.01, "Map", "Cell size (m). Changing it clears the map.", True),
    "width_m": (8.0, "float", 2.0, 30.0, 0.5, "Map", "Map width (m). Changing it clears the map.", True),
    "height_m": (8.0, "float", 2.0, 30.0, 0.5, "Map", "Map height (m). Changing it clears the map.", True),
    "origin_x_m": (-4.0, "float", -30.0, 0.0, 0.1, "Map", "World x of the map's lower-left corner (start is x=0).", True),
    "origin_y_m": (-4.0, "float", -30.0, 0.0, 0.1, "Map", "World y of the map's lower-left corner (start is y=0).", True),
    "border_enabled": (False, "bool", None, None, None, "Map", "Only explore and score inside the border below.", False),
    "border_min_x": (-0.3, "float", -30.0, 30.0, 0.05, "Map", "Border: smallest x (m, forward from start).", False),
    "border_min_y": (-0.3, "float", -30.0, 30.0, 0.05, "Map", "Border: smallest y (m, left of start).", False),
    "border_max_x": (3.3, "float", -30.0, 30.0, 0.05, "Map", "Border: largest x (m).", False),
    "border_max_y": (2.7, "float", -30.0, 30.0, 0.05, "Map", "Border: largest y (m).", False),
    "map_rotation_deg": (0.0, "float", -180.0, 180.0, 1.0, "Map", "Rotates the view and the saved map images.", False),
    "l_occ": (0.85, "float", 0.1, 3.0, 0.05, "Map", "Log-odds added to a cell a beam ends in.", False),
    "l_free": (0.50, "float", 0.05, 3.0, 0.05, "Map", "Log-odds removed from cells a beam passes through.", False),
    "l_clamp": (4.0, "float", 1.0, 10.0, 0.5, "Map", "Log-odds limit (lower = map changes faster).", False),
    "occ_prob": (0.65, "float", 0.5, 0.95, 0.01, "Map", "Probability above which a cell is a wall.", False),
    "free_prob": (0.40, "float", 0.05, 0.5, 0.01, "Map", "Probability below which a cell is free.", False),
    "clean_isolated": (True, "bool", None, None, None, "Map", "Hide small wall specks (noise), see min_wall_blob_cells.", False),
    "min_wall_blob_cells": (3, "int", 1, 50, 1, "Map", "Wall blobs smaller than this many cells are hidden as noise.", False),

    # ---- scanning ------------------------------------------------------------
    "scan_mode": ("sweep", ["sweep", "step"], None, None, None, "Scan",
                  "sweep = turn the gimbal smoothly (fast); step = stop at each angle and average (accurate).", False),
    "scan_start_deg": (-180.0, "float", -250.0, 0.0, 5.0, "Scan", "Gimbal scan start (deg from chassis front).", False),
    "scan_end_deg": (180.0, "float", 0.0, 250.0, 5.0, "Scan", "Gimbal scan end (deg).", False),
    "scan_speed_dps": (45.0, "float", 5.0, 300.0, 5.0, "Scan", "Gimbal speed during a sweep (deg/s).", False),
    "scan_step_deg": (2.0, "float", 0.5, 15.0, 0.5, "Scan", "Angle between beams (sweep bin width / step size).", False),
    "samples_per_step": (4, "int", 1, 20, 1, "Scan", "ToF readings averaged at each step (step mode).", False),
    "settle_s": (0.12, "float", 0.0, 1.0, 0.02, "Scan", "Wait after each gimbal step before reading (s).", False),
    "scan_pitch_deg": (0.0, "float", -20.0, 25.0, 1.0, "Scan", "Gimbal pitch while scanning.", False),
    "gimbal_move_speed_dps": (240.0, "float", 30.0, 540.0, 10.0, "Scan", "Gimbal speed for repositioning moves.", False),
    "alternate_sweep": (True, "bool", None, None, None, "Scan", "Sweep back and forth to save time.", False),
    "tof_min_m": (0.08, "float", 0.0, 1.0, 0.01, "Scan", "Readings shorter than this are ignored.", False),
    "tof_max_m": (3.0, "float", 0.5, 10.0, 0.1, "Scan", "Readings longer than this are treated as no wall.", False),
    "no_hit_free_m": (1.0, "float", 0.0, 5.0, 0.1, "Scan", "With no wall in range, mark this far along the beam as free.", False),
    "tof_offset_m": (0.08, "float", -0.5, 0.5, 0.005, "Scan", "ToF lens distance in front of the gimbal yaw axis (m).", False),
    "gimbal_offset_x_m": (0.0, "float", -0.3, 0.3, 0.005, "Scan", "Gimbal yaw axis distance in front of chassis centre (m).", False),
    "tof_scale": (1.0, "float", 0.8, 1.2, 0.005, "Scan", "ToF scale correction (set by wall calibration).", False),
    "tof_bias_m": (0.0, "float", -0.3, 0.3, 0.005, "Scan", "ToF offset correction (set by wall calibration).", False),
    "tof_latency_s": (0.05, "float", 0.0, 0.5, 0.005, "Scan", "ToF delay vs gimbal angle (auto calibration).", False),
    "map_while_moving": (False, "bool", None, None, None, "Scan", "Add the forward ToF beam to the map while driving.", False),

    # ---- noise ----------------------------------------------------------------
    "outlier_mad_k": (3.0, "float", 1.0, 10.0, 0.5, "Noise", "Drop readings more than k x MAD from the bin median.", False),
    "outlier_abs_m": (0.05, "float", 0.005, 0.5, 0.005, "Noise", "Always keep readings within this of the median (m).", False),
    "spike_tol_m": (0.25, "float", 0.02, 2.0, 0.01, "Noise", "Drop a beam that differs from both agreeing neighbours by more.", False),
    "min_bin_samples": (1, "int", 1, 10, 1, "Noise", "Beams with fewer good readings are dropped.", False),
    "isolated_join_m": (0.08, "float", 0.0, 0.5, 0.01, "Noise", "Drop a wall point with no neighbouring point this close (0 = off).", False),

    # ---- localisation ----------------------------------------------------------
    "scan_matching": (True, "bool", None, None, None, "Localisation", "Correct odometry drift by matching each scan to the map.", False),
    "match_window_m": (0.15, "float", 0.0, 0.5, 0.01, "Localisation", "Search +/- this far for a better position (m).", False),
    "match_window_deg": (6.0, "float", 0.0, 20.0, 0.5, "Localisation", "Search +/- this many degrees.", False),
    "match_sigma_m": (0.05, "float", 0.01, 0.3, 0.005, "Localisation", "How close a beam must land to a wall to count.", False),
    "match_min_points": (40, "int", 5, 400, 5, "Localisation", "Skip matching with fewer wall points.", False),
    "match_prior_weight": (0.01, "float", 0.0, 0.2, 0.002, "Localisation", "How much to trust odometry/IMU over the scan match (0 = scan only).", False),
    "yaw_drift_dps": (0.0, "float", -2.0, 2.0, 0.001, "Localisation", "Gyro drift removed from the heading (auto calibration).", False),
    "chassis_yaw_sign": (-1, "int", -1, 1, 2, "Localisation", "-1 if the robot reports clockwise yaw as positive.", False),
    "odom_y_sign": (-1, "int", -1, 1, 2, "Localisation", "-1 if the robot reports right as +y.", False),
    "gimbal_yaw_sign": (-1, "int", -1, 1, 2, "Localisation", "-1 if a positive gimbal yaw turns right.", False),
    "cmd_z_sign": (1, "int", -1, 1, 2, "Localisation", "+1 if a positive chassis z command turns left.", False),

    # ---- motion ------------------------------------------------------------------
    "linear_speed_mps": (0.25, "float", 0.05, 1.0, 0.05, "Motion", "Driving speed (m/s).", False),
    "angular_speed_dps": (60.0, "float", 10.0, 300.0, 5.0, "Motion", "Turning speed (deg/s).", False),
    "turn_tolerance_deg": (2.0, "float", 0.5, 10.0, 0.5, "Motion", "Stop turning within this angle.", False),
    "robot_radius_m": (0.20, "float", 0.1, 0.5, 0.01, "Motion", "Closest the robot centre may get to a wall (m).", False),
    "safety_margin_m": (0.10, "float", 0.0, 0.5, 0.01, "Motion", "Paths prefer to stay this much further from walls (m).", False),
    "stop_distance_m": (0.15, "float", 0.05, 1.0, 0.01, "Motion", "Emergency stop when the forward ToF (lens to wall) reads less than this.", False),
    "scan_every_m": (0.6, "float", 0.1, 3.0, 0.05, "Motion", "Stop and scan after driving this far.", False),

    # ---- exploration -----------------------------------------------------------------
    "min_frontier_cells": (4, "int", 1, 100, 1, "Explore", "Ignore frontiers smaller than this.", False),
    "frontier_view_m": (0.35, "float", 0.1, 2.0, 0.05, "Explore", "Drive to within this of a frontier.", False),
    "min_goal_m": (0.25, "float", 0.0, 2.0, 0.05, "Explore", "Frontiers closer than this after a full scan are ignored.", False),
    "gain_weight": (0.02, "float", 0.0, 1.0, 0.005, "Explore", "Prefer bigger frontiers (higher) or closer ones (lower).", False),
    "coverage_goal_pct": (98.0, "float", 10.0, 100.0, 1.0, "Explore", "Stop when border coverage reaches this (border on).", False),
    "max_iterations": (60, "int", 1, 500, 1, "Explore", "Maximum scan-and-move cycles.", False),
    "max_time_s": (900.0, "float", 30.0, 7200.0, 30.0, "Explore", "Maximum mission time (s).", False),
    "return_home": (False, "bool", None, None, None, "Explore", "Drive back to the start when exploration ends.", False),
    "auto_calibrate": (True, "bool", None, None, None, "Explore", "Run auto calibration when a mission starts.", False),
    "calibrate_with_motion": (True, "bool", None, None, None, "Explore", "Calibration may turn and drive the robot a little.", False),

    # ---- auto grid (maze on a square lattice) ---------------------------------------
    "grid_mode": ("auto", ["auto", "fixed", "off"], None, None, None, "Grid",
                  "auto = find the maze grid (angle, cell size, offset); fixed = use grid_cell_m; off = free-form map.", False),
    "grid_cell_m": (0.6, "float", 0.2, 2.0, 0.005, "Grid", "Cell size for fixed mode (auto mode shows what it found).", False),
    "grid_min_cell_m": (0.3, "float", 0.15, 2.0, 0.05, "Grid", "Smallest cell size auto mode tries.", False),
    "grid_max_cell_m": (1.2, "float", 0.2, 3.0, 0.05, "Grid", "Largest cell size auto mode tries.", False),
    "grid_snap_pose": (True, "bool", None, None, None, "Grid", "Correct heading and position so walls sit on the grid.", False),
    "grid_snap_output": (True, "bool", None, None, None, "Grid", "Saved map and scores use the clean grid map.", False),
    "grid_wall_frac": (0.3, "float", 0.05, 1.0, 0.05, "Grid", "Part of an edge that must look like wall to make it a wall.", False),
    "grid_open_frac": (0.5, "float", 0.05, 1.0, 0.05, "Grid", "Part of an edge that must be seen free to make it open.", False),
    "grid_cell_seen_frac": (0.3, "float", 0.05, 1.0, 0.05, "Grid", "Part of a cell that must be seen free to show it as explored.", False),
    "grid_wall_thickness_m": (0.05, "float", 0.01, 0.3, 0.01, "Grid", "Wall thickness drawn in the grid map.", False),
    "grid_drive": (True, "bool", None, None, None, "Grid", "Move cell to cell along the grid (centred) once the grid is found.", False),
    "grid_stop_each_cell": (True, "bool", None, None, None, "Grid", "Stop and re-centre in every cell (off = drive straight runs).", False),
    "grid_scan_each_cell": (False, "bool", None, None, None, "Grid", "Scan in every cell passed, not only at each target.", False),
    "grid_center_tol_m": (0.03, "float", 0.005, 0.2, 0.005, "Grid", "Strafe back to the cell's centre line when further off than this.", False),
    "grid_align_view": (True, "bool", None, None, None, "Grid", "Rotate the view and saved images so the grid is straight.", False),

    # ---- evaluation ------------------------------------------------------------------
    "eval_resolution_m": (0.05, "float", 0.02, 1.0, 0.01, "Evaluate", "Cell size used for accuracy / coverage.", False),
    "wall_tolerance_cells": (1, "int", 0, 5, 1, "Evaluate", "A wall found this many cells off still counts as correct.", False),
    "gt_start_x": (0.3, "float", -30.0, 30.0, 0.05, "Evaluate", "Robot start x in the ground-truth map (m).", False),
    "gt_start_y": (0.3, "float", -30.0, 30.0, 0.05, "Evaluate", "Robot start y in the ground-truth map (m).", False),
    "gt_start_deg": (0.0, "float", -180.0, 180.0, 1.0, "Evaluate", "Robot start heading in the ground-truth map (deg).", False),
}


def defaults():
    return {key: spec[0] for key, spec in PARAMS.items()}


def coerce(key, value):
    """Convert ``value`` to the type of ``key`` and clamp it. Raises ValueError."""
    if key not in PARAMS:
        raise ValueError(f"Unknown SLAM setting '{key}'")
    _, kind, lo, hi, _, _, _, _ = PARAMS[key]
    if isinstance(kind, list):
        value = str(value)
        if value not in kind:
            raise ValueError(f"'{key}' must be one of {kind}")
        return value
    if kind == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    try:
        value = int(round(float(value))) if kind == "int" else float(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{key}' must be a number")
    if key.endswith("_sign"):
        return -1 if value < 0 else 1
    if lo is not None:
        value = max(lo, value)
    if hi is not None:
        value = min(hi, value)
    return value


def schema():
    out = []
    for key, (default, kind, lo, hi, step, group, text, reset) in PARAMS.items():
        out.append({"key": key, "default": default, "type": kind, "min": lo, "max": hi, "step": step,
                    "group": group, "help": text, "reset": reset})
    return out


def _read_yaml(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load(extra=None):
    """Defaults <- settings.yaml slam: <- slam_settings.yaml <- extra."""
    values = defaults()
    layers = [_read_yaml(SETTINGS_PATH).get("slam") or {}, _read_yaml(OVERRIDE_PATH), extra or {}]
    for layer in layers:
        for key, value in layer.items():
            if key in PARAMS:
                values[key] = coerce(key, value)
    return values


def save_overrides(values, path=OVERRIDE_PATH):
    """Write the values that differ from the defaults."""
    base = defaults()
    changed = {k: v for k, v in values.items() if k in base and v != base[k]}
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Written by the SLAM console. Delete this file to go back to the defaults.\n")
        yaml.safe_dump(changed, f, sort_keys=True, allow_unicode=True)
    return path


def copy_values(values):
    return copy.deepcopy(values)
