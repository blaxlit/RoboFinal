"""Map images and the end-of-mission report."""

import json
import math
import os

import cv2
import numpy as np

from .grid import FREE, OCCUPIED, UNKNOWN


def _rotate(img, deg, border_value):
    if abs(deg) < 1e-6:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_NEAREST, borderValue=border_value)


def render_map(grid, classes, params, trajectory=None, start=None, end=None, walls=None, scale=None):
    """BGR image of the map (north = +y up) with trajectory and markers."""
    scale = scale or max(1, int(round(160 * grid.res)))  # 160 px per metre
    flipped = np.flipud(classes)  # image row 0 = largest y
    img = np.full((grid.rows, grid.cols, 3), 170, np.uint8)
    img[flipped == FREE] = (255, 255, 255)
    img = cv2.resize(img, (grid.cols * scale, grid.rows * scale), interpolation=cv2.INTER_NEAREST)

    def px(x, y):
        return (int((x - grid.origin_x) / grid.res * scale),
                int((grid.rows - (y - grid.origin_y) / grid.res) * scale))

    # 0.5 m grid lines
    step = 0.5
    x0, y0, x1, y1 = grid.extent()
    for gx in np.arange(math.ceil(x0 / step) * step, x1, step):
        cv2.line(img, px(gx, y0), px(gx, y1), (215, 215, 215), 1)
    for gy in np.arange(math.ceil(y0 / step) * step, y1, step):
        cv2.line(img, px(x0, gy), px(x1, gy), (215, 215, 215), 1)
    img[np.kron(flipped == OCCUPIED, np.ones((scale, scale), bool))] = (30, 30, 30)

    if params.get("border_enabled"):
        cv2.rectangle(img, px(params["border_min_x"], params["border_max_y"]),
                      px(params["border_max_x"], params["border_min_y"]), (200, 120, 0), 2)
    if walls:
        for x1_, y1_, x2_, y2_ in walls:
            cv2.line(img, px(x1_, y1_), px(x2_, y2_), (80, 180, 80), 1, cv2.LINE_AA)
    if trajectory is not None and len(trajectory) > 1:
        pts = np.array([px(x, y) for x, y in trajectory], np.int32)
        cv2.polylines(img, [pts], False, (230, 90, 30), 2, cv2.LINE_AA)
    for pose, color, label in ((start, (40, 170, 40), "START"), (end, (40, 40, 220), "END")):
        if pose is None:
            continue
        c = px(pose[0], pose[1])
        tip = px(pose[0] + 0.18 * math.cos(pose[2]), pose[1] + 0.18 * math.sin(pose[2]))
        cv2.circle(img, c, 7, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(img, c, tip, color, 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(img, label, (c[0] + 9, c[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    img = crop_to_content(img, classes, scale)
    img = _rotate(img, params.get("map_rotation_deg", 0.0), (170, 170, 170))
    # scale bar
    bar = int(1.0 / grid.res * scale)
    h = img.shape[0]
    cv2.line(img, (10, h - 12), (10 + bar, h - 12), (0, 0, 0), 3)
    cv2.putText(img, "1 m", (14, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def crop_to_content(img, classes, scale, margin=20):
    known = np.argwhere(np.flipud(classes) != UNKNOWN)
    if known.size == 0:
        return img
    r0, c0 = known.min(axis=0) - margin
    r1, c1 = known.max(axis=0) + margin
    r0, c0 = max(0, r0), max(0, c0)
    return img[r0 * scale:(r1 + 1) * scale, c0 * scale:(c1 + 1) * scale].copy()


def write_grid_csv(path, classes, grid):
    """Top row = largest y. 0 unknown, 1 free, 2 wall."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# resolution_m={grid.res}, origin_x_m={grid.origin_x}, origin_y_m={grid.origin_y}, "
                f"values: 0=unknown 1=free 2=wall, first row = top (largest y)\n")
        for row in np.flipud(classes):
            f.write(",".join(str(int(v)) for v in row) + "\n")


def write_report(out_dir, report):
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    s, e = report["start"], report["end"]
    m = report.get("metrics", {})
    lines = [
        "# SLAM exploration report",
        "",
        f"- Run folder: `{out_dir}`",
        f"- Robot: {report['robot']}",
        f"- Started: {report['started_at']}   Duration: {report['duration_s']:.1f} s",
        f"- Scans: {report['scans']}   Distance driven: {report['distance_m']:.2f} m   "
        f"Emergency stops: {report['blocked_moves']}",
        f"- Finished because: {report['finish_reason']}",
        "",
        "## Where the robot started and ended",
        "",
        "| | x (m) | y (m) | heading (deg) |",
        "| --- | --- | --- | --- |",
        f"| Start (map frame) | {s['x']:.3f} | {s['y']:.3f} | {s['deg']:.1f} |",
        f"| End (map frame) | {e['x']:.3f} | {e['y']:.3f} | {e['deg']:.1f} |",
    ]
    if "gt" in s:
        lines += [f"| Start (ground-truth frame) | {s['gt']['x']:.3f} | {s['gt']['y']:.3f} | {s['gt']['deg']:.1f} |",
                  f"| End (ground-truth frame) | {e['gt']['x']:.3f} | {e['gt']['y']:.3f} | {e['gt']['deg']:.1f} |"]
    if "true_end" in report:
        t = report["true_end"]
        lines += [f"| End (simulator truth) | {t['x']:.3f} | {t['y']:.3f} | {t['deg']:.1f} |",
                  "", f"Localisation error at the end: {report['end_error_m'] * 100:.1f} cm, "
                      f"{report['end_error_deg']:.1f} deg"]
    lines += ["", "## Scores", ""]
    if m.get("has_ground_truth"):
        lines += [
            f"- **Map Accuracy = {m['accuracy_pct']:.2f} %** ({m['correct_cells']} correct / {m['total_cells']} cells, "
            f"cell {m['eval_cell_m']} m, wall tolerance {m['wall_tolerance_cells']} cell)",
            f"- Strict accuracy (no tolerance): {m['strict_accuracy_pct']:.2f} %",
            f"- Accuracy of explored cells only: {m['explored_accuracy_pct']:.2f} %",
            f"- **Coverage = {m['coverage_pct']:.2f} %** ({m['explored_cells']} explored / {m['total_cells']} cells)",
            f"- Walls found {m['wall_found']}, missed {m['wall_missed']}, false walls {m['false_walls']}",
            "",
            "Unknown cells never count as correct. With a wall tolerance, a wall found a cell away counts as "
            "correct even if the exact wall cell was not seen, so Map Accuracy can be a little above Coverage; "
            "the strict accuracy has no tolerance.",
        ]
    elif "coverage_pct" in m:
        lines.append(f"- **Coverage (inside border) = {m['coverage_pct']:.2f} %** — load a ground truth for accuracy")
    else:
        lines.append(f"- Mapped area: {m.get('known_area_m2', 0)} m² — load a ground truth or set a border for scores")
    if report.get("calibration"):
        lines += ["", "## Calibration", ""] + [f"- {k}: {v}" for k, v in report["calibration"].items()]
    lines += ["", "## Files", "",
              "- `map.png` map with trajectory, start (green) and end (red)",
              "- `map_grid.csv` / `map_prob.npy` the grid itself",
              "- `trajectory.csv` robot path (map + odometry frames)",
              "- `log_events.csv`, `log_scans_raw.csv`, `log_scans_filtered.csv` exploration logs",
              "- `comparison.png` ground truth check (white/black correct, red missed wall, orange false wall, grey unexplored)"]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
