"""Ground-truth maps and the Map Accuracy / Coverage scores.

Ground-truth file (JSON, metres, in the arena's own frame):

    {
      "name": "Lab maze",
      "border": [0, 0, 3.6, 3.0],            # xmin, ymin, xmax, ymax of the arena
      "wall_thickness": 0.02,
      "walls": [[x1, y1, x2, y2], ...]       # wall centre lines
    }

The robot's start pose inside that frame (gt_start_x / gt_start_y /
gt_start_deg) links it to the SLAM map, whose origin is the start pose.

    Map Accuracy = cells guessed right / all cells x 100
    Coverage     = explored cells     / all cells x 100

"all cells" are the arena cells (inside the ground-truth border), counted at
eval_resolution_m. A cell counts as guessed right when its SLAM class
(wall / free) matches the ground truth; unknown cells are never right. With
wall_tolerance_cells > 0, a wall found up to that many cells away still counts.
"""

import json
import math

import cv2
import numpy as np

from .grid import FREE, OCCUPIED, UNKNOWN


def load_ground_truth(path):
    with open(path, "r", encoding="utf-8") as f:
        return validate_ground_truth(json.load(f))


def validate_ground_truth(data):
    walls = [[float(v) for v in w] for w in data.get("walls", [])]
    if any(len(w) != 4 for w in walls):
        raise ValueError("each wall must be [x1, y1, x2, y2]")
    border = data.get("border")
    if border is None:
        if not walls:
            raise ValueError("ground truth needs walls or a border")
        arr = np.asarray(walls)
        border = [float(min(arr[:, 0].min(), arr[:, 2].min())), float(min(arr[:, 1].min(), arr[:, 3].min())),
                  float(max(arr[:, 0].max(), arr[:, 2].max())), float(max(arr[:, 1].max(), arr[:, 3].max()))]
    border = [float(v) for v in border]
    if len(border) != 4 or border[2] <= border[0] or border[3] <= border[1]:
        raise ValueError("border must be [xmin, ymin, xmax, ymax] with max > min")
    return {"name": str(data.get("name", "ground truth")), "border": border,
            "wall_thickness": float(data.get("wall_thickness", 0.02)), "walls": walls}


def border_walls(border):
    x0, y0, x1, y1 = border
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]


def gt_to_world(params):
    """Function mapping ground-truth (x, y) into the SLAM world frame."""
    sx, sy, th = params["gt_start_x"], params["gt_start_y"], math.radians(params["gt_start_deg"])
    c, s = math.cos(-th), math.sin(-th)

    def fn(x, y):
        dx, dy = np.asarray(x) - sx, np.asarray(y) - sy
        return c * dx - s * dy, s * dx + c * dy
    return fn


def world_to_gt(params):
    sx, sy, th = params["gt_start_x"], params["gt_start_y"], math.radians(params["gt_start_deg"])
    c, s = math.cos(th), math.sin(th)

    def fn(x, y):
        x, y = np.asarray(x), np.asarray(y)
        return sx + c * x - s * y, sy + s * x + c * y
    return fn


def gt_walls_world(gt, params):
    """All walls (incl. the arena border) as world-frame segments."""
    tf = gt_to_world(params)
    out = []
    for x1, y1, x2, y2 in gt["walls"] + border_walls(gt["border"]):
        a = tf(x1, y1)
        b = tf(x2, y2)
        out.append([float(a[0]), float(a[1]), float(b[0]), float(b[1])])
    return out


def rasterize_gt(gt, grid, params):
    """(gt_class, arena_mask) on the SLAM grid."""
    tf = gt_to_world(params)
    occ = np.zeros((grid.rows, grid.cols), np.uint8)
    thick = max(1, int(round(gt["wall_thickness"] / grid.res)))

    def to_px(x, y):
        wx, wy = tf(x, y)
        return (int(round((wx - grid.origin_x) / grid.res - 0.5)), int(round((wy - grid.origin_y) / grid.res - 0.5)))

    for x1, y1, x2, y2 in gt["walls"] + border_walls(gt["border"]):
        cv2.line(occ, to_px(x1, y1), to_px(x2, y2), 1, thick)
    x0, y0, x1, y1 = gt["border"]
    poly = np.array([to_px(x0, y0), to_px(x1, y0), to_px(x1, y1), to_px(x0, y1)], np.int32)
    arena = np.zeros_like(occ)
    cv2.fillPoly(arena, [poly], 1)
    arena = (arena > 0) | (occ > 0)
    cls = np.where(occ > 0, OCCUPIED, FREE).astype(np.uint8)
    return cls, arena


def _block_reduce(classes, arena, k, is_gt):
    if k <= 1:
        return classes, arena
    rows, cols = (classes.shape[0] // k) * k, (classes.shape[1] // k) * k
    c = classes[:rows, :cols].reshape(rows // k, k, cols // k, k)
    a = arena[:rows, :cols].reshape(rows // k, k, cols // k, k)
    occ = (c == OCCUPIED).any(axis=(1, 3))
    free_frac = (c == FREE).mean(axis=(1, 3))
    out = np.full(occ.shape, UNKNOWN, np.uint8)
    if is_gt:
        out[:] = FREE
    else:
        out[free_frac >= 0.5] = FREE
    out[occ] = OCCUPIED
    return out, a.any(axis=(1, 3))


def evaluate(est_classes, grid, params, gt=None, border=None):
    """Scores as a dict. Without ground truth only coverage of the border is given."""
    k = max(1, int(round(params["eval_resolution_m"] / grid.res)))
    result = {"eval_cell_m": round(grid.res * k, 4), "has_ground_truth": gt is not None}
    if gt is None:
        if border is None:
            known = int((est_classes != UNKNOWN).sum())
            result.update(known_cells=known, known_area_m2=round(known * grid.res ** 2, 3))
            return result
        est, area = _block_reduce(est_classes, border, k, False)
        total = int(area.sum())
        explored = int(((est != UNKNOWN) & area).sum())
        result.update(total_cells=total, explored_cells=explored,
                      coverage_pct=round(100.0 * explored / total, 2) if total else 0.0)
        return result

    gt_cls, arena = rasterize_gt(gt, grid, params)
    gt_cls, arena = _block_reduce(gt_cls, arena, k, True)
    est, _ = _block_reduce(est_classes, arena, k, False)
    tol = int(params["wall_tolerance_cells"])
    kernel = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    est_occ = est == OCCUPIED
    gt_occ = gt_cls == OCCUPIED
    est_occ_d = cv2.dilate(est_occ.astype(np.uint8), kernel) > 0
    gt_occ_d = cv2.dilate(gt_occ.astype(np.uint8), kernel) > 0

    correct = np.zeros_like(arena)
    correct |= gt_occ & est_occ_d
    correct |= (~gt_occ) & (est == FREE)
    correct |= (~gt_occ) & est_occ & gt_occ_d
    correct &= arena
    strict = arena & (((gt_occ) & est_occ) | ((~gt_occ) & (est == FREE)))
    explored = arena & (est != UNKNOWN)
    total = int(arena.sum())
    n_correct = int(correct.sum())
    n_explored = int(explored.sum())
    n_explored_correct = int((correct & explored).sum())
    wall_tp = int((gt_occ & est_occ_d & arena).sum())
    wall_fp = int((est_occ & ~gt_occ_d & arena).sum())
    wall_fn = int((gt_occ & ~est_occ_d & arena).sum())
    result.update(
        total_cells=total,
        correct_cells=n_correct,
        explored_cells=n_explored,
        accuracy_pct=round(100.0 * n_correct / total, 2) if total else 0.0,
        strict_accuracy_pct=round(100.0 * int(strict.sum()) / total, 2) if total else 0.0,
        explored_accuracy_pct=round(100.0 * n_explored_correct / n_explored, 2) if n_explored else 0.0,
        coverage_pct=round(100.0 * n_explored / total, 2) if total else 0.0,
        wall_found=wall_tp, wall_missed=wall_fn, false_walls=wall_fp,
        wall_tolerance_cells=tol,
    )
    result["_images"] = {"gt": gt_cls, "est": est, "arena": arena, "correct": correct}
    return result


def comparison_image(images, scale=6):
    """BGR image: white/black = correct free/wall, red = missed wall,
    orange = wall where there is none, grey = unexplored, dark = outside."""
    gt, est, arena, correct = images["gt"], images["est"], images["arena"], images["correct"]
    img = np.full(gt.shape + (3,), 40, np.uint8)
    img[arena & (est == UNKNOWN)] = (150, 150, 150)
    img[arena & correct & (gt == FREE)] = (255, 255, 255)
    img[arena & correct & (gt == OCCUPIED)] = (0, 0, 0)
    img[arena & correct & (gt == FREE) & (est == OCCUPIED)] = (0, 0, 0)
    img[arena & ~correct & (gt == OCCUPIED)] = (40, 40, 230)
    img[arena & ~correct & (gt == FREE) & (est == OCCUPIED)] = (0, 150, 255)
    rows, cols = np.nonzero(arena)
    if rows.size:
        img = img[max(0, rows.min() - 2):rows.max() + 3, max(0, cols.min() - 2):cols.max() + 3]
    img = np.flipud(img)
    return cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
