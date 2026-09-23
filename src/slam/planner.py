"""Frontier detection and path planning on the occupancy grid."""

import heapq
import math

import cv2
import numpy as np

from .grid import FREE, OCCUPIED, UNKNOWN, border_mask

SQRT2 = math.sqrt(2.0)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, SQRT2), (-1, 1, SQRT2), (1, -1, SQRT2), (1, 1, SQRT2)]


def _disk(radius_cells):
    r = max(0, int(math.ceil(radius_cells)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def open_unknown(classes, inside, min_area=12):
    """Unknown cells that belong to real unexplored space.

    Small unknown pockets between beams inside mapped rooms are noise; they
    must neither block paths nor count as frontiers.
    """
    unknown = ((classes == UNKNOWN) & inside).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(unknown, connectivity=4)
    small = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] < min_area)
    return (unknown > 0) & ~np.isin(labels, small[small > 0])


class Costmap:
    """lethal: the robot centre may never enter (wall closer than 60 % of
    robot_radius_m, or unknown). cost: closer than robot_radius_m is very
    expensive (only used to get out of a tight spot), inside the safety
    margin is mildly expensive, so paths run down the middle of corridors."""

    def __init__(self, grid, classes, params):
        self.grid = grid
        self.classes = classes
        inside = border_mask(grid, params)
        occ = ((classes == OCCUPIED) | ~inside).astype(np.uint8)
        rad = params["robot_radius_m"] / grid.res
        hard = cv2.dilate(occ, _disk(rad * 0.6)) > 0
        body = cv2.dilate(occ, _disk(rad)) > 0
        margin = cv2.dilate(occ, _disk(rad + params["safety_margin_m"] / grid.res)) > 0
        self.lethal = hard | open_unknown(classes, np.ones_like(inside))
        self.body = body
        self.cost = np.where(body, 30.0, np.where(margin, 4.0, 0.0)).astype(np.float32)
        self.inside = inside


def dijkstra(costmap, start_rc, bbox=None):
    """Distances (cells) from start over non-lethal cells, plus parents.

    The start cell is always allowed, so the robot can leave a spot that
    became lethal after a new scan.
    """
    rows, cols = costmap.lethal.shape
    r0, c0, r1, c1 = bbox if bbox else (0, 0, rows, cols)
    dist = np.full((rows, cols), np.inf, np.float64)  # float64: heap keys must equal stored values
    parent = np.full((rows, cols, 2), -1, np.int32)
    sr, sc = start_rc
    if not (0 <= sr < rows and 0 <= sc < cols):
        return dist, parent
    lethal = costmap.lethal
    cost = costmap.cost
    dist[sr, sc] = 0.0
    heap = [(0.0, sr, sc)]
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r, c]:
            continue
        for dr, dc, step in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if nr < r0 or nr >= r1 or nc < c0 or nc >= c1 or lethal[nr, nc]:
                continue
            nd = d + step + cost[nr, nc]
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                parent[nr, nc] = (r, c)
                heapq.heappush(heap, (nd, nr, nc))
    return dist, parent


def extract_path(parent, goal_rc):
    path = []
    r, c = goal_rc
    while r >= 0:
        path.append((r, c))
        r, c = parent[r, c]
    return path[::-1]


def known_bbox(classes, margin=3):
    known = np.argwhere(classes != UNKNOWN)
    rows, cols = classes.shape
    if known.size == 0:
        return 0, 0, rows, cols
    r0, c0 = known.min(axis=0) - margin
    r1, c1 = known.max(axis=0) + margin + 1
    return max(0, r0), max(0, c0), min(rows, r1), min(cols, c1)


def find_frontiers(classes, inside, min_cells, blacklist_mask=None):
    """List of (cells Nx2, centroid_rc) for free cells touching unknown ones."""
    free = (classes == FREE) & inside
    # unknown cells right behind a wall are unreachable anyway, and small
    # unknown pockets between beams are noise: neither makes a frontier
    near_wall = cv2.dilate((classes == OCCUPIED).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    unknown = open_unknown(classes, inside & ~near_wall, max(12, min_cells * 3))
    touch = cv2.dilate(unknown.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)) > 0
    frontier = free & touch
    if blacklist_mask is not None:
        frontier &= ~blacklist_mask
    n, labels = cv2.connectedComponents(frontier.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        cells = np.argwhere(labels == i)
        if len(cells) >= min_cells:
            out.append((cells, cells.mean(axis=0)))
    return out


def choose_frontier_goal(grid, classes, pose, params, blacklist_mask=None):
    """Pick the best reachable frontier. Returns dict or None."""
    costmap = Costmap(grid, classes, params)
    frontiers = find_frontiers(classes, costmap.inside, params["min_frontier_cells"], blacklist_mask)
    if not frontiers:
        return {"frontiers": [], "goal": None}
    start = tuple(int(v) for v in grid.world_to_cell(pose[0], pose[1]))
    dist, parent = dijkstra(costmap, start, known_bbox(classes))
    view = _disk(params["frontier_view_m"] / grid.res)
    rr, cc = np.indices(classes.shape)
    far_enough = np.hypot(rr - start[0], cc - start[1]) * grid.res >= params["min_goal_m"]
    best = None
    for cells, centroid in frontiers:
        mask = np.zeros(classes.shape, np.uint8)
        mask[cells[:, 0], cells[:, 1]] = 1
        near = (cv2.dilate(mask, view) > 0) & np.isfinite(dist) & far_enough & ~costmap.body
        if not near.any():
            continue
        cand = np.argwhere(near)
        d = dist[cand[:, 0], cand[:, 1]]
        i = int(np.argmin(d))
        score = float(d[i]) * grid.res - params["gain_weight"] * len(cells)
        if best is None or score < best["score"]:
            best = {"score": score, "goal_rc": tuple(int(v) for v in cand[i]), "frontier": cells,
                    "centroid": centroid}
    info = {"frontiers": [(c, cells) for cells, c in frontiers], "goal": None}
    if best is None:
        return info
    path = extract_path(parent, best["goal_rc"])
    info.update(goal=best["goal_rc"], path=path, frontier=best["frontier"], centroid=best["centroid"])
    return info


def plan_to(grid, classes, pose, goal_xy, params):
    """Path (list of cells) from pose to the nearest reachable cell around goal."""
    costmap = Costmap(grid, classes, params)
    start = tuple(int(v) for v in grid.world_to_cell(pose[0], pose[1]))
    dist, parent = dijkstra(costmap, start, known_bbox(classes))
    gr, gc = (int(v) for v in grid.world_to_cell(goal_xy[0], goal_xy[1]))
    reach = np.argwhere(np.isfinite(dist) & ~costmap.body)
    if reach.size == 0:
        return None
    d2 = (reach[:, 0] - gr) ** 2 + (reach[:, 1] - gc) ** 2
    goal = tuple(int(v) for v in reach[int(np.argmin(d2))])
    return extract_path(parent, goal)


def line_is_clear(lethal, a, b):
    (r0, c0), (r1, c1) = a, b
    n = int(max(abs(r1 - r0), abs(c1 - c0)) * 2) + 1
    rr = np.rint(np.linspace(r0, r1, n)).astype(int)
    cc = np.rint(np.linspace(c0, c1, n)).astype(int)
    return not lethal[rr[1:], cc[1:]].any()


def simplify_path(path, lethal):
    """Greedy line-of-sight shortcutting -> list of waypoint cells."""
    if len(path) <= 2:
        return list(path)
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_is_clear(lethal, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out
