"""Auto grid: the arena is a maze whose walls lie on a square lattice.

1. detect()   finds the lattice from the wall points: its angle (walls meet at
              90 degrees), cell size and offset. Points are grouped by the
              direction of the wall they lie on, projected across it, and the
              cell size / offset that puts the most points on lattice lines
              wins (minus what a random lattice would catch).
2. snap_pose() uses the lattice to correct each scan: heading from the wall
              directions, position from how far the walls sit off the lines.
3. classify_edges() votes every lattice edge wall / open / unknown from the
              occupancy map. A wall seen along part of an edge becomes the
              whole edge (fills gaps); stray blobs off the lines disappear.
4. render()   draws the clean snapped map; unknown_edge_targets() says where
              to look next.
"""

import math

import cv2
import numpy as np

from .grid import FREE, OCCUPIED, UNKNOWN

EDGE_UNKNOWN, EDGE_OPEN, EDGE_WALL = 0, 1, 2


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def oriented_points(xs, ys, max_step=0.08):
    """Wall points from one scan (in beam order) with the direction of the
    wall through them, from their neighbours (two on each side when they are
    on the same wall, for a steadier direction). Returns (pts Nx2, angles N)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    n = len(xs)
    if n < 3:
        return np.zeros((0, 2)), np.zeros(0)
    p = np.stack([xs, ys], axis=1)
    prev, nxt = np.roll(p, 1, axis=0), np.roll(p, -1, axis=0)
    ok = (np.hypot(*(p - prev).T) < max_step) & (np.hypot(*(nxt - p).T) < max_step)
    ang = np.arctan2(nxt[:, 1] - prev[:, 1], nxt[:, 0] - prev[:, 0])
    if n >= 5:
        prev2, nxt2 = np.roll(p, 2, axis=0), np.roll(p, -2, axis=0)
        wide = ok & (np.hypot(*(prev - prev2).T) < max_step) & (np.hypot(*(nxt2 - nxt).T) < max_step)
        ang = np.where(wide, np.arctan2(nxt2[:, 1] - prev2[:, 1], nxt2[:, 0] - prev2[:, 0]), ang)
    return p[ok], ang[ok]


class GridModel:
    def __init__(self, theta, cell, ou, ov, score, n_points, cell_trusted, theta_locked=False, margin=0.0):
        self.theta, self.cell, self.ou, self.ov = theta, cell, ou, ov
        self.score, self.n_points, self.cell_trusted = score, n_points, cell_trusted
        self.theta_locked = theta_locked
        self.margin = margin

    def to_grid(self, x, y):
        c, s = math.cos(self.theta), math.sin(self.theta)
        x, y = np.asarray(x, float), np.asarray(y, float)
        return c * x + s * y - self.ou, -s * x + c * y - self.ov

    def to_world(self, u, v):
        c, s = math.cos(self.theta), math.sin(self.theta)
        u, v = np.asarray(u, float) + self.ou, np.asarray(v, float) + self.ov
        return c * u - s * v, s * u + c * v

    def as_dict(self):
        return {"theta_deg": round(math.degrees(self.theta), 3), "cell_m": round(self.cell, 4),
                "offset_u_m": round(self.ou, 4), "offset_v_m": round(self.ov, 4), "score": round(self.score, 3),
                "points": self.n_points, "cell_trusted": self.cell_trusted, "theta_locked": self.theta_locked,
                "margin": round(self.margin, 3)}


def _direction_split(ang, theta, tol_deg=15.0):
    """Masks of points on walls running along v (constant u) and along u."""
    rel = np.mod(ang - theta, math.pi)
    tol = math.radians(tol_deg)
    along_v = np.abs(rel - math.pi / 2) < tol
    along_u = (rel < tol) | (rel > math.pi - tol)
    return along_v, along_u


def _sharpness(pts, ang, theta, pivot=(0.0, 0.0), dths=(0.0,)):
    """For each rotation dth of the points about ``pivot``: how tightly the
    points pile up on lines along the grid axes (sum of squared 1 cm
    histogram counts). Walls line up best at the right angle."""
    along_v, along_u = _direction_split(ang, theta, 20.0)
    out = []
    c0, s0 = math.cos(theta), math.sin(theta)
    for dth in dths:
        c, s = math.cos(dth), math.sin(dth)
        x = pivot[0] + c * (pts[:, 0] - pivot[0]) - s * (pts[:, 1] - pivot[1])
        y = pivot[1] + s * (pts[:, 0] - pivot[0]) + c * (pts[:, 1] - pivot[1])
        u = c0 * x + s0 * y
        v = -s0 * x + c0 * y
        score = 0.0
        for vals in (u[along_v], v[along_u]):
            if len(vals):
                h = np.bincount(((vals - vals.min()) / 0.01).astype(int))
                score += float((h.astype(float) ** 2).sum())
        out.append(score)
    return np.array(out)


def _refine_theta(pts, ang, theta, span_deg=4.0, step_deg=0.1):
    dths = np.radians(np.arange(-span_deg, span_deg + 1e-9, step_deg))
    # rotating the points by +d is the same as the grid being at theta - d
    best = dths[int(np.argmax(_sharpness(pts, ang, theta, (0.0, 0.0), dths)))]
    return theta - best


def _comb(x, cell, win=0.02, step=0.005):
    """Best lattice offset for positions x: (excess fraction on lines, offset)."""
    nb = max(8, int(round(cell / step)))
    ph = (np.mod(x, cell) / cell * nb).astype(int) % nb
    h = np.bincount(ph, minlength=nb).astype(float)
    w = max(1, int(round(win / step)))
    k = np.convolve(np.concatenate([h[-w:], h, h[:w]]), np.ones(2 * w + 1), "valid")
    i = int(np.argmax(k))
    return k[i] / max(len(x), 1) - (2 * w + 1) / nb, (i + 0.5) * cell / nb


def detect(pts, ang, params, previous=None, history=(), final=False, anchor=None):
    """Fit the lattice to oriented wall points. None if there is too little.

    history: earlier models (oldest first) used to confirm the cell size.
    final: relaxed rules, for drawing the saved map from all the data.
    anchor: (x, y) that is a cell centre (the start pose): the lattice offset is
    taken from it instead of fitted, which is exact from the first scan.
    """
    if len(pts) < 40:
        return None
    if previous is not None and previous.theta_locked and not final:
        theta = previous.theta
    else:
        # angle: wall directions are 90 degrees apart -> 4x angle trick, then refine
        z = np.exp(4j * np.mod(ang, math.pi / 2))
        theta = float(np.angle(z.mean()) / 4)
        theta = _refine_theta(pts, ang, _refine_theta(pts, ang, theta, 8.0, 0.5), 1.0, 0.05)
        if previous is not None:  # keep the same branch of the 90-degree ambiguity
            theta = previous.theta + _wrap(4 * (theta - previous.theta)) / 4
    recent_t = [h for h in history if h is not None][-2:]
    theta_locked = (previous is not None and previous.theta_locked) or (
        len(recent_t) == 2 and all(abs(_wrap(4 * (h.theta - theta)) / 4) < math.radians(0.5) for h in recent_t))
    along_v, along_u = _direction_split(ang, theta)
    if along_v.sum() < 10 or along_u.sum() < 10:
        return None
    c, s = math.cos(theta), math.sin(theta)
    U = c * pts[:, 0] + s * pts[:, 1]
    V = -s * pts[:, 0] + c * pts[:, 1]
    fixed = params["grid_mode"] == "fixed"
    cells = [params["grid_cell_m"]] if fixed else np.arange(params["grid_min_cell_m"], params["grid_max_cell_m"] + 1e-9, 0.005)
    results = []
    for cell in cells:
        su, ou = _comb(U[along_v], cell)
        sv, ov = _comb(V[along_u], cell)
        results.append((su + sv, float(cell), ou, ov))
    score, cell, ou, ov = max(results)
    if anchor is not None:
        a0 = c * anchor[0] + s * anchor[1]
        a1 = -s * anchor[0] + c * anchor[1]
        ou, ov = (a0 - cell / 2) % cell, (a1 - cell / 2) % cell
    # how clearly this cell size beats a different one (not a neighbour of it)
    def related(c):  # neighbours and exact multiples / fractions fit the same walls
        ratio = c / cell if c > cell else cell / c
        return abs(ratio - round(ratio)) < 0.04 * round(ratio)
    others = [r[0] for r in results if not related(r[1])]
    margin = score / max(others) - 1.0 if others and max(others) > 0 else 1.0
    # the cell size needs walls from several cells, and must come out the same
    # again once more walls have been seen, before it is trusted (and locked)
    span = min(np.ptp(U[along_v]), np.ptp(V[along_u]))
    span_min = min(np.ptp(U[along_v]), np.ptp(V[along_u]))
    plausible = score > 0.35 and span_min > 3.0 * cell and len(pts) > 400 and margin > 0.15
    recent = [h for h in history if h is not None][-3:]
    confirmed = len(recent) == 3 and all(abs(h.cell - cell) <= 0.012 and h.margin > 0.1 for h in recent) \
        and len(pts) >= 1.3 * recent[0].n_points
    if final:
        trusted = fixed or (score > 0.3 and span > 1.5 * cell)
    else:
        # fixed: the cell size is known, so the grid can be used at once; the angle
        # keeps being refined (from unsnapped scans) until it is locked
        trusted = (fixed and span > 0.8 * cell) or (plausible and confirmed and theta_locked)
    return GridModel(theta, cell, ou, ov, float(score), int(len(pts)), bool(trusted), theta_locked, margin)


def snap_pose(model, pose, pts, ang, max_deg=5.0, max_shift=0.12):
    """Correction (dx, dy, dtheta) that lines one scan up with the lattice.

    Heading: median deviation of the scan's wall directions from the lattice.
    Position: median distance of wall points from the nearest lattice line,
    across the wall (only once the cell size is trusted).
    """
    if len(pts) < 15 or not model.theta_locked:
        return 0.0, 0.0, 0.0
    dev = np.angle(np.exp(4j * (ang - model.theta))) / 4
    near = np.abs(dev) < math.radians(max_deg + 10)
    if near.sum() < 15:
        return 0.0, 0.0, 0.0
    pts, ang = pts[near], ang[near]
    dths = np.radians(np.arange(-max_deg, max_deg + 1e-9, 0.1))
    sharp = _sharpness(pts, ang, model.theta, (pose[0], pose[1]), dths)
    dth = float(dths[int(np.argmax(sharp))])
    if sharp.max() < 1.05 * sharp[len(dths) // 2]:
        dth = 0.0  # no clear improvement: keep the heading
    # rotate the scan about the robot, then measure the offsets
    x0, y0 = pose[0], pose[1]
    c, s = math.cos(dth), math.sin(dth)
    rx = x0 + c * (pts[:, 0] - x0) - s * (pts[:, 1] - y0)
    ry = y0 + s * (pts[:, 0] - x0) + c * (pts[:, 1] - y0)
    if not model.cell_trusted:
        return 0.0, 0.0, dth
    along_v, along_u = _direction_split(ang + dth, model.theta, 10.0)
    u, v = model.to_grid(rx, ry)
    du = dv = 0.0
    cell = model.cell
    if along_v.sum() >= 8:
        r = u[along_v] - cell * np.round(u[along_v] / cell)
        m = float(np.median(r))
        if abs(m) < max_shift:
            du = -m
    if along_u.sum() >= 8:
        r = v[along_u] - cell * np.round(v[along_u] / cell)
        m = float(np.median(r))
        if abs(m) < max_shift:
            dv = -m
    ct, st = math.cos(model.theta), math.sin(model.theta)
    return ct * du - st * dv, st * du + ct * dv, dth


class EdgeMap:
    """States of the lattice edges in index range [i0, i1) x [j0, j1) cells.

    vert[k, j]: line u = k*cell between v = j*cell and (j+1)*cell, k in i0..i1
    horz[i, k]: line v = k*cell between u = i*cell and (i+1)*cell, k in j0..j1
    """

    def __init__(self, i0, i1, j0, j1):
        self.i0, self.i1, self.j0, self.j1 = i0, i1, j0, j1
        self.vert = np.zeros((i1 - i0 + 1, j1 - j0), np.uint8)
        self.horz = np.zeros((i1 - i0, j1 - j0 + 1), np.uint8)
        self.vert_ev = np.zeros(self.vert.shape + (2,), np.float32)   # (wall, open) fractions
        self.horz_ev = np.zeros(self.horz.shape + (2,), np.float32)
        self.cell_free = np.zeros((i1 - i0, j1 - j0), np.float32)     # free fraction inside each cell

    def segments(self, model, state=None):
        """World segments [(x1, y1, x2, y2, state)]."""
        c = model.cell
        out = []
        for a in range(self.vert.shape[0]):
            for b in range(self.vert.shape[1]):
                st = int(self.vert[a, b])
                if state is None or st == state:
                    k, j = self.i0 + a, self.j0 + b
                    (x1, x2), (y1, y2) = model.to_world([k * c, k * c], [j * c, (j + 1) * c])
                    out.append((float(x1), float(y1), float(x2), float(y2), st))
        for a in range(self.horz.shape[0]):
            for b in range(self.horz.shape[1]):
                st = int(self.horz[a, b])
                if state is None or st == state:
                    i, k = self.i0 + a, self.j0 + b
                    (x1, x2), (y1, y2) = model.to_world([i * c, (i + 1) * c], [k * c, k * c])
                    out.append((float(x1), float(y1), float(x2), float(y2), st))
        return out


def _sample_classes(grid, classes, xs, ys):
    r, c = grid.world_to_cell(xs, ys)
    ok = grid.inside(r, c)
    out = np.full(np.shape(xs), UNKNOWN, np.uint8)
    out[ok] = classes[r[ok], c[ok]]
    return out


def classify_edges(model, grid, classes, params):
    """Vote every lattice edge inside the mapped area."""
    known = np.argwhere(classes != UNKNOWN)
    if known.size == 0:
        return None
    xs, ys = grid.cell_to_world(known[:, 0], known[:, 1])
    u, v = model.to_grid(xs, ys)
    cell = model.cell
    # one extra ring of cells, so openings to unseen cells show up as open edges
    i0, i1 = int(math.floor(u.min() / cell)) - 1, int(math.ceil(u.max() / cell)) + 1
    j0, j1 = int(math.floor(v.min() / cell)) - 1, int(math.ceil(v.max() / cell)) + 1
    if (i1 - i0) * (j1 - j0) > 2500:
        return None
    em = EdgeMap(i0, i1, j0, j1)
    res = grid.res
    n_along = max(6, int(cell / res * 1.5))
    t = np.linspace(0.12, 0.88, n_along) * cell                    # skip the corners
    band = max(0.06, 1.6 * res)
    offs = np.arange(-band, band + 1e-9, res / 2)
    wall_frac, open_frac = params["grid_wall_frac"], params["grid_open_frac"]

    def edge_state(pu, pv, across_u):
        # pu, pv: points along the edge in grid coords; across_u: offsets move in u (vertical edge) or v
        if across_u:
            gu = pu[:, None] + offs[None, :]
            gv = np.repeat(pv[:, None], len(offs), axis=1)
        else:
            gu = np.repeat(pu[:, None], len(offs), axis=1)
            gv = pv[:, None] + offs[None, :]
        wx, wy = model.to_world(gu, gv)
        cls = _sample_classes(grid, classes, wx, wy)
        wall = (cls == OCCUPIED).any(axis=1)
        crossing = (cls == FREE).all(axis=1)
        fw, ff = float(wall.mean()), float(crossing.mean())
        seen = float((cls != UNKNOWN).any(axis=1).mean())
        if fw >= wall_frac or (fw >= wall_frac / 2 and ff < 0.1 and seen - fw < 0.3):
            st = EDGE_WALL
        elif ff >= open_frac and fw < wall_frac / 2:
            st = EDGE_OPEN
        else:
            st = EDGE_UNKNOWN
        return st, fw, ff

    for a in range(em.vert.shape[0]):
        for b in range(em.vert.shape[1]):
            k, j = i0 + a, j0 + b
            st, fw, ff = edge_state(np.full(n_along, k * cell), j * cell + t, True)
            em.vert[a, b] = st
            em.vert_ev[a, b] = (fw, ff)
    for a in range(em.horz.shape[0]):
        for b in range(em.horz.shape[1]):
            i, k = i0 + a, j0 + b
            st, fw, ff = edge_state(i * cell + t, np.full(n_along, k * cell), False)
            em.horz[a, b] = st
            em.horz_ev[a, b] = (fw, ff)
    # free fraction inside each cell (away from its walls)
    inner = np.linspace(0.2, 0.8, 5) * cell
    gu, gv = np.meshgrid(inner, inner)
    for a in range(em.cell_free.shape[0]):
        for b in range(em.cell_free.shape[1]):
            wx, wy = model.to_world((i0 + a) * cell + gu, (j0 + b) * cell + gv)
            cls = _sample_classes(grid, classes, wx, wy)
            em.cell_free[a, b] = float((cls == FREE).mean())
    # a wall needs a seen cell on at least one side (drops junk outside the arena)
    seen = em.cell_free >= params["grid_cell_seen_frac"]
    pad = np.pad(seen, 1)
    left, right = pad[:-1, 1:-1], pad[1:, 1:-1]          # cells on each side of vertical edges
    below, above = pad[1:-1, :-1], pad[1:-1, 1:]           # cells on each side of horizontal edges
    em.vert[(em.vert == EDGE_WALL) & ~(left | right)] = EDGE_UNKNOWN
    em.horz[(em.horz == EDGE_WALL) & ~(below | above)] = EDGE_UNKNOWN
    return em


def render(model, edges, grid, params):
    """Clean class map: seen cells free, wall edges drawn, everything else unknown."""
    out = np.full((grid.rows, grid.cols), UNKNOWN, np.uint8)
    if edges is None:
        return out
    cell = model.cell

    def px(u, v):
        x, y = model.to_world(u, v)
        return [int(round(((x - grid.origin_x) / grid.res - 0.5) * 16)),
                int(round(((y - grid.origin_y) / grid.res - 0.5) * 16))]

    min_free = params["grid_cell_seen_frac"]
    free = np.zeros_like(out)
    for a in range(edges.cell_free.shape[0]):
        for b in range(edges.cell_free.shape[1]):
            if edges.cell_free[a, b] < min_free:
                continue
            i, j = edges.i0 + a, edges.j0 + b
            poly = np.array([px(i * cell, j * cell), px((i + 1) * cell, j * cell),
                             px((i + 1) * cell, (j + 1) * cell), px(i * cell, (j + 1) * cell)], np.int32)
            cv2.fillPoly(free, [poly], 1, cv2.LINE_8, 4)
    out[free > 0] = FREE
    thick = max(1, int(round(params["grid_wall_thickness_m"] / grid.res)))
    walls = np.zeros_like(out)
    for x1, y1, x2, y2, _ in edges.segments(model, EDGE_WALL):
        a = ((x1 - grid.origin_x) / grid.res - 0.5, (y1 - grid.origin_y) / grid.res - 0.5)
        b = ((x2 - grid.origin_x) / grid.res - 0.5, (y2 - grid.origin_y) / grid.res - 0.5)
        cv2.line(walls, (int(round(a[0] * 16)), int(round(a[1] * 16))),
                 (int(round(b[0] * 16)), int(round(b[1] * 16))), 1, thick, cv2.LINE_8, 4)
    out[walls > 0] = OCCUPIED
    return out


def edge_state(edges, a_cell, b_cell):
    """State of the edge between two neighbouring cells (EDGE_UNKNOWN outside the map)."""
    (i, j), (k, l) = a_cell, b_cell
    if i != k:  # vertical edge at u = max(i, k)
        a = max(i, k) - edges.i0
        b = j - edges.j0
        if 0 <= a < edges.vert.shape[0] and 0 <= b < edges.vert.shape[1]:
            return int(edges.vert[a, b])
    else:
        a = i - edges.i0
        b = max(j, l) - edges.j0
        if 0 <= a < edges.horz.shape[0] and 0 <= b < edges.horz.shape[1]:
            return int(edges.horz[a, b])
    return EDGE_UNKNOWN


def neighbours(edges, cell, blocked=(), opened=()):
    """Cells reachable from ``cell`` through open edges (lattice indices).
    ``opened`` edges were checked open with the ToF; ``blocked`` were not."""
    i, j = cell
    a, b = i - edges.i0, j - edges.j0
    ni, nj = edges.cell_free.shape
    out = []
    if not (0 <= a < ni and 0 <= b < nj):
        return out
    for (da, db), state in (((1, 0), edges.vert[a + 1, b]), ((-1, 0), edges.vert[a, b]),
                            ((0, 1), edges.horz[a, b + 1]), ((0, -1), edges.horz[a, b])):
        n = (i + da, j + db)
        key = frozenset((cell, n))
        if (state == EDGE_OPEN or key in opened) and 0 <= a + da < ni and 0 <= b + db < nj and \
                key not in blocked:
            out.append(n)
    return out


def unknown_count(edges, cell):
    a, b = cell[0] - edges.i0, cell[1] - edges.j0
    if not (0 <= a < edges.cell_free.shape[0] and 0 <= b < edges.cell_free.shape[1]):
        return 0
    return int(edges.vert[a, b] == EDGE_UNKNOWN) + int(edges.vert[a + 1, b] == EDGE_UNKNOWN) + \
        int(edges.horz[a, b] == EDGE_UNKNOWN) + int(edges.horz[a, b + 1] == EDGE_UNKNOWN)


def cell_center(model, cell):
    x, y = model.to_world((cell[0] + 0.5) * model.cell, (cell[1] + 0.5) * model.cell)
    return float(x), float(y)


def render_grid_image(model, edges, params, trajectory=(), start=None, end=None, px_per_m=160):
    """Crisp map in the grid's own frame (grid lines straight), BGR."""
    cell = model.cell
    ni, nj = edges.cell_free.shape
    cp = int(round(cell * px_per_m))
    margin = 30
    w, h = ni * cp + 2 * margin, nj * cp + 2 * margin
    img = np.full((h, w, 3), 170, np.uint8)
    seen_frac = params["grid_cell_seen_frac"]

    def px(u, v):  # grid metres -> image pixel
        return (int(round(margin + (u / cell - edges.i0) * cp)), int(round(h - margin - (v / cell - edges.j0) * cp)))

    for a in range(ni):
        for b in range(nj):
            if edges.cell_free[a, b] >= seen_frac:
                x0, y0 = px((edges.i0 + a) * cell, (edges.j0 + b + 1) * cell)
                cv2.rectangle(img, (x0, y0), (x0 + cp, y0 + cp), (255, 255, 255), -1)
    for k in range(ni + 1):
        x = px((edges.i0 + k) * cell, 0)[0]
        cv2.line(img, (x, margin), (x, h - margin), (215, 215, 215), 1)
    for k in range(nj + 1):
        y = px(0, (edges.j0 + k) * cell)[1]
        cv2.line(img, (margin, y), (w - margin, y), (215, 215, 215), 1)
    thick = max(3, int(round(params["grid_wall_thickness_m"] * px_per_m)))
    for a in range(ni + 1):
        for b in range(nj):
            st = edges.vert[a, b]
            if st == EDGE_UNKNOWN:
                continue
            u = (edges.i0 + a) * cell
            p1, p2 = px(u, (edges.j0 + b) * cell), px(u, (edges.j0 + b + 1) * cell)
            if st == EDGE_WALL:
                cv2.line(img, p1, p2, (25, 25, 25), thick)
    for a in range(ni):
        for b in range(nj + 1):
            if edges.horz[a, b] == EDGE_WALL:
                v = (edges.j0 + b) * cell
                cv2.line(img, px((edges.i0 + a) * cell, v), px((edges.i0 + a + 1) * cell, v), (25, 25, 25), thick)
    if len(trajectory) > 1:
        u, v = model.to_grid([p[0] for p in trajectory], [p[1] for p in trajectory])
        pts = np.array([px(a, b) for a, b in zip(u, v)], np.int32)
        cv2.polylines(img, [pts], False, (230, 90, 30), 2, cv2.LINE_AA)
    for pose, color, label in ((start, (40, 170, 40), "START"), (end, (40, 40, 220), "END")):
        if pose is None:
            continue
        u, v = model.to_grid(pose[0], pose[1])
        c = px(float(u), float(v))
        th = pose[2] - model.theta
        tip = (int(c[0] + 26 * math.cos(th)), int(c[1] - 26 * math.sin(th)))
        cv2.circle(img, c, 8, color, -1, cv2.LINE_AA)
        cv2.arrowedLine(img, c, tip, color, 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(img, label, (c[0] + 10, c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.putText(img, f"grid {cell:.3f} m cells", (margin, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1,
                cv2.LINE_AA)
    return img


def unknown_edge_targets(model, edges):
    """Centres (world) of seen cells that still have an unknown edge,
    with the number of unknown edges, most first."""
    if edges is None:
        return []
    out = []
    cell = model.cell
    for a in range(edges.cell_free.shape[0]):
        for b in range(edges.cell_free.shape[1]):
            if edges.cell_free[a, b] < 0.3:
                continue
            n = int(edges.vert[a, b] == EDGE_UNKNOWN) + int(edges.vert[a + 1, b] == EDGE_UNKNOWN) + \
                int(edges.horz[a, b] == EDGE_UNKNOWN) + int(edges.horz[a, b + 1] == EDGE_UNKNOWN)
            if n:
                x, y = model.to_world((edges.i0 + a + 0.5) * cell, (edges.j0 + b + 0.5) * cell)
                out.append((float(x), float(y), n, (edges.i0 + a, edges.j0 + b)))
    out.sort(key=lambda t: -t[2])
    return out


def cell_of(model, x, y):
    u, v = model.to_grid(x, y)
    return int(math.floor(float(u) / model.cell)), int(math.floor(float(v) / model.cell))
