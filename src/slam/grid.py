"""Log-odds occupancy grid.

World frame: x = forward at the start, y = left, heading = counter-clockwise
radians, origin = where the robot started. Cell (row, col) covers
x = origin_x + col*res and y = origin_y + row*res.
"""

import math

import cv2
import numpy as np

UNKNOWN, FREE, OCCUPIED = 0, 1, 2


def prob_to_logodds(p):
    return math.log(p / (1.0 - p))


class OccupancyGrid:
    def __init__(self, resolution, width_m, height_m, origin_x, origin_y):
        self.res = float(resolution)
        self.cols = max(4, int(round(width_m / self.res)))
        self.rows = max(4, int(round(height_m / self.res)))
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.logodds = np.zeros((self.rows, self.cols), np.float32)
        self.hits = np.zeros((self.rows, self.cols), np.uint16)  # times a beam ended in the cell
        self.version = 0

    # ---- geometry ----------------------------------------------------------
    def world_to_cell(self, x, y):
        col = np.floor((np.asarray(x) - self.origin_x) / self.res).astype(np.int64)
        row = np.floor((np.asarray(y) - self.origin_y) / self.res).astype(np.int64)
        return row, col

    def cell_to_world(self, row, col):
        return (self.origin_x + (np.asarray(col) + 0.5) * self.res,
                self.origin_y + (np.asarray(row) + 0.5) * self.res)

    def inside(self, row, col):
        return (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)

    def extent(self):
        return (self.origin_x, self.origin_y,
                self.origin_x + self.cols * self.res, self.origin_y + self.rows * self.res)

    # ---- updates -------------------------------------------------------------
    def _px(self, x, y):
        """Sub-pixel image coordinates (col, row) for cv2 drawing, 4 fractional bits."""
        return (np.round(((np.asarray(x) - self.origin_x) / self.res - 0.5) * 16).astype(np.int32),
                np.round(((np.asarray(y) - self.origin_y) / self.res - 0.5) * 16).astype(np.int32))

    def integrate(self, origin, angles, ranges, hits, l_occ, l_free, clamp, weight=1.0, max_gap=None,
                  wall_join_m=0.12):
        """Add one scan taken from ``origin`` (x, y) in world coordinates.

        angles are world beam directions (rad) in scan order; ranges in metres;
        hits says whether the beam ended on a wall (True) or only proves free
        space. Neighbouring beams (at most ``max_gap`` rad apart) also clear the
        wedge between them and, when both end on the same wall, draw the wall
        between their end points, so gaps between beams do not stay unknown.
        Each cell changes at most once per scan, so dense beams near the robot
        do not outvote the sparse beams far away.
        """
        angles = np.asarray(angles, float)
        ranges = np.asarray(ranges, float)
        hits = np.asarray(hits, bool)
        if angles.size == 0:
            return
        ox, oy = origin
        angles, ranges, hits = self._drop_through_walls(ox, oy, angles, ranges, hits, l_occ)
        if angles.size == 0:
            return
        free_len = np.where(hits, ranges - self.res, ranges).clip(min=0)
        fx, fy = ox + free_len * np.cos(angles), oy + free_len * np.sin(angles)
        hx, hy = ox + ranges * np.cos(angles), oy + ranges * np.sin(angles)
        opx, opy = self._px(ox, oy)
        fpx, fpy = self._px(fx, fy)
        hpx, hpy = self._px(hx, hy)

        free = np.zeros((self.rows, self.cols), np.uint8)
        occ = np.zeros((self.rows, self.cols), np.uint8)
        for i in range(len(angles)):
            cv2.line(free, (int(opx), int(opy)), (int(fpx[i]), int(fpy[i])), 1, 1, cv2.LINE_8, 4)
        n = len(angles)
        if max_gap and n > 1:
            for i in range(n):
                j = (i + 1) % n
                if j == 0 and n < 3:
                    break
                gap = abs((angles[j] - angles[i] + math.pi) % (2 * math.pi) - math.pi)
                if gap > max_gap:
                    continue
                tri = np.array([[opx, opy], [fpx[i], fpy[i]], [fpx[j], fpy[j]]], np.int32)
                cv2.fillPoly(free, [tri], 1, cv2.LINE_8, 4)
                if hits[i] and hits[j] and abs(ranges[i] - ranges[j]) < wall_join_m + ranges[i] * gap:
                    cv2.line(occ, (int(hpx[i]), int(hpy[i])), (int(hpx[j]), int(hpy[j])), 1, 1, cv2.LINE_8, 4)
        if hits.any():
            r, c = self.world_to_cell(hx[hits], hy[hits])
            ok = self.inside(r, c)
            occ[r[ok], c[ok]] = 1
            np.add.at(self.hits, (r[ok], c[ok]), 1)
        occ = occ.astype(bool)
        free = free.astype(bool) & ~occ
        self.logodds[free] -= l_free * weight
        self.logodds[occ] += l_occ * weight
        np.clip(self.logodds, -clamp, clamp, out=self.logodds)
        self.version += 1

    def _drop_through_walls(self, ox, oy, angles, ranges, hits, l_occ, confident=2.0, slack_cells=3):
        """Remove beams that pass through a wall the map is already sure of.

        A reading that goes more than ``slack_cells`` past a confident wall is
        almost always an outlier or a dropout, and would otherwise punch a
        free-space hole through the wall.
        """
        if angles.size == 0:
            return angles, ranges, hits
        thresh = confident * l_occ
        step = self.res * 0.5
        n_steps = int(ranges.max() / step) + 1
        t = np.arange(n_steps) * step
        xs = ox + np.cos(angles)[:, None] * t[None, :]
        ys = oy + np.sin(angles)[:, None] * t[None, :]
        r, c = self.world_to_cell(xs, ys)
        ok = self.inside(r, c)
        wall = np.zeros(r.shape, bool)
        wall[ok] = self.logodds[r[ok], c[ok]] >= thresh
        before_end = t[None, :] < (ranges[:, None] - slack_cells * self.res)
        through = (wall & before_end).any(axis=1)
        keep = ~through
        return angles[keep], ranges[keep], hits[keep]

    def clear(self):
        self.logodds[:] = 0
        self.hits[:] = 0
        self.version += 1

    # ---- reading -------------------------------------------------------------
    def probability(self):
        return 1.0 - 1.0 / (1.0 + np.exp(self.logodds))

    def classify(self, occ_prob, free_prob, clean_isolated=False):
        """uint8 grid of UNKNOWN / FREE / OCCUPIED."""
        occ_l = prob_to_logodds(occ_prob)
        free_l = prob_to_logodds(free_prob)
        out = np.full((self.rows, self.cols), UNKNOWN, np.uint8)
        out[self.logodds <= free_l] = FREE
        occ = self.logodds >= occ_l
        if clean_isolated:
            occ = remove_isolated(occ)
        out[occ] = OCCUPIED
        return out

    def occupied_mask(self, occ_prob):
        return self.logodds >= prob_to_logodds(occ_prob)


def remove_isolated(mask):
    """Drop True cells that have no True 8-neighbour (single-cell noise)."""
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.float32)
    kernel[1, 1] = 0
    neighbours = cv2.filter2D(m, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return mask & (neighbours > 0)


def border_mask(grid, params):
    """True for cells inside the exploration border (all True when it is off)."""
    if not params.get("border_enabled"):
        return np.ones((grid.rows, grid.cols), bool)
    xs, ys = grid.cell_to_world(np.arange(grid.rows)[:, None], np.arange(grid.cols)[None, :])
    return ((xs >= params["border_min_x"]) & (xs <= params["border_max_x"]) &
            (ys >= params["border_min_y"]) & (ys <= params["border_max_y"]))
