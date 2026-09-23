"""Simulated RoboMaster EP with a gimbal-mounted ToF, for testing without the robot.

It reports values in the same raw conventions as the real SDK (odometry y to
the right, clockwise yaw positive, positive gimbal yaw turns right) and adds
realistic faults: ToF noise, outliers, dropouts and latency, odometry scale
error and gyro drift. Walls come from a ground-truth JSON file.
"""

import collections
import math
import random
import threading
import time

import numpy as np

from .io_real import interpolate

TOF_NO_RETURN_MM = 10000.0


class SimRobotIO:
    name = "simulator"

    def __init__(self, ground_truth, start=(0.3, 0.3, 0.0), time_scale=4.0, seed=1,
                 tof_sigma=0.01, tof_outlier_rate=0.02, tof_dropout_rate=0.01, tof_latency=0.04,
                 odom_scale=1.02, yaw_drift_dps=0.05, lens_offset=0.08, flip=None):
        self.rng = random.Random(seed)
        segs = ground_truth["walls"] + _border(ground_truth["border"])
        self.segs = np.asarray(segs, float).reshape(-1, 4)
        self.time_scale = float(time_scale)
        self.tof_sigma, self.outlier_rate, self.dropout_rate = tof_sigma, tof_outlier_rate, tof_dropout_rate
        self.tof_latency = tof_latency
        self.odom_scale, self.yaw_drift = odom_scale, math.radians(yaw_drift_dps)
        self.lens_offset = lens_offset
        # sign conventions of the simulated hardware (flip to test auto calibration)
        flip = flip or {}
        self.yaw_sign = -1 * (-1 if flip.get("yaw") else 1)       # raw yaw = sign * ccw
        self.y_sign = -1 * (-1 if flip.get("y") else 1)           # raw y = sign * left
        self.gimbal_sign = -1                                      # raw gimbal = sign * ccw
        self.cmd_z_sign = 1 * (-1 if flip.get("cmd_z") else 1)     # ccw rate = sign * cmd

        self.true = list(start)            # x, y, th in ground-truth frame
        self.start = tuple(start)
        self.odom_state = [0.0, 0.0, 0.0]  # x fwd, y left, th ccw in odom frame (with errors)
        self.gimbal_ccw = 0.0              # deg, relative to chassis
        self.gimbal_pitch = 0.0
        self.cmd = (0.0, 0.0, 0.0, 0.0)    # vx, vy_right, wz_raw, deadline
        self.gimbal_cmd = (0.0, 0.0)
        self.vel = [0.0, 0.0, 0.0]
        self.t = 0.0
        self._lock = threading.Lock()
        self.gimbal_hist = collections.deque(maxlen=800)
        self.tof = collections.deque(maxlen=4000)
        self._pending_tof = collections.deque()
        self._next_tof = 0.0
        self.collisions = 0
        self.mode = "free"
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---- simulation loop --------------------------------------------------------
    def _loop(self):
        dt = 0.01
        last = time.monotonic()
        while self._running:
            with self._lock:
                self._step(dt)
            last += dt / self.time_scale
            delay = last - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                last = time.monotonic()

    def _step(self, dt):
        self.t += dt
        vx, vy_r, wz_raw, deadline = self.cmd
        if self.t > deadline:
            vx = vy_r = wz_raw = 0.0
        target = (vx, -vy_r, math.radians(wz_raw) * self.cmd_z_sign)
        for i in range(3):  # first-order motor lag
            self.vel[i] += (target[i] - self.vel[i]) * min(1.0, dt / 0.08)
        vxt, vyt, wt = self.vel
        x, y, th = self.true
        nth = th + wt * dt
        nx = x + (vxt * math.cos(th) - vyt * math.sin(th)) * dt
        ny = y + (vxt * math.sin(th) + vyt * math.cos(th)) * dt
        if self._clearance(nx, ny) < 0.12:
            nx, ny = x, y
            if abs(vxt) + abs(vyt) > 1e-3:
                self.collisions += 1
        dxw, dyw = nx - x, ny - y
        self.true = [nx, ny, nth]
        # odometry: body-frame motion integrated with scale error and drift
        bx = math.cos(th) * dxw + math.sin(th) * dyw
        by = -math.sin(th) * dxw + math.cos(th) * dyw
        ox, oy, oth = self.odom_state
        oth += wt * dt + self.yaw_drift * dt
        bx *= self.odom_scale
        by *= self.odom_scale
        self.odom_state = [ox + bx * math.cos(oth) - by * math.sin(oth),
                           oy + bx * math.sin(oth) + by * math.cos(oth), oth]
        # gimbal
        pitch_rate, yaw_rate_raw = self.gimbal_cmd
        self.gimbal_ccw = max(-250.0, min(250.0, self.gimbal_ccw + yaw_rate_raw * self.gimbal_sign * dt))
        self.gimbal_pitch = max(-20.0, min(35.0, self.gimbal_pitch + pitch_rate * dt))
        self.gimbal_hist.append((self.t, self.gimbal_sign * self.gimbal_ccw))
        # ToF at 20 Hz, delivered after the latency
        if self.t >= self._next_tof:
            self._next_tof = self.t + 0.05
            self._pending_tof.append((self.t + self.tof_latency, self._measure()))
        while self._pending_tof and self._pending_tof[0][0] <= self.t:
            self.tof.append(self._pending_tof.popleft())

    def _measure(self):
        x, y, th = self.true
        ang = th + math.radians(self.gimbal_ccw)
        d = self._raycast(x, y, ang)
        r = self.rng.random()
        if r < self.dropout_rate or d > 9.9:
            return TOF_NO_RETURN_MM
        if r < self.dropout_rate + self.outlier_rate:
            return self.rng.uniform(100, 3000)
        d = d - self.lens_offset
        d += self.rng.gauss(0, self.tof_sigma + 0.01 * d)
        return max(20.0, d * 1000.0)

    def _raycast(self, x, y, ang):
        dx, dy = math.cos(ang), math.sin(ang)
        s = self.segs
        ex, ey = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        den = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((s[:, 0] - x) * ey - (s[:, 1] - y) * ex) / den
            u = ((s[:, 0] - x) * dy - (s[:, 1] - y) * dx) / den
        ok = (np.abs(den) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        return float(t[ok].min()) if ok.any() else 99.0

    def _clearance(self, x, y):
        s = self.segs
        ex, ey = s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]
        l2 = np.maximum(ex * ex + ey * ey, 1e-12)
        u = np.clip(((x - s[:, 0]) * ex + (y - s[:, 1]) * ey) / l2, 0, 1)
        return float(np.min(np.hypot(s[:, 0] + u * ex - x, s[:, 1] + u * ey - y)))

    # ---- interface shared with RealRobotIO ---------------------------------------
    def now(self):
        with self._lock:
            return self.t

    def sleep(self, seconds):
        end = self.now() + max(0.0, seconds)
        while self._running and self.now() < end:
            time.sleep(0.002)

    def odom_raw(self):
        with self._lock:
            x, y, th = self.odom_state
        return x, self.y_sign * y, self.yaw_sign * math.degrees(th)

    def gimbal_raw(self):
        with self._lock:
            return self.gimbal_pitch, self.gimbal_sign * self.gimbal_ccw

    def gimbal_yaw_at(self, t):
        with self._lock:
            hist = list(self.gimbal_hist)
        return interpolate(hist, t)

    def tof_since(self, t):
        with self._lock:
            return [s for s in self.tof if s[0] > t]

    def tof_latest(self):
        with self._lock:
            return self.tof[-1] if self.tof else (0.0, float("nan"))

    def battery(self):
        return 100

    def drive(self, vx, vy, wz):
        with self._lock:
            self.cmd = (float(vx), float(vy), float(wz), self.t + 0.5)

    def stop(self):
        with self._lock:
            self.cmd = (0.0, 0.0, 0.0, 0.0)
            self.gimbal_cmd = (0.0, 0.0)

    def gimbal_speed(self, pitch_dps, yaw_dps):
        with self._lock:
            self.gimbal_cmd = (float(pitch_dps), float(yaw_dps))

    def set_mode(self, mode):
        self.mode = mode

    def true_pose(self):
        with self._lock:
            return tuple(self.true)

    def close(self):
        self._running = False


def _border(b):
    x0, y0, x1, y1 = b
    return [[x0, y0, x1, y0], [x1, y0, x1, y1], [x1, y1, x0, y1], [x0, y1, x0, y0]]
