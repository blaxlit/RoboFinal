"""Closed-loop motion and scanning in the SLAM world frame.

The driver turns raw robot values into world-frame values (x forward at the
start, y left, counter-clockwise heading) using the *_sign settings, and runs
the feedback loops for turning, driving, gimbal positioning and ToF scans.
"""

import math
import threading

import numpy as np


class Abort(Exception):
    """Raised inside a motion when the operator presses Stop."""


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Driver:
    def __init__(self, io, params, checkpoint=lambda: None):
        self.io = io
        self.p = params                 # shared dict, read live so console changes apply at once
        self.checkpoint = checkpoint    # raises Abort / blocks while paused
        self.sweep_forward = True
        self.last_blocked_range = None
        self._odom_lock = threading.Lock()
        self._odom_last = None   # (t, x, y, th_raw, signs) of the previous raw reading
        self._odom = (0.0, 0.0, 0.0)

    # ---- conversions --------------------------------------------------------------
    def odom_pose(self):
        """Odometry pose with the gyro drift removed.

        The robot integrates x/y with its own (drifting) heading, so removing
        drift from the heading alone would rotate every later move. Instead
        each small raw step is turned back into forward/sideways motion and
        re-integrated with the corrected heading.
        """
        with self._odom_lock:
            x, y_raw, yaw = self.io.odom_raw()
            t = self.io.now()
            signs = (self.p["chassis_yaw_sign"], self.p["odom_y_sign"])
            y = signs[1] * y_raw
            th_raw = math.radians(signs[0] * yaw)
            last = self._odom_last
            if last is None or last[4] != signs:
                # first reading, or calibration changed a sign: restart here
                self._odom = (x, y, th_raw) if last is None else (self._odom[0], self._odom[1], th_raw)
            else:
                lt, lx, ly, lth, _ = last
                d_raw = wrap(th_raw - lth)
                mid_raw = lth + d_raw / 2
                dx, dy = x - lx, y - ly
                fwd = math.cos(mid_raw) * dx + math.sin(mid_raw) * dy
                side = -math.sin(mid_raw) * dx + math.cos(mid_raw) * dy
                ox, oy, oth = self._odom
                nth = oth + d_raw - math.radians(self.p["yaw_drift_dps"]) * (t - lt)
                mid = (oth + nth) / 2
                self._odom = (ox + math.cos(mid) * fwd - math.sin(mid) * side,
                              oy + math.sin(mid) * fwd + math.cos(mid) * side, wrap(nth))
            self._odom_last = (t, x, y, th_raw, signs)
            return self._odom

    def gimbal_deg(self):
        """Gimbal yaw relative to the chassis, counter-clockwise degrees."""
        return self.p["gimbal_yaw_sign"] * self.io.gimbal_raw()[1]

    def gimbal_deg_at(self, t):
        return self.p["gimbal_yaw_sign"] * self.io.gimbal_yaw_at(t)

    def tof_m(self):
        """Latest forward reading (raw metres, lens to wall), NaN when none."""
        t, mm = self.io.tof_latest()
        return mm / 1000.0

    def _drive(self, vx, wz_ccw_dps):
        self.io.drive(vx, 0.0, self.p["cmd_z_sign"] * wz_ccw_dps)

    def stop(self):
        self.io.stop()

    # ---- gimbal -----------------------------------------------------------------------
    def gimbal_goto(self, yaw_deg, pitch_deg=None, tol=1.0, timeout=6.0, speed=None):
        speed = speed or self.p["gimbal_move_speed_dps"]
        pitch_deg = self.p["scan_pitch_deg"] if pitch_deg is None else pitch_deg
        end = self.io.now() + timeout
        settled = 0
        while self.io.now() < end:
            self.checkpoint()
            err = yaw_deg - self.gimbal_deg()
            perr = pitch_deg - self.io.gimbal_raw()[0]
            if abs(err) < tol and abs(perr) < 2.0:
                settled += 1
                if settled >= 3:
                    break
            else:
                settled = 0
            rate = max(-speed, min(speed, 4.0 * err))
            if abs(err) >= tol and abs(rate) < 8.0:
                rate = math.copysign(8.0, err)
            prate = max(-60.0, min(60.0, 4.0 * perr))
            self.io.gimbal_speed(prate, self.p["gimbal_yaw_sign"] * rate)
            self.io.sleep(0.02)
        self.io.gimbal_speed(0, 0)
        return abs(yaw_deg - self.gimbal_deg()) < max(tol * 3, 3.0)

    # ---- chassis --------------------------------------------------------------------------
    def _motion_mode(self):
        """Point the ToF forward, then let the gimbal follow the chassis.

        The gimbal is re-centred in FREE mode first: in CHASSIS_LEAD mode the
        gimbal follows the chassis and may ignore yaw commands.
        """
        if abs(self.gimbal_deg()) > 3.0:
            self.io.set_mode("free")
            self.gimbal_goto(0.0, 0.0)
        self.io.set_mode("chassis_lead")

    def turn_by(self, dtheta, on_tick=None, timeout=20.0):
        """Rotate in place by dtheta radians (counter-clockwise positive)."""
        self._motion_mode()
        start = self.odom_pose()[2]
        target = start + dtheta
        tol = math.radians(self.p["turn_tolerance_deg"])
        end = self.io.now() + timeout + abs(math.degrees(dtheta)) / max(self.p["angular_speed_dps"], 1.0)
        try:
            while self.io.now() < end:
                self.checkpoint()
                err = wrap(target - self.odom_pose()[2])
                if abs(err) < tol:
                    break
                rate = max(-self.p["angular_speed_dps"], min(self.p["angular_speed_dps"], math.degrees(err) * 2.5))
                if abs(rate) < 12.0:
                    rate = math.copysign(12.0, rate)
                self._drive(0.0, rate)
                if on_tick:
                    on_tick()
                self.io.sleep(0.03)
        finally:
            self._drive(0.0, 0.0)
        self.io.sleep(0.15)
        return abs(wrap(target - self.odom_pose()[2])) < tol * 2

    def forward(self, distance, on_tick=None, on_tof=None, timeout=None):
        """Drive straight; returns (travelled_m, blocked)."""
        self._motion_mode()
        x0, y0, th0 = self.odom_pose()
        sign = 1.0 if distance >= 0 else -1.0
        dist = abs(distance)
        speed = self.p["linear_speed_mps"]
        timeout = timeout or (dist / max(speed, 0.02) * 2.5 + 3.0)
        end = self.io.now() + timeout
        blocked = False
        last_tof_t = self.io.now()
        travelled = 0.0
        try:
            while self.io.now() < end:
                self.checkpoint()
                x, y, th = self.odom_pose()
                travelled = (x - x0) * math.cos(th0) + (y - y0) * math.sin(th0)
                travelled *= sign
                remaining = dist - travelled
                if remaining <= 0.01:
                    break
                samples = self.io.tof_since(last_tof_t)
                if samples:
                    last_tof_t = samples[-1][0]
                    if on_tof:
                        for t, mm in samples:
                            on_tof(t, mm / 1000.0)
                front = self.tof_m() * self.p["tof_scale"] + self.p["tof_bias_m"]
                if sign > 0 and math.isfinite(front) and front < self.p["stop_distance_m"]:
                    # close to the goal this is just arriving next to a wall
                    blocked = remaining > 0.08
                    self.last_blocked_range = front
                    break
                v = min(speed, max(0.06, remaining * 1.5))
                herr = wrap(th0 - th)
                self._drive(sign * v, max(-30.0, min(30.0, math.degrees(herr) * 2.0)))
                if on_tick:
                    on_tick()
                self.io.sleep(0.03)
        finally:
            self._drive(0.0, 0.0)
        self.io.sleep(0.2)
        x, y, _ = self.odom_pose()
        travelled = sign * ((x - x0) * math.cos(th0) + (y - y0) * math.sin(th0))
        return abs(travelled), blocked

    def strafe(self, distance_left, on_tick=None, timeout=None):
        """Slide sideways (mecanum) without turning; + = left. Returns metres moved."""
        self._motion_mode()
        x0, y0, th0 = self.odom_pose()
        sign = 1.0 if distance_left >= 0 else -1.0
        dist = abs(distance_left)
        speed = min(self.p["linear_speed_mps"], 0.2)
        end = self.io.now() + (timeout or dist / max(speed, 0.02) * 3 + 2.0)
        lx, ly = -math.sin(th0), math.cos(th0)
        try:
            while self.io.now() < end:
                self.checkpoint()
                x, y, th = self.odom_pose()
                moved = sign * ((x - x0) * lx + (y - y0) * ly)
                remaining = dist - moved
                if remaining <= 0.008:
                    break
                v = min(speed, max(0.05, remaining * 1.5))
                herr = wrap(th0 - th)
                self.io.drive(0.0, self.p["odom_y_sign"] * sign * v,
                              self.p["cmd_z_sign"] * max(-30.0, min(30.0, math.degrees(herr) * 2.0)))
                if on_tick:
                    on_tick()
                self.io.sleep(0.03)
        finally:
            self._drive(0.0, 0.0)
        self.io.sleep(0.15)
        x, y, _ = self.odom_pose()
        return abs((x - x0) * lx + (y - y0) * ly)

    def front_range(self, samples=5, timeout=1.0):
        """Median of fresh forward ToF readings (lens to wall, metres), None if none.
        Points the gimbal forward first."""
        if abs(self.gimbal_deg()) > 3.0:
            self.io.set_mode("free")
            self.gimbal_goto(0.0, 0.0)
        t0 = self.io.now()
        end = t0 + timeout
        got = []
        while self.io.now() < end and len(got) < samples:
            self.checkpoint()
            got = [mm for _, mm in self.io.tof_since(t0 + self.p["tof_latency_s"])]
            self.io.sleep(0.02)
        vals = [mm / 1000.0 for mm in got if 0 < mm < 9990]
        if not got:
            return None
        if not vals:  # nothing in range: wide open
            return 10.0
        return float(np.median(vals)) * self.p["tof_scale"] + self.p["tof_bias_m"]

    def manual(self, vx, vy_left, wz_ccw_dps):
        self.io.set_mode("chassis_lead")  # no re-centre here: it would block the dead-man loop
        self.io.drive(vx, self.p["odom_y_sign"] * vy_left, self.p["cmd_z_sign"] * wz_ccw_dps)

    # ---- scanning -----------------------------------------------------------------------------
    def scan(self, direction=None, step_deg=None, mode=None):
        """360-degree (configurable) ToF scan with the gimbal.

        Returns a list of (robot_angle_rad, raw_range_m); raw_range is inf when
        the sensor saw nothing.
        """
        self.io.set_mode("free")
        p = self.p
        mode = mode or p["scan_mode"]
        a0, a1 = p["scan_start_deg"], p["scan_end_deg"]
        if direction is None:
            direction = self.sweep_forward
            if p["alternate_sweep"]:
                self.sweep_forward = not self.sweep_forward
        if not direction:
            a0, a1 = a1, a0
        if mode == "step":
            return self._scan_step(a0, a1, step_deg or p["scan_step_deg"])
        return self._scan_sweep(a0, a1)

    def _to_range(self, mm):
        if mm >= 9990 or mm <= 0 or not math.isfinite(mm):
            return math.inf
        return mm / 1000.0

    def _scan_sweep(self, a0, a1):
        p = self.p
        self.gimbal_goto(a0)
        rate = p["scan_speed_dps"] * (1 if a1 > a0 else -1)
        t_start = self.io.now()
        end = t_start + abs(a1 - a0) / p["scan_speed_dps"] * 2.0 + 3.0
        self.io.gimbal_speed(0.0, p["gimbal_yaw_sign"] * rate)
        try:
            while self.io.now() < end:
                self.checkpoint()
                g = self.gimbal_deg()
                if (rate > 0 and g >= a1) or (rate < 0 and g <= a1):
                    break
                self.io.sleep(0.02)
        finally:
            self.io.gimbal_speed(0, 0)
        t_end = self.io.now()
        self.io.sleep(p["tof_latency_s"] + 0.05)
        out = []
        lo, hi = min(a0, a1), max(a0, a1)
        for t, mm in self.io.tof_since(t_start):
            ts = t - p["tof_latency_s"]
            if ts < t_start or ts > t_end:
                continue
            ang = self.gimbal_deg_at(ts)
            if lo - 1 <= ang <= hi + 1:
                out.append((math.radians(ang), self._to_range(mm)))
        return out

    def _scan_step(self, a0, a1, step):
        p = self.p
        n = int(abs(a1 - a0) / step) + 1
        sgn = 1 if a1 > a0 else -1
        out = []
        for i in range(n):
            target = a0 + sgn * i * step
            self.gimbal_goto(target, tol=0.6, timeout=3.0)
            self.io.sleep(p["settle_s"])
            t0 = self.io.now()
            wait_end = t0 + 0.08 * p["samples_per_step"] + 0.5
            got = []
            while len(got) < p["samples_per_step"] and self.io.now() < wait_end:
                self.checkpoint()
                got = self.io.tof_since(t0 + p["tof_latency_s"] * 0.5)
                self.io.sleep(0.02)
            ang = math.radians(self.gimbal_deg())
            for _, mm in got[:p["samples_per_step"]]:
                out.append((ang, self._to_range(mm)))
        return out
