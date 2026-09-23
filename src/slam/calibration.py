"""Auto calibration.

1. ToF noise   - robot still, 40+ readings: noise level sets the outlier filter.
2. Gyro drift  - heading change while still becomes yaw_drift_dps.
3. Turn signs  - scan, turn the chassis, scan again. The angle the scan
                 profile shifted by is the real turn, so it tells whether the
                 reported yaw and the turn command have the right sign.
4. Latency     - one sweep each way; the angle shift between them is
                 2 x speed x latency error.
5. Odometry y  - turn 90 deg, drive 0.2 m and back: the reported movement must
                 point the way the robot faces.
Steps 3-5 move the robot, so they only run with calibrate_with_motion on.

The whole map could still be a mirror image if every sign were flipped; that
one choice is gimbal_yaw_sign (default: positive gimbal yaw turns right, as
in analysis/analyze_lidar.py). If a saved map comes out mirrored, flip it.
"""

import math

import numpy as np

from .driver import wrap
from .scan import bin_samples


def profile(samples, params, step_deg=1.0):
    """Range per whole degree (NaN where no wall), robot frame."""
    p = dict(params, scan_step_deg=step_deg, no_hit_free_m=0.0, tof_offset_m=0.0, isolated_join_m=0.0)
    angles, ranges, hits = bin_samples(samples, p)
    grid = np.arange(-180, 180, step_deg)
    out = np.full(grid.shape, np.nan)
    for a, r, h in zip(np.degrees(angles), ranges, hits):
        if h:
            i = int(round((a + 180) / step_deg)) % len(grid)
            out[i] = r
    return grid, out


def best_shift(prof_a, prof_b, max_shift=90.0, step=0.5):
    """Shift s (deg) minimising |B(phi) - A(phi + s)|; returns (s, residual)."""
    grid, a = prof_a
    _, b = prof_b
    ok_a = np.isfinite(a)
    if ok_a.sum() < 20 or np.isfinite(b).sum() < 20:
        return None, None
    ext = np.concatenate([grid[ok_a] - 360, grid[ok_a], grid[ok_a] + 360])
    val = np.concatenate([a[ok_a]] * 3)
    best = (None, math.inf)
    ok_b = np.isfinite(b)
    for s in np.arange(-max_shift, max_shift + 1e-9, step):
        shifted = np.interp(grid[ok_b] + s, ext, val)
        res = float(np.median(np.abs(shifted - b[ok_b])))
        if res < best[1]:
            best = (float(s), res)
    return best


class Calibrator:
    def __init__(self, driver, params, log):
        self.d = driver
        self.p = params
        self.log = log

    def run(self, with_motion=True):
        results = {}
        results.update(self.noise_and_drift())
        if with_motion:
            results.update(self.turn_signs())
            if self.p["scan_mode"] == "sweep":
                results.update(self.latency())
            results.update(self.odom_y())
        self.d.gimbal_goto(0.0, 0.0)
        return results

    # ---- 1 + 2 -----------------------------------------------------------------------
    def noise_and_drift(self, seconds=2.5):
        d, io = self.d, self.d.io
        d.stop()
        d.gimbal_goto(0.0, 0.0)
        io.sleep(0.3)
        t0 = io.now()
        th0 = d.odom_pose()[2]
        io.sleep(seconds)
        dt = io.now() - t0
        dth = wrap(d.odom_pose()[2] - th0)
        readings = [mm / 1000.0 for _, mm in io.tof_since(t0) if 0 < mm < 9990]
        n_all = len(io.tof_since(t0))
        out = {}
        if len(readings) >= 5:
            arr = np.asarray(readings)
            med = float(np.median(arr))
            mad = float(np.median(np.abs(arr - med))) * 1.4826
            outliers = float(np.mean(np.abs(arr - med) > max(4 * mad, 0.05)))
            self.p["outlier_abs_m"] = round(min(0.15, max(0.01, 3.0 * mad)), 3)
            out.update(tof_median_m=round(med, 4), tof_noise_m=round(mad, 4),
                       tof_outlier_rate=round(outliers, 3),
                       tof_dropout_rate=round(1 - len(readings) / max(n_all, 1), 3),
                       outlier_abs_m=self.p["outlier_abs_m"])
            self.log(f"ToF noise {mad * 1000:.1f} mm at {med:.2f} m, outliers {outliers * 100:.0f}% "
                     f"-> filter keeps +/-{self.p['outlier_abs_m'] * 100:.1f} cm")
        else:
            self.log("ToF gave too few readings for noise calibration (nothing in range ahead?)")
        drift = math.degrees(dth) / max(dt, 1e-3)
        if abs(drift) < 2.0:
            self.p["yaw_drift_dps"] = round(self.p["yaw_drift_dps"] + drift, 4)
            out["yaw_drift_dps"] = self.p["yaw_drift_dps"]
            self.log(f"Gyro drift {drift:+.3f} deg/s -> yaw_drift_dps = {self.p['yaw_drift_dps']:+.3f}")
        return out

    # ---- 3 ---------------------------------------------------------------------------------
    def turn_signs(self, turn_cmd_dps=30.0, turn_time=1.2):
        d, io = self.d, self.d.io
        a = profile(d.scan(direction=True, mode="sweep"), self.p)
        d.io.set_mode("chassis_lead")
        _, _, yaw0 = io.odom_raw()
        end = io.now() + turn_time
        while io.now() < end:
            d.checkpoint()
            io.drive(0, 0, turn_cmd_dps)
            io.sleep(0.03)
        io.drive(0, 0, 0)
        io.sleep(0.5)
        _, _, yaw1 = io.odom_raw()
        raw_delta = (yaw1 - yaw0 + 180) % 360 - 180
        b = profile(d.scan(direction=True, mode="sweep"), self.p)
        shift, residual = best_shift(a, b)
        out = {"turn_raw_yaw_deg": round(raw_delta, 2)}
        if shift is None or abs(shift) < 5:
            self.log("Turn-sign check skipped: the scans did not show enough walls or the robot did not turn")
            self._turn_back(raw_delta)
            return out
        att = self.p["chassis_yaw_sign"] * raw_delta
        if att * shift < 0:
            self.p["chassis_yaw_sign"] *= -1
            self.log("Reported yaw had the wrong sign -> flipped chassis_yaw_sign")
        self.p["cmd_z_sign"] = 1 if shift > 0 else -1
        out.update(turn_scan_deg=round(shift, 1), turn_scan_residual_m=round(residual, 3),
                   chassis_yaw_sign=self.p["chassis_yaw_sign"], cmd_z_sign=self.p["cmd_z_sign"],
                   turn_scale=round(abs(raw_delta) / abs(shift), 3))
        self.log(f"Turn check: scan saw {shift:+.1f} deg, IMU {raw_delta:+.1f} deg (raw) -> "
                 f"chassis_yaw_sign={self.p['chassis_yaw_sign']}, cmd_z_sign={self.p['cmd_z_sign']}")
        self._turn_back(raw_delta)
        return out

    def _turn_back(self, raw_delta):
        dth = math.radians(self.p["chassis_yaw_sign"] * raw_delta)
        if abs(dth) > math.radians(2):
            self.d.turn_by(-dth)

    # ---- 4 -----------------------------------------------------------------------------------
    def latency(self):
        d = self.d
        fwd = profile(d.scan(direction=True, mode="sweep"), self.p)
        back = profile(d.scan(direction=False, mode="sweep"), self.p)
        shift, residual = best_shift(fwd, back, max_shift=20.0, step=0.25)
        if shift is None:
            self.log("Latency check skipped: not enough walls in view")
            return {}
        err = shift / (2.0 * self.p["scan_speed_dps"])
        new = min(0.5, max(0.0, self.p["tof_latency_s"] + err))
        self.log(f"Sweep directions differ by {shift:+.2f} deg -> tof_latency_s "
                 f"{self.p['tof_latency_s']:.3f} -> {new:.3f}")
        self.p["tof_latency_s"] = round(new, 4)
        return {"tof_latency_s": self.p["tof_latency_s"], "sweep_shift_deg": round(shift, 2)}

    # ---- 5 -------------------------------------------------------------------------------------
    def odom_y(self, distance=0.2):
        d, io = self.d, self.d.io
        for turn in (math.pi / 2, -math.pi / 2):
            d.turn_by(turn)
            d.gimbal_goto(0.0, 0.0)
            io.sleep(0.3)
            front = d.tof_m()
            if not (math.isfinite(front) and front > distance + self.p["stop_distance_m"] + 0.15):
                d.turn_by(-turn)
                continue
            x0, y0_raw, _ = io.odom_raw()
            th = d.odom_pose()[2]
            travelled, _ = d.forward(distance)
            x1, y1_raw, _ = io.odom_raw()
            d.forward(-travelled)
            d.turn_by(-turn)
            dx, dy_raw = x1 - x0, y1_raw - y0_raw
            if math.hypot(dx, dy_raw) < 0.05:
                break
            expected = math.sin(th)          # sideways part of the move in the world frame
            if abs(expected) > 0.5 and dy_raw * self.p["odom_y_sign"] * expected < 0:
                self.p["odom_y_sign"] *= -1
                self.log("Odometry y had the wrong sign -> flipped odom_y_sign")
            self.log(f"Odometry check: moved ({dx:+.2f}, {dy_raw:+.2f}) raw at heading "
                     f"{math.degrees(th):+.0f} deg -> odom_y_sign={self.p['odom_y_sign']}")
            return {"odom_y_sign": self.p["odom_y_sign"]}
        self.log("Odometry y check skipped: no free space to the sides")
        return {}

    # ---- known-distance wall calibration ---------------------------------------------------
    def wall(self, true_distance_m, seconds=2.0):
        d, io = self.d, self.d.io
        d.gimbal_goto(0.0, 0.0)
        io.sleep(0.3)
        t0 = io.now()
        io.sleep(seconds)
        readings = [mm / 1000.0 for _, mm in io.tof_since(t0) if 0 < mm < 9990]
        if len(readings) < 5:
            raise ValueError("no ToF readings - is a wall in front of the robot?")
        med = float(np.median(readings))
        self.p["tof_bias_m"] = round(true_distance_m - med * self.p["tof_scale"], 4)
        self.log(f"Wall calibration: read {med:.3f} m, real {true_distance_m:.3f} m -> "
                 f"tof_bias_m = {self.p['tof_bias_m']:+.3f}")
        return {"tof_bias_m": self.p["tof_bias_m"], "wall_reading_m": round(med, 4)}
