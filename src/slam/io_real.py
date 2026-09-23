"""RoboMaster EP hardware access through the robomaster SDK.

All values are the robot's raw conventions; slam.driver converts them to the
SLAM world frame using the *_sign settings (which auto calibration can set).
"""

import bisect
import collections
import threading
import time


class RealRobotIO:
    name = "robot"

    def __init__(self, connection="ap"):
        from robomaster import robot  # imported here so the simulator works without the SDK

        self._robot_mod = robot
        self.ep = robot.Robot()
        print(f"Connecting to RoboMaster using '{connection}' mode...")
        if connection == "ap":
            self.ep.initialize(conn_type="ap", proto_type="udp")
        else:
            self.ep.initialize(conn_type=connection)
        self._lock = threading.Lock()
        self.odom = (0.0, 0.0, 0.0)          # x, y, yaw_deg (raw)
        self.gimbal = (0.0, 0.0)             # pitch, yaw relative to chassis (raw deg)
        self.gimbal_hist = collections.deque(maxlen=600)  # (t, yaw)
        self.tof = collections.deque(maxlen=4000)         # (t, mm)
        self.battery_pct = None
        self._pos = (0.0, 0.0)
        self._yaw = 0.0
        self._mode = None

        ch = self.ep.chassis
        ch.sub_position(cs=0, freq=50, callback=self._on_position)
        ch.sub_attitude(freq=50, callback=self._on_attitude)
        self.ep.gimbal.sub_angle(freq=50, callback=self._on_gimbal)
        self.ep.sensor.sub_distance(freq=50, callback=self._on_tof)
        try:
            self.ep.battery.sub_battery(freq=1, callback=self._on_battery)
        except Exception:
            pass
        self.set_mode("free")
        time.sleep(0.5)

    # ---- callbacks (SDK threads) ----------------------------------------------
    def _on_position(self, info):
        with self._lock:
            self._pos = (float(info[0]), float(info[1]))
            self.odom = (self._pos[0], self._pos[1], self._yaw)

    def _on_attitude(self, info):
        with self._lock:
            self._yaw = float(info[0])
            self.odom = (self._pos[0], self._pos[1], self._yaw)

    def _on_gimbal(self, info):
        t = time.monotonic()
        with self._lock:
            self.gimbal = (float(info[0]), float(info[1]))
            self.gimbal_hist.append((t, float(info[1])))

    def _on_tof(self, info):
        mm = info[0] if isinstance(info, (list, tuple)) else info
        with self._lock:
            self.tof.append((time.monotonic(), float(mm)))

    def _on_battery(self, pct):
        self.battery_pct = pct

    # ---- time ------------------------------------------------------------------
    def now(self):
        return time.monotonic()

    def sleep(self, seconds):
        time.sleep(max(0.0, seconds))

    # ---- sensors --------------------------------------------------------------
    def odom_raw(self):
        with self._lock:
            return self.odom

    def gimbal_raw(self):
        with self._lock:
            return self.gimbal

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
        return self.battery_pct

    # ---- actuators ----------------------------------------------------------------
    def drive(self, vx, vy, wz):
        self.ep.chassis.drive_speed(x=vx, y=vy, z=wz, timeout=0.5)

    def stop(self):
        try:
            self.ep.chassis.drive_speed(x=0, y=0, z=0)
            self.ep.gimbal.drive_speed(pitch_speed=0, yaw_speed=0)
        except Exception:
            pass

    def gimbal_speed(self, pitch_dps, yaw_dps):
        self.ep.gimbal.drive_speed(pitch_speed=pitch_dps, yaw_speed=yaw_dps)

    def set_mode(self, mode):
        if mode == self._mode:
            return
        robot = self._robot_mod
        self.ep.set_robot_mode(mode={"free": robot.FREE, "chassis_lead": robot.CHASSIS_LEAD,
                                     "gimbal_lead": robot.GIMBAL_LEAD}[mode])
        self._mode = mode

    def close(self):
        self.stop()
        for fn in (self.ep.chassis.unsub_position, self.ep.chassis.unsub_attitude,
                   self.ep.gimbal.unsub_angle, self.ep.sensor.unsub_distance):
            try:
                fn()
            except Exception:
                pass
        try:
            self.ep.battery.unsub_battery()
        except Exception:
            pass
        self.ep.close()


def interpolate(hist, t):
    """Value at time t from a list of (time, value), clamped at the ends."""
    if not hist:
        return 0.0
    times = [h[0] for h in hist]
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return hist[0][1]
    if i >= len(hist):
        return hist[-1][1]
    (t0, v0), (t1, v1) = hist[i - 1], hist[i]
    if t1 == t0:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
