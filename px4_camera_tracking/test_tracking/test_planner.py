"""
路径规划
"""

import math
import time

from test_params import (
    TARGET_POSITION_FILTER_ALPHA,
    TARGET_PREDICT_TIME_S,
    MAX_PREDICT_SPEED_M_S,
    POSITION_SETPOINT_UPDATE_S,
    MAX_POSITION_SETPOINT_STEP_M,
)
from test_utils import clamp, distance_ne, limit_position_setpoint_step

class TargetPositionEstimator:

    def __init__(self, desired_distance_m):
        self._desired_distance_m = desired_distance_m

        self._filtered_n = None
        self._filtered_e = None
        self._alpha = TARGET_POSITION_FILTER_ALPHA

        self._prev_raw_n = None
        self._prev_raw_e = None
        self._prev_time = None

        self._last_setpoint_time = 0.0
        self._prev_follow_n = None
        self._prev_follow_e = None

        self._drone_n = 0.0
        self._drone_e = 0.0

        self.target_n = 0.0
        self.target_e = 0.0
        self.target_vel_n = 0.0
        self.target_vel_e = 0.0
        self.follow_n = 0.0
        self.follow_e = 0.0
        self.position_valid = False
        self.follow_valid = False

    def update(self, bearing_deg, distance_m, drone_n, drone_e, heading_deg):
        now = time.time()

        self._drone_n = drone_n
        self._drone_e = drone_e

        target_heading_deg = heading_deg + bearing_deg
        target_heading_rad = math.radians(target_heading_deg)

        raw_target_n = drone_n + distance_m * math.cos(target_heading_rad)
        raw_target_e = drone_e + distance_m * math.sin(target_heading_rad)

        if self._filtered_n is None:
            self._filtered_n = raw_target_n
            self._filtered_e = raw_target_e
            self._prev_raw_n = raw_target_n
            self._prev_raw_e = raw_target_e
            self._prev_time = now
        else:
            self._filtered_n = (
                self._alpha * raw_target_n
                + (1.0 - self._alpha) * self._filtered_n
            )
            self._filtered_e = (
                self._alpha * raw_target_e
                + (1.0 - self._alpha) * self._filtered_e
            )

        self.target_n = self._filtered_n
        self.target_e = self._filtered_e
        self.position_valid = True

        dt = now - self._prev_time if self._prev_time else 0.1
        if dt > 1e-6 and self._prev_raw_n is not None:
            raw_vel_n = (raw_target_n - self._prev_raw_n) / dt
            raw_vel_e = (raw_target_e - self._prev_raw_e) / dt
            self.target_vel_n = clamp(raw_vel_n, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S)
            self.target_vel_e = clamp(raw_vel_e, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S)

        self._prev_raw_n = raw_target_n
        self._prev_raw_e = raw_target_e
        self._prev_time = now

        self._compute_follow_point()

    def _compute_follow_point(self):
        predict_n = self.target_n + self.target_vel_n * TARGET_PREDICT_TIME_S
        predict_e = self.target_e + self.target_vel_e * TARGET_PREDICT_TIME_S

        dn = self._drone_n - self.target_n
        de = self._drone_e - self.target_e
        drone_dist = math.sqrt(dn ** 2 + de ** 2)

        if drone_dist > 0.5:
            dir_n = dn / drone_dist
            dir_e = de / drone_dist
        else:
            self.follow_valid = False
            return

        raw_follow_n = predict_n + dir_n * self._desired_distance_m
        raw_follow_e = predict_e + dir_e * self._desired_distance_m

        now = time.time()
        if now - self._last_setpoint_time < POSITION_SETPOINT_UPDATE_S:
            return

        if self._prev_follow_n is not None:
            raw_follow_n, raw_follow_e = limit_position_setpoint_step(
                self._prev_follow_n, self._prev_follow_e,
                raw_follow_n, raw_follow_e,
                MAX_POSITION_SETPOINT_STEP_M,
            )

        self.follow_n = raw_follow_n
        self.follow_e = raw_follow_e
        self.follow_valid = True
        self._prev_follow_n = raw_follow_n
        self._prev_follow_e = raw_follow_e
        self._last_setpoint_time = now

    def mark_target_lost(self):
        self.position_valid = False
        self.follow_valid = False

    def reset(self):
        self._filtered_n = None
        self._filtered_e = None
        self._prev_raw_n = None
        self._prev_raw_e = None
        self._prev_time = None
        self._last_setpoint_time = 0.0
        self._prev_follow_n = None
        self._prev_follow_e = None
        self._drone_n = 0.0
        self._drone_e = 0.0
        self.target_n = 0.0
        self.target_e = 0.0
        self.target_vel_n = 0.0
        self.target_vel_e = 0.0
        self.follow_n = 0.0
        self.follow_e = 0.0
        self.position_valid = False
        self.follow_valid = False