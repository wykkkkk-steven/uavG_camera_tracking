"""
路径规划（KF）
"""

import math
import time

import numpy as np

from .test_params import (
    MAX_PREDICT_SPEED_M_S,
    KF_PROCESS_NOISE_POS,
    KF_PROCESS_NOISE_VEL,
    KF_MEASUREMENT_NOISE_POS,
    KF_INITIAL_COV_POS,
    KF_INITIAL_COV_VEL,
    KF_WARMUP_FRAMES,
    KF_MAX_DT_S,
)
from .test_utils import clamp


class _CVKalmanFilter:


    def __init__(self):
        self.x = np.zeros(4)
        self.P = np.diag([
            KF_INITIAL_COV_POS,
            KF_INITIAL_COV_POS,
            KF_INITIAL_COV_VEL,
            KF_INITIAL_COV_VEL,
        ])
        self._initialized = False

    def initialize(self, n, e, vn=0.0, ve=0.0):
        self.x = np.array([n, e, vn, ve], dtype=float)
        self.P = np.diag([
            KF_INITIAL_COV_POS,
            KF_INITIAL_COV_POS,
            KF_INITIAL_COV_VEL,
            KF_INITIAL_COV_VEL,
        ])
        self._initialized = True

    def predict(self, dt):
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        self.x = F @ self.x

        Q = np.diag([
            KF_PROCESS_NOISE_POS,
            KF_PROCESS_NOISE_POS,
            KF_PROCESS_NOISE_VEL,
            KF_PROCESS_NOISE_VEL,
        ])
        self.P = F @ self.P @ F.T + Q

    def update(self, z_n, z_e):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        R = np.diag([KF_MEASUREMENT_NOISE_POS, KF_MEASUREMENT_NOISE_POS])

        y = np.array([z_n, z_e]) - H @ self.x
        S = H @ self.P @ H.T + R

        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    @property
    def pos_n(self):
        return float(self.x[0])

    @property
    def pos_e(self):
        return float(self.x[1])

    @property
    def vel_n(self):
        return float(self.x[2])

    @property
    def vel_e(self):
        return float(self.x[3])

    @property
    def initialized(self):
        return self._initialized


class TargetPositionEstimator:

    def __init__(self, desired_distance_m):
        self._desired_distance_m = desired_distance_m
        self._kf = _CVKalmanFilter()
        self._prev_time = None
        self._frame_count = 0  

        self._drone_n = 0.0
        self._drone_e = 0.0

        self.target_n = 0.0
        self.target_e = 0.0
        self.target_vel_n = 0.0
        self.target_vel_e = 0.0
        self.direction_n = 0.0
        self.direction_e = 0.0
        self.position_valid = False
        self.direction_valid = False

    def update(self, bearing_deg, distance_m, drone_n, drone_e, heading_deg):
        now = time.time()

        self._drone_n = drone_n
        self._drone_e = drone_e

        target_heading_deg = heading_deg + bearing_deg
        target_heading_rad = math.radians(target_heading_deg)

        raw_target_n = drone_n + distance_m * math.cos(target_heading_rad)
        raw_target_e = drone_e + distance_m * math.sin(target_heading_rad)

        if not self._kf.initialized:
            self._kf.initialize(raw_target_n, raw_target_e)
            self._prev_time = now
        else:
            dt = now - self._prev_time if self._prev_time else 0.1
            if dt <= 0:
                dt = 0.01
            elif dt > KF_MAX_DT_S:
                dt = KF_MAX_DT_S
            self._prev_time = now

            self._kf.predict(dt)
            self._kf.update(raw_target_n, raw_target_e)

        self._frame_count += 1

        if self._frame_count <= KF_WARMUP_FRAMES:
            self.target_n = raw_target_n
            self.target_e = raw_target_e
            self.target_vel_n = 0.0
            self.target_vel_e = 0.0
        else:
            self.target_n = self._kf.pos_n
            self.target_e = self._kf.pos_e
            self.target_vel_n = clamp(
                self._kf.vel_n, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S
            )
            self.target_vel_e = clamp(
                self._kf.vel_e, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S
            )
        self.position_valid = True

        dn = self._drone_n - self.target_n
        de = self._drone_e - self.target_e
        drone_dist = math.sqrt(dn ** 2 + de ** 2)

        if drone_dist > 0.5:
            self.direction_n = dn / drone_dist
            self.direction_e = de / drone_dist
            self.direction_valid = True
        else:
            self.direction_valid = False

    def mark_target_lost(self):
        self.position_valid = False
        self.direction_valid = False

    def reset(self):
        self._kf = _CVKalmanFilter()
        self._prev_time = None
        self._frame_count = 0
        self._drone_n = 0.0
        self._drone_e = 0.0
        self.target_n = 0.0
        self.target_e = 0.0
        self.target_vel_n = 0.0
        self.target_vel_e = 0.0
        self.direction_n = 0.0
        self.direction_e = 0.0
        self.position_valid = False
        self.direction_valid = False
