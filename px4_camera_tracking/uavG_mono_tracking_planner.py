import asyncio
import select
import signal
import sys
import time
import math

import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, PositionNedYaw


# ================= 基本连接参数 =================
UDP_ADDR = "udpin://0.0.0.0:14540"

IMAGE_TOPIC = "/camera/image_raw"
TAKEOFF_ALT_M = 5.0
CAMERA_INFO_TOPIC = "/camera/camera_info"

# ================= Home / hover command parameters =================
REACHED_XY_THR_M = 1.0
REACHED_ALT_THR_M = 0.5
WAIT_REACHED_TIMEOUT_S = 45.0

# ================= 搜索 / 绕飞参数 =================
CONTROL_HZ = 10.0

# 起飞后先不搜索，用户输入 START 后才开始搜索
SEARCH_YAW_RATE_DEG_S = 5.0
SEARCH_TIMEOUT_S = 80.0
SEARCH_CONFIRM_MIN_FRAMES = 2

# orbit 中目标丢失保护
TARGET_LOST_HOVER_S = 2.0
TARGET_LOST_LAND_S = 20.0

# OpenCV 视觉控制参数
KP_YAW = 0.10
KP_FORWARD = 0.00060

DESIRED_AREA = 3500.0
MIN_TARGET_AREA = 90.0

MAX_FORWARD_SPEED = 2.00
MAX_BACKWARD_SPEED = 1.80
ORBIT_RIGHT_SPEED = 0.55
MAX_YAW_RATE_DEG_S = 75.0
CENTER_ERROR_LIMIT = 60.0

KP_YAW_NORM = 12.0
MAX_TRACK_YAW_RATE_DEG_S = 15.0
YAW_DEADBAND_PX = 90.0
YAW_SLEW_RATE_DEG_S2 = 25.0
# ================= Monocular whole-board pose / tracking =================
CAMERA_INFO_TIMEOUT_S = 2.0

# IMPORTANT: these four values must match the ChArUco board model exactly.
# Current defaults assume a 5 x 5 board, 0.40 m square length and
# 0.32 m ArUco marker length.
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 7
CHARUCO_SQUARE_LENGTH_M = 3.0 / 7.0
CHARUCO_MARKER_LENGTH_M = 0.30
CHARUCO_LEGACY_PATTERN = False

DEFAULT_DESIRED_DISTANCE_M = 7.5
MAX_VALID_MONO_DISTANCE_M = 30.0
MIN_CHARUCO_CORNERS_FOR_POSE = 4
POSE_MAX_REPROJECTION_ERROR_PX = 8.0
POSE_FILTER_ALPHA = 0.35
NORMAL_FILTER_ALPHA = 0.30

# Simple controller: ALIGN -> TRACK. Emergency retreat is only a priority
# override, not another persistent state.
MAX_TRACK_FORWARD_SPEED = 0.75
MAX_TRACK_BACKWARD_SPEED = 0.75
MAX_TRACK_LATERAL_SPEED = 0.30

KP_OBSERVATION_POINT = 0.30
KP_BEARING_YAW = 1.10
MAX_TRACK_YAW_RATE_DEG_S = 15.0
BEARING_DEADBAND_DEG = 1.5

ALIGN_BEARING_DEG = 2.0
ALIGN_CONFIRM_FRAMES = 5
REALIGN_BEARING_DEG = 10.0

DISTANCE_TOLERANCE_M = 0.40
CENTER_TOLERANCE_DEG = 2.0
FACE_TOLERANCE_DEG = 5.0
LATERAL_TOLERANCE_M = 0.35
ARRIVAL_CONFIRM_FRAMES = 5

TARGET_PREDICTION_TIME_S = 0.20
TARGET_VELOCITY_FILTER_ALPHA = 0.25
MAX_PREDICT_SPEED_M_S = 2.0
VELOCITY_FEEDFORWARD_GAIN = 0.70

URGENT_TOO_CLOSE_MARGIN_M = 0.80
URGENT_CLOSING_SPEED_M_S = 0.70
URGENT_CONFIRM_FRAMES = 3
URGENT_RETREAT_MIN_SPEED = 0.20
URGENT_RETREAT_MAX_SPEED = 0.75
URGENT_RETREAT_KP = 0.55

FORWARD_SLEW_M_S2 = 0.90
LATERAL_SLEW_M_S2 = 0.60
YAW_SLEW_DEG_S2 = 30.0

POSE_LOST_HOVER_S = 0.60

# ================= CV-KF path planning (Planner mode) =================
# Direction (yaw) and distance (forward) are controlled independently:
#   - KF estimates target N/E position and velocity from ChArUco 3D pose
#   - yaw points at the predicted target position (position + velocity * T)
#   - forward uses the measured distance error with a hard P controller
#   - KF never touches distance control, distance never touches yaw
PLANNER_ENABLED = False
KF_PROCESS_NOISE_POS = 0.01
KF_PROCESS_NOISE_VEL = 0.10
KF_MEASUREMENT_NOISE_POS = 0.25
KF_INITIAL_COV_POS = 10.0
KF_INITIAL_COV_VEL = 5.0
KF_WARMUP_FRAMES = 20
KF_MAX_DT_S = 0.5
PLANNER_KP_YAW = 0.80
PLANNER_KP_DIST = 0.30
PLANNER_MAX_FORWARD = 1.0
PLANNER_MAX_RETREAT = 0.75
PLANNER_DEADBAND_M = 0.5
PLANNER_SAFE_DISTANCE_M = 1.0
PLANNER_PREDICT_TIME_S = 0.25
PLANNER_FF_GAIN = 0.70

# Target point offset in the board frame (metres). (0, 0) = board centre.
# Set these to follow a specific pixel block / ArUco marker on the board.
TARGET_POINT_OFFSET_X_M = 0.0
TARGET_POINT_OFFSET_Y_M = 0.0

# ================= Minimal smooth-orbit fixes =================
# Keep the original simple controller, but stop DEPTH_ORBIT from immediately
# switching back to DEPTH_APPROACH for small range drift.
ORBIT_DISTANCE_BAND_M = 1.00
ORBIT_CORRECTION_SPEED = 0.30
ORBIT_REAPPROACH_ERROR_M = 1.60
APPROACH_FULL_SPEED_ERROR_M = 1.20
MIN_DEPTH_APPROACH_SPEED = 0.30
MIN_AREA_APPROACH_SPEED = 0.35
AREA_FAST_APPROACH_RATIO = 0.70

# Optional visibility of whether this Python OpenCV build can see CUDA/OpenCL.
PRINT_VISION_ACCELERATION = True

# ArUco / ChArUco target detection
# Five-face vehicle target: one ArUco marker per face, DICT_4X4_1000, ID 0-4.
# Board 390x390 mm, ArUco marker 320x320 mm, white border 35 mm.
USE_CHARUCO_TARGET = False
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_1000
TARGET_MARKER_IDS = {0, 1, 2, 3, 4}
ARUCO_MARKER_LENGTH_M = 0.320
MIN_MARKER_AREA = 60.0
DEBUG_ARUCO_DETECTION = False

# Image-edge safety: if the target is going out of the bottom of the camera view,
# do not trust area; back off horizontally.
TARGET_LOW_RATIO = 0.78
BBOX_BOTTOM_RATIO = 0.95
IMAGE_EDGE_BACKOFF_SPEED = -0.35

# ================= Local waypoint position tracking parameters =================
# B version:
# - estimate target local N/E using UAV local position + heading + depth/image error
# - filter/predict target N/E to reduce visual delay and jitter
# - generate a full follow point
# - keep MAX_POSITION_SETPOINT_STEP_M to prevent sudden setpoint jumps
# - update the active PositionNedYaw setpoint only when the change is meaningful
TARGET_POSITION_FILTER_ALPHA = 0.30
TARGET_PREDICT_TIME_S = 0.25
MAX_PREDICT_SPEED_M_S = 1.20

POSITION_SETPOINT_UPDATE_S = 0.30
POSITION_UPDATE_DISTANCE_M = 0.30
YAW_UPDATE_DEG = 5.0
MAX_POSITION_SETPOINT_STEP_M = 1.20

# Keep B position tracking stable during short visual/pose dropouts.
POSITION_TRACK_HOLD_AFTER_DEPTH_LOST_S = 1.2
POSITION_TRACK_HOLD_AFTER_TARGET_LOST_S = 1.5

# After a short target-loss hold, search by yawing in place.
# 8 deg/s gives one full 360 deg search turn in 45 s.
LOST_YAW_SEARCH_RATE_DEG_S = 8.0
LOST_YAW_SEARCH_FULL_TURN_DEG = 360.0
LOST_YAW_SEARCH_TIMEOUT_S = (
    LOST_YAW_SEARCH_FULL_TURN_DEG / LOST_YAW_SEARCH_RATE_DEG_S
)

def clamp(value, low, high):
    return max(low, min(high, value))


def slew_limit(target, previous, max_rate_per_s, dt):
    max_step = max_rate_per_s * dt
    return clamp(target, previous - max_step, previous + max_step)


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return vector / norm


def angle_between_vectors_deg(a, b):
    a_n = normalize_vector(a)
    b_n = normalize_vector(b)
    if a_n is None or b_n is None:
        return None
    dot_value = clamp(float(np.dot(a_n, b_n)), -1.0, 1.0)
    return math.degrees(math.acos(dot_value))


def depth_approach_speed(distance_error):
    """Fast approach when far, then taper near the desired distance."""
    if abs(distance_error) < DISTANCE_DEADBAND_M:
        return 0.0

    if distance_error > 0.0:
        if distance_error >= APPROACH_FULL_SPEED_ERROR_M:
            return MAX_FORWARD_SPEED

        return clamp(
            max(KP_DISTANCE * distance_error, MIN_DEPTH_APPROACH_SPEED),
            0.0,
            MAX_FORWARD_SPEED
        )

    return clamp(
        KP_DISTANCE * distance_error,
        -MAX_BACKWARD_SPEED,
        0.0
    )


async def read_console_line(prompt, stop_event=None):
    print(prompt, end="", flush=True)

    while True:
        if stop_event is not None and stop_event.is_set():
            print()
            return None

        ready, _, _ = select.select([sys.stdin], [], [], 0.05)
        if ready:
            line = sys.stdin.readline()
            if line == "":
                return ""
            return line.rstrip("\n")

        await asyncio.sleep(0.05)
async def ask_float(prompt, default_value, min_value=None, max_value=None, stop_event=None):
    while True:
        answer = await read_console_line(
            f"{prompt} 默认 {default_value}: ",
            stop_event=stop_event
        )

        if answer is None:
            return None

        answer = answer.strip()

        if answer == "":
            value = default_value
        else:
            try:
                value = float(answer)
            except ValueError:
                print("请输入数字，或者直接回车使用默认值")
                continue

        if min_value is not None and value < min_value:
            print(f"数值不能小于 {min_value}")
            continue

        if max_value is not None and value > max_value:
            print(f"数值不能大于 {max_value}")
            continue

        return value

async def get_in_air_once(drone, timeout_s=2.0):
    try:
        return await asyncio.wait_for(
            drone.telemetry.in_air().__anext__(),
            timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return None



def meters_from_gps(lat0_deg, lon0_deg, lat_deg, lon_deg):
    dn = (lat_deg - lat0_deg) * 111_320.0
    de = (lon_deg - lon0_deg) * 111_320.0 * math.cos(math.radians(lat0_deg))
    return dn, de


async def get_position_once(drone, timeout_s=2.0):
    try:
        return await asyncio.wait_for(
            drone.telemetry.position().__anext__(),
            timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return None


async def print_position_once(drone, prefix="[位置]"):
    pos = await get_position_once(drone)
    if pos is None:
        print(f"{prefix} 暂无位置数据")
        return

    print(
        f"{prefix} "
        f"lat={pos.latitude_deg:.7f} "
        f"lon={pos.longitude_deg:.7f} "
        f"abs_alt={pos.absolute_altitude_m:.2f}m "
        f"rel_alt={pos.relative_altitude_m:.2f}m"
    )


async def get_heading_once(drone, timeout_s=1.0):
    try:
        heading = await asyncio.wait_for(
            drone.telemetry.heading().__anext__(),
            timeout=timeout_s
        )
        return heading.heading_deg
    except asyncio.TimeoutError:
        return None
    except Exception:
        return None


def estimate_target_body_from_image(detector):
    """Rough target position in UAV body frame using current image error and depth.

    Assumption for this first version:
    - camera is approximately facing forward
    - image x error maps to body-right offset
    - depth gives approximate forward distance
    """
    if detector.target_distance_m is None:
        return None

    if detector.image_w <= 0:
        return None

    half_w = detector.image_w / 2.0
    if half_w <= 1e-6:
        return None

    target_forward_body = detector.target_distance_m
    target_right_body = (
        detector.error_x / half_w
    ) * detector.target_distance_m

    return target_forward_body, target_right_body


def body_to_local_ne(drone_n, drone_e, heading_deg, forward_body, right_body):
    yaw_rad = math.radians(heading_deg)

    # PX4/MAVSDK heading convention:
    # heading 0 deg = North, 90 deg = East.
    # body forward axis in N/E = [cos(yaw), sin(yaw)]
    # body right axis in N/E   = [-sin(yaw), cos(yaw)]
    target_n = drone_n + forward_body * math.cos(yaw_rad) - right_body * math.sin(yaw_rad)
    target_e = drone_e + forward_body * math.sin(yaw_rad) + right_body * math.cos(yaw_rad)

    return target_n, target_e


def local_ne_error_to_body_velocity(err_n, err_e, heading_deg):
    yaw_rad = math.radians(heading_deg)

    forward_cmd = err_n * math.cos(yaw_rad) + err_e * math.sin(yaw_rad)
    right_cmd = -err_n * math.sin(yaw_rad) + err_e * math.cos(yaw_rad)

    return forward_cmd, right_cmd


def generate_follow_point(drone_n, drone_e, target_n, target_e, desired_distance_m):
    vec_n = target_n - drone_n
    vec_e = target_e - drone_e
    target_range = math.sqrt(vec_n * vec_n + vec_e * vec_e)

    if target_range < 1e-3:
        return drone_n, drone_e, target_range

    follow_n = target_n - desired_distance_m * vec_n / target_range
    follow_e = target_e - desired_distance_m * vec_e / target_range

    return follow_n, follow_e, target_range


def limit_position_setpoint_step(prev_n, prev_e, new_n, new_e, max_step_m):
    """Limit how far the active position setpoint is allowed to jump.

    This is not a small-step trajectory generator. The follow point is still the
    full target-relative waypoint. This only rejects sudden visual-estimation
    jumps that could make PX4 rush toward a bad position setpoint.
    """
    if prev_n is None or prev_e is None:
        return new_n, new_e

    dn = new_n - prev_n
    de = new_e - prev_e
    dist = math.sqrt(dn * dn + de * de)

    if dist <= max_step_m or dist < 1e-6:
        return new_n, new_e

    scale = max_step_m / dist
    return prev_n + dn * scale, prev_e + de * scale


def distance_ne(n1, e1, n2, e2):
    dn = n1 - n2
    de = e1 - e2
    return math.sqrt(dn * dn + de * de)


def angle_diff_deg(a, b):
    """Smallest signed difference a-b in degrees, returned in [-180, 180]."""
    return (a - b + 180.0) % 360.0 - 180.0


class CVKalmanFilter:
    """Constant-velocity Kalman filter for target N/E position and velocity.

    State: x = [n, e, vn, ve].
    Observations: target N/E (from ChArUco solvePnP transformed to local NE).
    """

    def __init__(self):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag(
            [
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_VEL,
                KF_INITIAL_COV_VEL,
            ]
        )
        self._initialized = False

    def initialize(self, n, e, vn=0.0, ve=0.0):
        self.x = np.array([n, e, vn, ve], dtype=np.float64)
        self.P = np.diag(
            [
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_VEL,
                KF_INITIAL_COV_VEL,
            ]
        )
        self._initialized = True

    def predict(self, dt):
        dt = max(dt, 1e-3)
        F = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        Q = np.diag(
            [
                KF_PROCESS_NOISE_POS,
                KF_PROCESS_NOISE_POS,
                KF_PROCESS_NOISE_VEL,
                KF_PROCESS_NOISE_VEL,
            ]
        )
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_n, z_e):
        H = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        R = np.diag([KF_MEASUREMENT_NOISE_POS, KF_MEASUREMENT_NOISE_POS])
        y = np.array([z_n, z_e], dtype=np.float64) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P

    def reset(self):
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.diag(
            [
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_POS,
                KF_INITIAL_COV_VEL,
                KF_INITIAL_COV_VEL,
            ]
        )
        self._initialized = False

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


async def wait_reached_offboard(
    drone,
    north_tgt,
    east_tgt,
    down_tgt,
    ref_lat,
    ref_lon,
    xy_thr=REACHED_XY_THR_M,
    alt_thr=REACHED_ALT_THR_M,
    timeout_s=WAIT_REACHED_TIMEOUT_S,
    stop_event=None,
    land_event=None,
):
    deadline = time.time() + timeout_s
    last_error = None

    while time.time() < deadline:
        if stop_event is not None and stop_event.is_set():
            return False
        if land_event is not None and land_event.is_set():
            return False

        await drone.offboard.set_position_ned(
            PositionNedYaw(north_tgt, east_tgt, down_tgt, 0.0)
        )

        try:
            pos = await asyncio.wait_for(
                drone.telemetry.position().__anext__(),
                timeout=1.5
            )
        except asyncio.TimeoutError:
            continue

        dn, de = meters_from_gps(
            ref_lat,
            ref_lon,
            pos.latitude_deg,
            pos.longitude_deg
        )

        d_now = -pos.relative_altitude_m
        north_err = north_tgt - dn
        east_err = east_tgt - de
        down_err = down_tgt - d_now
        last_error = (north_err, east_err, down_err)

        if (
            abs(north_err) < xy_thr and
            abs(east_err) < xy_thr and
            abs(down_err) < alt_thr
        ):
            return True

        await asyncio.sleep(1.0 / CONTROL_HZ)

    if last_error is None:
        print(f"[HOME] 等待到达目标点超时（{timeout_s:.0f}s），未收到位置数据")
        return False

    north_err, east_err, down_err = last_error
    print(
        f"[HOME] 等待到达目标点超时（{timeout_s:.0f}s），"
        f"剩余偏差: 北{north_err:+.1f}m 东{east_err:+.1f}m Down{down_err:+.1f}m"
    )
    return False


async def send_zero_velocity_loop(drone, stop_event, land_event):
    dt = 1.0 / CONTROL_HZ
    last_warn = 0.0

    while not stop_event.is_set() and not land_event.is_set():
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
        except Exception as e:
            now = time.time()
            if now - last_warn > 2.0:
                last_warn = now
                print(f"[HOVER] 发送零速度失败/跳过: {e}")

        await asyncio.sleep(dt)


async def return_home_and_hover(drone, ref_lat, ref_lon, stop_event, land_event):
    print(
        f"[HOME] 返回起飞点上方并悬停，速度限制使用 MAX_FORWARD_SPEED={MAX_FORWARD_SPEED:.2f} m/s"
    )

    dt = 1.0 / CONTROL_HZ
    timeout_s = WAIT_REACHED_TIMEOUT_S
    deadline = time.time() + timeout_s
    last_print = 0.0

    target_north = 0.0
    target_east = 0.0
    target_alt = TAKEOFF_ALT_M

    while time.time() < deadline:
        if stop_event.is_set() or land_event.is_set():
            return

        pos = await get_position_once(drone, timeout_s=1.0)
        if pos is None:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(dt)
            continue

        north_now, east_now = meters_from_gps(
            ref_lat,
            ref_lon,
            pos.latitude_deg,
            pos.longitude_deg
        )

        north_err = target_north - north_now
        east_err = target_east - east_now
        alt_err = target_alt - pos.relative_altitude_m

        xy_dist = math.sqrt(north_err * north_err + east_err * east_err)

        if xy_dist < REACHED_XY_THR_M and abs(alt_err) < REACHED_ALT_THR_M:
            print("[HOME] 已返回 home 点上方")
            break

        heading_deg = 0.0
        try:
            heading = await asyncio.wait_for(
                drone.telemetry.heading().__anext__(),
                timeout=0.5
            )
            heading_deg = heading.heading_deg
        except Exception:
            heading_deg = 0.0

        yaw_rad = math.radians(heading_deg)

        # 距离远时接近 MAX_FORWARD_SPEED，距离近时自动减速，避免冲过头。
        xy_speed = clamp(
            xy_dist * 0.5,
            0.0,
            MAX_FORWARD_SPEED
        )

        if xy_dist > 1e-3:
            vel_n = north_err / xy_dist * xy_speed
            vel_e = east_err / xy_dist * xy_speed
        else:
            vel_n = 0.0
            vel_e = 0.0

        # world N/E velocity -> body forward/right velocity
        forward_speed = vel_n * math.cos(yaw_rad) + vel_e * math.sin(yaw_rad)
        right_speed = -vel_n * math.sin(yaw_rad) + vel_e * math.cos(yaw_rad)

        forward_speed = clamp(
            forward_speed,
            -MAX_BACKWARD_SPEED,
            MAX_FORWARD_SPEED
        )

        right_speed = clamp(
            right_speed,
            -MAX_FORWARD_SPEED,
            MAX_FORWARD_SPEED
        )

        # MAVSDK body velocity 第三个量是 down。
        # alt_err > 0 表示当前高度低于目标高度，需要向上飞，所以 down_speed 应该为负。
        down_speed = clamp(
            -0.4 * alt_err,
            -0.4,
            0.4
        )

        # 如果高度已经接近目标，就不要再上下修正。
        if abs(alt_err) < REACHED_ALT_THR_M:
            down_speed = 0.0

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_speed,
                right_speed,
                down_speed,
                0.0
            )
        )

        now = time.time()
        if now - last_print > 0.5:
            last_print = now
            print(
                f"[HOME] dist={xy_dist:.1f}m "
                f"alt_err={alt_err:+.1f}m "
                f"forward={forward_speed:+.2f} "
                f"right={right_speed:+.2f} "
                f"down={down_speed:+.2f} "
                f"heading={heading_deg:.1f}"
            )

        await asyncio.sleep(dt)

    else:
        print("[HOME] 返回 home 超时，保持当前位置悬停")

    await print_position_once(drone, prefix="[HOME后位置]")

    for _ in range(10):
        if stop_event.is_set() or land_event.is_set():
            break
        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
        )
        await asyncio.sleep(0.1)


async def hover_wait_for_command(drone, stop_event, land_event, ref_lat, ref_lon):
    print("\n[WAIT] 当前保持原地悬停，等待下一个指令。")
    print("可用指令：START / home / position / land / stop / 0")

    hover_stop_event = asyncio.Event()
    hover_task = asyncio.create_task(
        send_zero_velocity_loop(drone, hover_stop_event, land_event)
    )

    try:
        while not stop_event.is_set() and not land_event.is_set():
            line = await read_console_line("\n输入指令: ", stop_event=stop_event)

            if line is None:
                return "stop"

            cmd = line.strip().lower()

            if not cmd:
                continue

            if cmd == "start":
                print("[WAIT] 收到 START，准备开始 search / tracking")
                return "start"

            if cmd in ("land", "stop", "0"):
                print("[WAIT] 收到降落/停止指令")
                land_event.set()
                return "land"

            if cmd == "home":
                hover_stop_event.set()
                try:
                    await hover_task
                except asyncio.CancelledError:
                    pass

                await return_home_and_hover(
                    drone,
                    ref_lat,
                    ref_lon,
                    stop_event,
                    land_event,
                )

                if stop_event.is_set() or land_event.is_set():
                    break

                hover_stop_event = asyncio.Event()
                hover_task = asyncio.create_task(
                    send_zero_velocity_loop(drone, hover_stop_event, land_event)
                )
                continue

            if cmd in ("position", "pos"):
                await print_position_once(drone)
                continue

            print("未知指令。可用指令：START / home / position / land / stop / 0")

    finally:
        hover_stop_event.set()
        hover_task.cancel()
        try:
            await hover_task
        except asyncio.CancelledError:
            pass

    return "stop"



async def wait_until_landed(drone, timeout_s=60.0):
    deadline = time.time() + timeout_s

    while time.time() < deadline:
        in_air = await get_in_air_once(drone, timeout_s=2.0)

        if in_air is False:
            print("[LAND] 已确认着陆")
            return True

        await asyncio.sleep(0.5)

    print("[LAND] 等待着陆超时，请人工确认 Gazebo/PX4 状态")
    return False


async def safe_stop_and_land(drone, reason="安全降落"):
    print(f"\n[{reason}] 开始安全停止流程")

    try:
        print("[SAFE] 发送零速度")
        for _ in range(10):
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
            )
            await asyncio.sleep(0.1)
    except Exception as e:
        print(f"[SAFE] 发送零速度失败/跳过: {e}")

    try:
        print("[SAFE] 停止 Offboard")
        await drone.offboard.stop()
    except OffboardError as e:
        print(f"[SAFE] Offboard stop 失败/跳过: {e}")
    except Exception as e:
        print(f"[SAFE] Offboard stop 失败/跳过: {e}")

    try:
        in_air = await get_in_air_once(drone, timeout_s=2.0)
        if in_air is False:
            print("[SAFE] 当前已不在空中，跳过 land")
            return

        print("[SAFE] 发送 land 指令")
        await drone.action.land()

    except Exception as e:
        print(f"[SAFE] land 指令失败: {e}")
        return

    await wait_until_landed(drone)


class CharucoTargetDetector(Node):
    def __init__(self):
        super().__init__("charuco_target_detector")

        self.bridge = CvBridge()

        self.target_found = False
        self.board_pose_valid = False
        self.error_x = 0.0
        self.error_y = 0.0
        self.area = 0.0
        self.last_seen_time = 0.0
        self.frame_count = 0

        self.bbox_x = 0
        self.bbox_y = 0
        self.bbox_w = 0
        self.bbox_h = 0
        self.target_cx = 0.0
        self.target_cy = 0.0
        self.image_w = 0
        self.image_h = 0

        # Whole-board pose in the OpenCV camera frame:
        # x right, y down, z forward.
        self.target_x_m = None
        self.target_y_m = None
        self.target_z_m = None
        self.target_distance_m = None
        self.target_forward_range_m = None
        self.target_bearing_deg = None

        # Board normal is explicitly oriented from the board toward the camera.
        self.board_normal_x = None
        self.board_normal_y = None
        self.board_normal_z = None
        self.target_face_error_deg = None
        self.target_lateral_error_m = None
        self.target_normal_range_m = None

        self.target_vx_m_s = 0.0
        self.target_vy_m_s = 0.0
        self.target_vz_m_s = 0.0
        self.last_pose_time = 0.0
        self.previous_pose_time = None
        self.previous_center = None

        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_time = 0.0

        self.lost_frame_count = 0
        self.max_lost_frames = 10
        self.last_aruco_debug_time = 0.0

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTIONARY
        )
        self.aruco_parameters = cv2.aruco.DetectorParameters()

        if hasattr(cv2.aruco, "ArucoDetector"):
            self.aruco_detector = cv2.aruco.ArucoDetector(
                self.aruco_dictionary,
                self.aruco_parameters
            )
        else:
            self.aruco_detector = None

        if USE_CHARUCO_TARGET:
            # OpenCV 4.x compatible ChArUco board construction.
            try:
                self.charuco_board = cv2.aruco.CharucoBoard(
                    (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
                    CHARUCO_SQUARE_LENGTH_M,
                    CHARUCO_MARKER_LENGTH_M,
                    self.aruco_dictionary,
                )
            except Exception:
                self.charuco_board = cv2.aruco.CharucoBoard_create(
                    CHARUCO_SQUARES_X,
                    CHARUCO_SQUARES_Y,
                    CHARUCO_SQUARE_LENGTH_M,
                    CHARUCO_MARKER_LENGTH_M,
                    self.aruco_dictionary,
                )

            if (
                CHARUCO_LEGACY_PATTERN
                and hasattr(self.charuco_board, "setLegacyPattern")
            ):
                self.charuco_board.setLegacyPattern(True)

            if hasattr(cv2.aruco, "CharucoDetector"):
                self.charuco_detector = cv2.aruco.CharucoDetector(
                    self.charuco_board
                )
            else:
                self.charuco_detector = None
        else:
            self.charuco_board = None
            self.charuco_detector = None

        if USE_CHARUCO_TARGET:
            self.get_logger().info(
                "Using whole-board ChArUco pose: "
                f"{CHARUCO_SQUARES_X}x{CHARUCO_SQUARES_Y}, "
                f"square={CHARUCO_SQUARE_LENGTH_M:.3f}m, "
                f"marker={CHARUCO_MARKER_LENGTH_M:.3f}m, "
                f"dictionary={ARUCO_DICTIONARY}"
            )
        else:
            self.get_logger().info(
                "Using single ArUco marker pose: "
                f"marker_ids={sorted(TARGET_MARKER_IDS)}, "
                f"marker_length={ARUCO_MARKER_LENGTH_M:.3f}m, "
                f"dictionary={ARUCO_DICTIONARY}"
            )

        self.sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            10
        )

        self.get_logger().info(f"Subscribing to RGB: {IMAGE_TOPIC}")
        self.get_logger().info(
            f"Subscribing to camera info: {CAMERA_INFO_TOPIC}"
        )

        if PRINT_VISION_ACCELERATION:
            cuda_count = 0
            try:
                if hasattr(cv2, "cuda"):
                    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            except Exception:
                cuda_count = 0

            opencl_available = False
            opencl_enabled = False
            try:
                opencl_available = cv2.ocl.haveOpenCL()
                cv2.ocl.setUseOpenCL(True)
                opencl_enabled = cv2.ocl.useOpenCL()
            except Exception:
                pass

            self.get_logger().info(
                f"Vision acceleration check: CUDA devices={cuda_count}, "
                f"OpenCL available={opencl_available}, "
                f"OpenCL enabled={opencl_enabled}, "
                f"OpenCV={cv2.__version__}"
            )

    def _mark_target_lost(self):
        self.target_found = False
        self.board_pose_valid = False
        self.lost_frame_count += 1

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(
            msg.k,
            dtype=np.float64
        ).reshape(3, 3)

        if len(msg.d) > 0:
            self.dist_coeffs = np.array(
                msg.d,
                dtype=np.float64
            ).reshape(-1, 1)
        else:
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.camera_info_time = time.time()

    def _ensure_camera_matrix(self, image_w, image_h):
        if self.camera_matrix is not None:
            return

        # Fallback for initial testing only. camera_info remains preferred.
        fx = image_w / (2.0 * math.tan(math.radians(90.0) / 2.0))
        fy = fx
        self.camera_matrix = np.array(
            [
                [fx, 0.0, image_w / 2.0],
                [0.0, fy, image_h / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    def _interpolate_charuco(self, gray, marker_corners, marker_ids):
        if self.charuco_detector is not None:
            try:
                charuco_corners, charuco_ids, _, _ = (
                    self.charuco_detector.detectBoard(
                        gray,
                        None,
                        None,
                        marker_corners,
                        marker_ids,
                    )
                )
                return charuco_corners, charuco_ids
            except Exception:
                # Some OpenCV Python builds only accept the image argument.
                try:
                    charuco_corners, charuco_ids, _, _ = (
                        self.charuco_detector.detectBoard(gray)
                    )
                    return charuco_corners, charuco_ids
                except Exception:
                    pass

        if hasattr(cv2.aruco, "interpolateCornersCharuco"):
            try:
                _, charuco_corners, charuco_ids = (
                    cv2.aruco.interpolateCornersCharuco(
                        marker_corners,
                        marker_ids,
                        gray,
                        self.charuco_board,
                        self.camera_matrix,
                        self.dist_coeffs,
                    )
                )
                return charuco_corners, charuco_ids
            except Exception:
                return None, None

        return None, None

    def image_callback(self, msg):
        self.frame_count += 1

        frame_rgb = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="rgb8"
        )
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self.aruco_detector is not None:
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray,
                self.aruco_dictionary,
                parameters=self.aruco_parameters,
            )

        now = time.time()

        if ids is None or len(corners) == 0:
            if (
                DEBUG_ARUCO_DETECTION
                and now - self.last_aruco_debug_time > 1.0
            ):
                self.last_aruco_debug_time = now
                self.get_logger().info(
                    f"No ArUco marker detected in "
                    f"{msg.width}x{msg.height}"
                )
            self._mark_target_lost()
            return

        valid_corners = []
        valid_ids = []
        all_points = []
        best_marker_corners = None
        best_marker_area = 0.0

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if (
                TARGET_MARKER_IDS
                and int(marker_id) not in TARGET_MARKER_IDS
            ):
                continue

            points = marker_corners.reshape(4, 2).astype(np.float32)
            marker_area = cv2.contourArea(points)
            if marker_area < MIN_MARKER_AREA:
                continue

            valid_corners.append(
                points.reshape(1, 4, 2)
            )
            valid_ids.append([int(marker_id)])
            all_points.append(points)

            if marker_area > best_marker_area:
                best_marker_area = marker_area
                best_marker_corners = points

        if not valid_corners:
            self._mark_target_lost()
            return

        valid_ids = np.asarray(valid_ids, dtype=np.int32)
        stacked_points = np.vstack(all_points)

        x, y, w, h = cv2.boundingRect(
            stacked_points.astype(np.int32)
        )
        if w <= 0 or h <= 0:
            self._mark_target_lost()
            return

        self.bbox_x = x
        self.bbox_y = y
        self.bbox_w = w
        self.bbox_h = h
        self.target_cx = x + w / 2.0
        self.target_cy = y + h / 2.0
        self.image_w = msg.width
        self.image_h = msg.height
        self.error_x = self.target_cx - msg.width / 2.0
        self.error_y = self.target_cy - msg.height / 2.0
        self.area = float(w * h)

        # Search is intentionally based on marker visibility, not on whether
        # whole-board pose is already available.
        self.target_found = True
        self.last_seen_time = now
        self.lost_frame_count = 0

        self._ensure_camera_matrix(msg.width, msg.height)

        if USE_CHARUCO_TARGET:
            charuco_corners, charuco_ids = self._interpolate_charuco(
                gray,
                valid_corners,
                valid_ids,
            )

            if (
                charuco_corners is None
                or charuco_ids is None
                or len(charuco_ids) < MIN_CHARUCO_CORNERS_FOR_POSE
            ):
                self.board_pose_valid = False
                return

            self._estimate_whole_board_pose(
                charuco_corners,
                charuco_ids,
            )
        else:
            self._estimate_single_marker_pose(best_marker_corners)

    def _estimate_whole_board_pose(
        self,
        charuco_corners,
        charuco_ids,
    ):
        chessboard_points = np.asarray(
            self.charuco_board.getChessboardCorners(),
            dtype=np.float32,
        ).reshape(-1, 3)

        corner_indices = charuco_ids.reshape(-1).astype(np.int32)
        if np.any(corner_indices < 0):
            self.board_pose_valid = False
            return
        if np.any(corner_indices >= len(chessboard_points)):
            self.board_pose_valid = False
            return

        object_points = chessboard_points[corner_indices]
        image_points = np.asarray(
            charuco_corners,
            dtype=np.float32,
        ).reshape(-1, 2)

        if len(object_points) < MIN_CHARUCO_CORNERS_FOR_POSE:
            self.board_pose_valid = False
            return

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.board_pose_valid = False
            return

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        reprojection_error = float(
            np.mean(
                np.linalg.norm(
                    projected.reshape(-1, 2) - image_points,
                    axis=1,
                )
            )
        )
        if reprojection_error > POSE_MAX_REPROJECTION_ERROR_PX:
            self.board_pose_valid = False
            return

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Board frame origin is at the board corner, not at its centre.
        board_center_object = np.array(
            [
                CHARUCO_SQUARES_X
                * CHARUCO_SQUARE_LENGTH_M
                / 2.0,
                CHARUCO_SQUARES_Y
                * CHARUCO_SQUARE_LENGTH_M
                / 2.0,
                0.0,
            ],
            dtype=np.float64,
        )

        # Optional target point offset in the board frame. This lets the
        # controller follow a specific pixel block / ArUco marker instead of
        # the board centre. The offset is rotated into the camera frame below.
        board_target_object = board_center_object + np.array(
            [TARGET_POINT_OFFSET_X_M, TARGET_POINT_OFFSET_Y_M, 0.0],
            dtype=np.float64,
        )

        center_camera = (
            rotation_matrix @ board_target_object
            + tvec.reshape(3)
        )

        normal_camera = normalize_vector(rotation_matrix[:, 2])
        if normal_camera is None:
            self.board_pose_valid = False
            return

        self._finalize_pose_state(center_camera, normal_camera)

    def _estimate_single_marker_pose(self, marker_corners):
        """Estimate the pose of one ArUco marker (4 corners)."""
        if marker_corners is None:
            self.board_pose_valid = False
            return

        half = ARUCO_MARKER_LENGTH_M / 2.0
        object_points = np.array(
            [
                [-half, -half, 0.0],
                [half, -half, 0.0],
                [half, half, 0.0],
                [-half, half, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = marker_corners.reshape(4, 2).astype(np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.board_pose_valid = False
            return

        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Marker centre in the camera frame, plus the optional board-frame
        # target offset rotated into the camera frame.
        center_camera = tvec.reshape(3).astype(np.float64)
        if TARGET_POINT_OFFSET_X_M != 0.0 or TARGET_POINT_OFFSET_Y_M != 0.0:
            offset = np.array(
                [TARGET_POINT_OFFSET_X_M, TARGET_POINT_OFFSET_Y_M, 0.0],
                dtype=np.float64,
            )
            center_camera = center_camera + rotation_matrix @ offset

        normal_camera = normalize_vector(rotation_matrix[:, 2])
        if normal_camera is None:
            self.board_pose_valid = False
            return

        self._finalize_pose_state(center_camera, normal_camera)

    def _finalize_pose_state(self, center_camera, normal_camera):
        """Shared filtering / velocity / output update for either target type."""
        if float(center_camera[2]) <= 0.05:
            self.board_pose_valid = False
            return

        # Choose the normal that points from the board/marker toward the camera.
        board_to_camera = -center_camera
        if float(np.dot(normal_camera, board_to_camera)) < 0.0:
            normal_camera = -normal_camera

        now = time.time()

        if (
            self.target_x_m is not None
            and self.board_normal_x is not None
        ):
            pose_alpha = POSE_FILTER_ALPHA
            normal_alpha = NORMAL_FILTER_ALPHA

            filtered_center = (
                pose_alpha * center_camera
                + (1.0 - pose_alpha)
                * np.array(
                    [
                        self.target_x_m,
                        self.target_y_m,
                        self.target_z_m,
                    ],
                    dtype=np.float64,
                )
            )

            filtered_normal = normalize_vector(
                normal_alpha * normal_camera
                + (1.0 - normal_alpha)
                * np.array(
                    [
                        self.board_normal_x,
                        self.board_normal_y,
                        self.board_normal_z,
                    ],
                    dtype=np.float64,
                )
            )
            if filtered_normal is not None:
                normal_camera = filtered_normal
            center_camera = filtered_center

        if (
            self.previous_center is not None
            and self.previous_pose_time is not None
        ):
            dt = now - self.previous_pose_time
            if 0.03 <= dt <= 0.50:
                raw_velocity = (
                    center_camera - self.previous_center
                ) / dt
                raw_velocity = np.clip(
                    raw_velocity,
                    -MAX_PREDICT_SPEED_M_S,
                    MAX_PREDICT_SPEED_M_S,
                )

                velocity_alpha = TARGET_VELOCITY_FILTER_ALPHA
                old_velocity = np.array(
                    [
                        self.target_vx_m_s,
                        self.target_vy_m_s,
                        self.target_vz_m_s,
                    ],
                    dtype=np.float64,
                )
                filtered_velocity = (
                    velocity_alpha * raw_velocity
                    + (1.0 - velocity_alpha) * old_velocity
                )
                (
                    self.target_vx_m_s,
                    self.target_vy_m_s,
                    self.target_vz_m_s,
                ) = map(float, filtered_velocity)

        self.previous_center = center_camera.copy()
        self.previous_pose_time = now

        x_m, y_m, z_m = map(float, center_camera)
        distance_m = float(np.linalg.norm(center_camera))
        bearing_deg = math.degrees(math.atan2(x_m, z_m))

        normal_range_m = float(
            np.dot(-center_camera, normal_camera)
        )
        current_camera_from_board = -center_camera
        cross_track = (
            current_camera_from_board
            - normal_range_m * normal_camera
        )
        lateral_error_m = float(
            math.hypot(cross_track[0], cross_track[2])
        )

        face_error_deg = angle_between_vectors_deg(
            normal_camera,
            current_camera_from_board,
        )

        self.target_x_m = x_m
        self.target_y_m = y_m
        self.target_z_m = z_m
        self.target_distance_m = distance_m
        self.target_forward_range_m = z_m
        self.target_bearing_deg = bearing_deg

        self.board_normal_x = float(normal_camera[0])
        self.board_normal_y = float(normal_camera[1])
        self.board_normal_z = float(normal_camera[2])
        self.target_face_error_deg = face_error_deg
        self.target_lateral_error_m = lateral_error_m
        self.target_normal_range_m = normal_range_m

        self.last_pose_time = now
        self.board_pose_valid = True


def reset_detector_tracking_state(detector):
    detector.target_found = False
    detector.board_pose_valid = False

    detector.target_distance_m = None
    detector.target_forward_range_m = None
    detector.target_bearing_deg = None
    detector.target_x_m = None
    detector.target_y_m = None
    detector.target_z_m = None

    detector.board_normal_x = None
    detector.board_normal_y = None
    detector.board_normal_z = None
    detector.target_face_error_deg = None
    detector.target_lateral_error_m = None
    detector.target_normal_range_m = None

    detector.target_vx_m_s = 0.0
    detector.target_vy_m_s = 0.0
    detector.target_vz_m_s = 0.0
    detector.previous_center = None
    detector.previous_pose_time = None

    detector.last_seen_time = 0.0
    detector.last_pose_time = 0.0
    detector.lost_frame_count = 0

    detector.area = 0.0
    detector.error_x = 0.0
    detector.error_y = 0.0

    detector.bbox_x = 0
    detector.bbox_y = 0
    detector.bbox_w = 0
    detector.bbox_h = 0
    detector.target_cx = 0.0
    detector.target_cy = 0.0


async def spin_ros_node(detector):
    while rclpy.ok():
        rclpy.spin_once(detector, timeout_sec=0.01)
        await asyncio.sleep(0.001)

async def wait_altitude_fast(drone, target=5.0, tol=0.3, timeout=25.0):
    """等待起飞高度；超时后提示并继续。"""
    t0 = time.time()

    while True:
        try:
            p = await asyncio.wait_for(
                drone.telemetry.position().__anext__(),
                timeout=1.5
            )
        except asyncio.TimeoutError:
            if time.time() - t0 > timeout:
                print(f"等待高度超时（{timeout:.0f}s），暂无位置数据，继续流程。")
                return False
            continue

        print(f"当前高度: {p.relative_altitude_m:.1f} m")

        if p.relative_altitude_m >= target - tol:
            print(f"已悬停于 {p.relative_altitude_m:.1f} m，等待指令")
            return True

        if time.time() - t0 > timeout:
            print(
                f"等待高度超时（{timeout:.0f}s），"
                f"当前 {p.relative_altitude_m:.1f} m，继续流程。"
            )
            return False

        await asyncio.sleep(0.5)

async def connect_and_takeoff(drone):
    print("连接无人机...")
    await drone.connect(system_address=UDP_ADDR)

    print("等待 PX4 连接...")
    async for st in drone.core.connection_state():
        if st.is_connected:
            print("PX4 已连接")
            break

    print("等待 local position / armable 就绪...")
    async for h in drone.telemetry.health():
        if h.is_local_position_ok and h.is_armable:
            print("Local position / armable 就绪")
            break

    await drone.action.set_takeoff_altitude(TAKEOFF_ALT_M)

    answer = input("确认起飞请输入 YES：")
    if answer.strip().upper() != "YES":
        print("未确认起飞，程序结束")
        return False

    print("解锁...")
    await drone.action.arm()

    print("起飞...")
    await drone.action.takeoff()

    print(f"等待起飞到 {TAKEOFF_ALT_M:.1f} m 左右...")
    await wait_altitude_fast(
        drone,
        target=TAKEOFF_ALT_M,
        tol=0.3,
        timeout=25.0
    )

    return True


async def start_offboard_velocity(drone):
    print("准备进入 Offboard velocity 模式...")

    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
    )

    try:
        await drone.offboard.start()
        print("已进入 Offboard 模式")
        return True
    except OffboardError as e:
        print(f"Offboard start 失败: {e._result.result}")
        await safe_stop_and_land(drone, reason="Offboard启动失败")
        return False
    except Exception as e:
        print(f"Offboard start 异常: {e}")
        await safe_stop_and_land(drone, reason="Offboard启动异常")
        return False


async def command_listener(stop_event, land_event):
    print("\n飞行中可输入：land / stop / 0 触发安全降落")

    while not stop_event.is_set():
        ready, _, _ = select.select([sys.stdin], [], [], 0.1)

        if ready:
            line = sys.stdin.readline()
            if line == "":
                continue

            cmd = line.strip().lower()

            if cmd in ("land", "stop", "0"):
                print("[CMD] 收到降落/停止指令")
                land_event.set()
                stop_event.set()
                return

            if cmd:
                print("飞行中仅支持：land / stop / 0")

        await asyncio.sleep(0.05)


async def ask_start_search(stop_event, land_event):
    print("\n[READY] 已起飞并悬停。")
    print("现在不会马上开始搜索。")
    print("确认开始 search ChArUco/ArUco 目标请输入 START")
    print("不想开始请输入 land")

    answer = await read_console_line("输入：", stop_event=stop_event)

    if answer is None:
        land_event.set()
        return False

    cmd = answer.strip().lower()

    if cmd in ("land", "stop", "0"):
        land_event.set()
        return False

    if answer.strip().upper() != "START":
        print("未输入 START，准备降落")
        land_event.set()
        return False

    return True


async def search_until_target_found(drone, detector, stop_event, land_event):
    print("\n[SEARCH] 开始搜索 ChArUco/ArUco 目标")
    print(f"[SEARCH] 最多搜索 {SEARCH_TIMEOUT_S:.1f} s")
    print("[SEARCH] 搜索阶段会原地慢速 yaw 旋转")

    # Clear stale target state from previous tracking/search.
    # Otherwise a target seen shortly before this START can make search return
    # FOUND even if the current view is not stable.
    reset_detector_tracking_state(detector)

    search_start = time.time()
    last_print = 0.0
    dt = 1.0 / CONTROL_HZ

    search_confirm_count = 0

    while not stop_event.is_set():
        now = time.time()

        if land_event.is_set():
            print("[SEARCH] 用户要求降落")
            return False

        recently_seen = (
            detector.target_found and
            now - detector.last_seen_time < TARGET_LOST_HOVER_S
        )

        if recently_seen:
            search_confirm_count += 1
        else:
            search_confirm_count = 0

        if search_confirm_count >= SEARCH_CONFIRM_MIN_FRAMES:
            print("[SEARCH] 已找到 ChArUco/ArUco 目标")

            dist_text = "None"
            if detector.target_distance_m is not None:
                dist_text = f"{detector.target_distance_m:.2f}m"

            print(
                f"[FOUND] area={detector.area:.0f}, "
                f"dist={dist_text}, "
                f"bbox=({detector.bbox_x},{detector.bbox_y},"
                f"{detector.bbox_w},{detector.bbox_h}), "
                f"center=({detector.target_cx:.1f},{detector.target_cy:.1f}), "
                f"image=({detector.image_w}x{detector.image_h}), "
                f"error_x={detector.error_x:+.1f}, "
                f"error_y={detector.error_y:+.1f}, "
                f"confirm_frames={search_confirm_count}"
            )

            # 找到目标后先停住，等待用户输入飞行时长
            for _ in range(10):
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                await asyncio.sleep(0.1)

            return True

        if now - search_start > SEARCH_TIMEOUT_S:
            print("[SEARCH] 搜索超时，未找到 ChArUco/ArUco 目标，进入悬停等待指令")
            return False

        if now - last_print > 1.0:
            last_print = now
            print(
                f"[SEARCH] target not found → 原地 yaw search "
                f"confirm={search_confirm_count}/{SEARCH_CONFIRM_MIN_FRAMES}"
            )

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                0.0,
                0.0,
                0.0,
                SEARCH_YAW_RATE_DEG_S
            )
        )

        await asyncio.sleep(dt)

    return False


async def ask_track_duration(stop_event, land_event):
    print("\n[TARGET FOUND] 已找到目标。")
    print("现在请输入 单目 ChArUco 跟踪飞行时长，单位秒。")
    print("例如输入 20 表示跟踪 20 秒。")
    print("如果不想继续跟踪，输入 land。")

    while not stop_event.is_set():
        answer = await read_console_line("飞行时长 / land：", stop_event=stop_event)

        if answer is None:
            land_event.set()
            return None

        cmd = answer.strip().lower()

        if cmd in ("land", "stop", "0"):
            land_event.set()
            return None

        try:
            duration_s = float(cmd)
            if duration_s <= 0:
                print("时长必须大于 0")
                continue

            return duration_s

        except ValueError:
            print("请输入数字，例如 20；或者输入 land")


async def visual_orbit_control(
    drone,
    detector,
    track_duration_s,
    desired_distance_m,
    stop_event,
    land_event,
):
    print("\n[TRACK-BOARD-NORMAL] 开始整板 ChArUco tracking")
    print("[TRACK-BOARD-NORMAL] ALIGN 后移动到板法向正前方指定距离")
    print("[TRACK-BOARD-NORMAL] right_speed 仅用于法向观察点纠偏，不再绕飞")

    dt = 1.0 / CONTROL_HZ
    track_start = time.time()
    last_print = 0.0

    aligned = False
    align_count = 0
    arrival_count = 0
    urgent_count = 0

    last_forward = 0.0
    last_right = 0.0
    last_yaw = 0.0

    while not stop_event.is_set():
        now = time.time()

        if land_event.is_set():
            return "land"

        if now - track_start >= track_duration_s:
            return "finished"

        marker_visible = (
            detector.target_found
            and now - detector.last_seen_time < TARGET_LOST_HOVER_S
        )
        pose_valid = (
            marker_visible
            and detector.board_pose_valid
            and detector.target_x_m is not None
            and detector.target_z_m is not None
            and detector.board_normal_x is not None
            and detector.board_normal_z is not None
            and detector.target_face_error_deg is not None
            and now - detector.last_pose_time
            < CAMERA_INFO_TIMEOUT_S
        )

        if not pose_valid:
            lost_time = now - detector.last_seen_time

            target_forward = 0.0
            target_right = 0.0

            # If markers are visible but whole-board pose is temporarily missing,
            # centre the visible board gently using the original pixel centre.
            if marker_visible and detector.image_w > 0:
                half_width = detector.image_w / 2.0
                pixel_ratio = detector.error_x / max(half_width, 1.0)
                target_yaw = clamp(
                    10.0 * pixel_ratio,
                    -MAX_TRACK_YAW_RATE_DEG_S,
                    MAX_TRACK_YAW_RATE_DEG_S,
                )
                mode = "WAIT_BOARD_POSE"
            else:
                target_yaw = (
                    0.0
                    if lost_time < POSE_LOST_HOVER_S
                    else LOST_YAW_SEARCH_RATE_DEG_S
                )
                mode = "POSE_LOST"

            forward = slew_limit(
                target_forward,
                last_forward,
                FORWARD_SLEW_M_S2,
                dt,
            )
            right = slew_limit(
                target_right,
                last_right,
                LATERAL_SLEW_M_S2,
                dt,
            )
            yaw = slew_limit(
                target_yaw,
                last_yaw,
                YAW_SLEW_DEG_S2,
                dt,
            )

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    forward,
                    right,
                    0.0,
                    yaw,
                )
            )

            last_forward = forward
            last_right = right
            last_yaw = yaw
            aligned = False
            align_count = 0
            arrival_count = 0
            urgent_count = 0

            if lost_time > TARGET_LOST_LAND_S:
                return "lost_timeout"

            if now - last_print > 0.5:
                last_print = now
                print(
                    f"[{mode}] markers={marker_visible} "
                    f"F={forward:+.2f} R={right:+.2f} Y={yaw:+.1f}"
                )

            await asyncio.sleep(dt)
            continue

        center = np.array(
            [
                detector.target_x_m,
                detector.target_y_m,
                detector.target_z_m,
            ],
            dtype=np.float64,
        )
        normal = normalize_vector(
            [
                detector.board_normal_x,
                detector.board_normal_y,
                detector.board_normal_z,
            ]
        )
        velocity = np.array(
            [
                detector.target_vx_m_s,
                detector.target_vy_m_s,
                detector.target_vz_m_s,
            ],
            dtype=np.float64,
        )

        if normal is None:
            await asyncio.sleep(dt)
            continue

        # Predict the board centre 0.2 s ahead.
        predicted_center = (
            center
            + TARGET_PREDICTION_TIME_S * velocity
        )

        predicted_x = float(predicted_center[0])
        predicted_z = float(predicted_center[2])
        predicted_bearing_deg = math.degrees(
            math.atan2(predicted_x, predicted_z)
        )

        # Desired camera position relative to the board is:
        # desired_distance * normal, where normal points board -> camera.
        # Current camera position relative to the board is -predicted_center.
        # Therefore desired camera displacement from the current position is:
        # desired_distance * normal + predicted_center.
        observation_point_error = (
            predicted_center
            + desired_distance_m * normal
        )

        forward_error_m = float(observation_point_error[2])
        lateral_error_m = float(observation_point_error[0])

        normal_range_m = float(
            np.dot(-predicted_center, normal)
        )
        normal_distance_error_m = (
            normal_range_m - desired_distance_m
        )

        current_camera_from_board = -predicted_center
        face_error_deg = angle_between_vectors_deg(
            normal,
            current_camera_from_board,
        )
        if face_error_deg is None:
            face_error_deg = 180.0

        if abs(predicted_bearing_deg) <= BEARING_DEADBAND_DEG:
            target_yaw = 0.0
        else:
            target_yaw = clamp(
                KP_BEARING_YAW * predicted_bearing_deg,
                -MAX_TRACK_YAW_RATE_DEG_S,
                MAX_TRACK_YAW_RATE_DEG_S,
            )

        board_distance = max(
            float(np.linalg.norm(predicted_center)),
            1e-3,
        )
        radial_closing_speed = float(
            np.dot(predicted_center, velocity)
            / board_distance
        )

        if radial_closing_speed < -URGENT_CLOSING_SPEED_M_S:
            urgent_count += 1
        else:
            urgent_count = max(0, urgent_count - 1)

        urgent = (
            normal_distance_error_m
            < -URGENT_TOO_CLOSE_MARGIN_M
            or urgent_count >= URGENT_CONFIRM_FRAMES
        )

        if urgent:
            retreat_speed = clamp(
                URGENT_RETREAT_KP
                * max(-normal_distance_error_m, 0.0)
                + VELOCITY_FEEDFORWARD_GAIN
                * max(-radial_closing_speed, 0.0),
                URGENT_RETREAT_MIN_SPEED,
                URGENT_RETREAT_MAX_SPEED,
            )

            horizontal_range = max(
                math.hypot(predicted_x, predicted_z),
                1e-3,
            )
            target_forward = (
                -retreat_speed
                * predicted_z
                / horizontal_range
            )
            target_right = (
                -retreat_speed
                * predicted_x
                / horizontal_range
            )
            target_right = clamp(
                target_right,
                -MAX_TRACK_LATERAL_SPEED,
                MAX_TRACK_LATERAL_SPEED,
            )

            mode = "URGENT_RETREAT"
            aligned = False
            align_count = 0
            arrival_count = 0

        else:
            # Turn first. No translation is allowed until the board centre has
            # been aligned for several consecutive frames.
            if (
                aligned
                and abs(predicted_bearing_deg)
                > REALIGN_BEARING_DEG
            ):
                aligned = False
                align_count = 0

            if not aligned:
                target_forward = 0.0
                target_right = 0.0
                mode = "ALIGN"

                if (
                    abs(predicted_bearing_deg)
                    <= ALIGN_BEARING_DEG
                ):
                    align_count += 1
                else:
                    align_count = 0

                if align_count >= ALIGN_CONFIRM_FRAMES:
                    aligned = True
                    align_count = 0
                    mode = "ALIGN_DONE"

            else:
                target_forward = clamp(
                    KP_OBSERVATION_POINT * forward_error_m
                    + VELOCITY_FEEDFORWARD_GAIN
                    * detector.target_vz_m_s,
                    -MAX_TRACK_BACKWARD_SPEED,
                    MAX_TRACK_FORWARD_SPEED,
                )
                target_right = clamp(
                    KP_OBSERVATION_POINT * lateral_error_m
                    + VELOCITY_FEEDFORWARD_GAIN
                    * detector.target_vx_m_s,
                    -MAX_TRACK_LATERAL_SPEED,
                    MAX_TRACK_LATERAL_SPEED,
                )

                distance_ok = (
                    abs(normal_distance_error_m)
                    < DISTANCE_TOLERANCE_M
                )
                center_ok = (
                    abs(predicted_bearing_deg)
                    < CENTER_TOLERANCE_DEG
                )
                face_ok = (
                    face_error_deg
                    < FACE_TOLERANCE_DEG
                )
                lateral_ok = (
                    abs(lateral_error_m)
                    < LATERAL_TOLERANCE_M
                )

                if (
                    distance_ok
                    and center_ok
                    and face_ok
                    and lateral_ok
                ):
                    arrival_count += 1
                else:
                    arrival_count = 0

                if arrival_count >= ARRIVAL_CONFIRM_FRAMES:
                    target_forward = 0.0
                    target_right = 0.0
                    mode = "HOLD"
                else:
                    mode = "TRACK_NORMAL_POINT"

        forward = slew_limit(
            target_forward,
            last_forward,
            FORWARD_SLEW_M_S2,
            dt,
        )
        right = slew_limit(
            target_right,
            last_right,
            LATERAL_SLEW_M_S2,
            dt,
        )
        yaw = slew_limit(
            target_yaw,
            last_yaw,
            YAW_SLEW_DEG_S2,
            dt,
        )

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward,
                right,
                0.0,
                yaw,
            )
        )

        last_forward = forward
        last_right = right
        last_yaw = yaw

        if now - last_print > 0.5:
            last_print = now
            print(
                f"[{mode}] "
                f"normal_range={normal_range_m:.2f}m "
                f"distance_err={normal_distance_error_m:+.2f}m "
                f"bearing={predicted_bearing_deg:+.1f}deg "
                f"face_err={face_error_deg:.1f}deg "
                f"obs_F_err={forward_error_m:+.2f}m "
                f"obs_R_err={lateral_error_m:+.2f}m "
                f"vx={detector.target_vx_m_s:+.2f} "
                f"vz={detector.target_vz_m_s:+.2f} "
                f"F={forward:+.2f} "
                f"R={right:+.2f} "
                f"Y={yaw:+.1f}"
            )

        await asyncio.sleep(dt)

    return "stopped"


async def visual_kf_tracking(
    drone,
    detector,
    track_duration_s,
    desired_distance_m,
    stop_event,
    land_event,
    planner,
    drone_position_provider,
):
    """CV-KF path-planning tracking.

    Direction (yaw) and distance (forward) are controlled independently:
      - KF estimates target N/E position + velocity from the ChArUco 3D pose
      - yaw points at the predicted target point (position + velocity * T)
      - forward is a hard P controller on the measured distance error
      - KF velocity is fed forward into body forward/right so the drone keeps
        pace with a moving vehicle (including lateral / circling motion)
      - distance safety (URGENT retreat) always overrides tracking
    """
    print("\n[KF-TRACK] 开始 CV-KF 路径规划跟踪")
    print(f"[KF-TRACK] 期望距离={desired_distance_m:.2f}m")

    dt = 1.0 / CONTROL_HZ
    track_start = time.time()
    last_print = 0.0

    last_forward = 0.0
    last_right = 0.0
    last_yaw = 0.0

    lost_start = None
    kf_frame_count = 0
    last_kf_time = None

    while not stop_event.is_set():
        now = time.time()

        if land_event.is_set():
            return "land"

        if now - track_start >= track_duration_s:
            return "finished"

        marker_visible = (
            detector.target_found
            and now - detector.last_seen_time < TARGET_LOST_HOVER_S
        )
        pose_valid = (
            marker_visible
            and detector.board_pose_valid
            and detector.target_x_m is not None
            and detector.target_z_m is not None
            and detector.target_distance_m is not None
            and now - detector.last_pose_time < CAMERA_INFO_TIMEOUT_S
        )

        if not pose_valid:
            if lost_start is None:
                lost_start = now
            lost_time = now - lost_start
            kf_frame_count = 0
            planner.reset()

            target_forward = 0.0
            target_right = 0.0

            if lost_time < POSE_LOST_HOVER_S:
                target_yaw = 0.0
                mode = "KF_LOST_HOVER"
            elif lost_time < TARGET_LOST_LAND_S:
                target_yaw = LOST_YAW_SEARCH_RATE_DEG_S
                mode = "KF_LOST_SEARCH"
            else:
                print("[KF-TRACK] 目标丢失超时")
                return "lost_timeout"

            forward = slew_limit(
                target_forward, last_forward, FORWARD_SLEW_M_S2, dt
            )
            right = slew_limit(
                target_right, last_right, LATERAL_SLEW_M_S2, dt
            )
            yaw = slew_limit(target_yaw, last_yaw, YAW_SLEW_DEG_S2, dt)

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, 0.0, yaw)
            )
            last_forward = forward
            last_right = right
            last_yaw = yaw

            if now - last_print > 0.5:
                last_print = now
                print(
                    f"[{mode}] markers={marker_visible} "
                    f"lost={lost_time:.1f}s F={forward:+.2f} "
                    f"R={right:+.2f} Y={yaw:+.1f}"
                )

            await asyncio.sleep(dt)
            continue

        lost_start = None

        try:
            pos_data = drone_position_provider()
            if pos_data is None:
                forward = slew_limit(
                    0.0, last_forward, FORWARD_SLEW_M_S2, dt
                )
                right = slew_limit(
                    0.0, last_right, LATERAL_SLEW_M_S2, dt
                )
                yaw = slew_limit(0.0, last_yaw, YAW_SLEW_DEG_S2, dt)
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(forward, right, 0.0, yaw)
                )
                last_forward = forward
                last_right = right
                last_yaw = yaw
                await asyncio.sleep(dt)
                continue
            drone_n, drone_e, heading_deg = pos_data
        except Exception as e:
            if now - last_print > 1.0:
                last_print = now
                print(f"[KF-TRACK] 位置获取失败: {e}")
            forward = slew_limit(
                0.0, last_forward, FORWARD_SLEW_M_S2, dt
            )
            right = slew_limit(
                0.0, last_right, LATERAL_SLEW_M_S2, dt
            )
            yaw = slew_limit(0.0, last_yaw, YAW_SLEW_DEG_S2, dt)
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, right, 0.0, yaw)
            )
            last_forward = forward
            last_right = right
            last_yaw = yaw
            await asyncio.sleep(dt)
            continue

        # ChArUco pose is in the camera frame: x right, y down, z forward.
        # Body frame: forward = z, right = x.
        forward_body = detector.target_z_m
        right_body = detector.target_x_m

        target_n, target_e = body_to_local_ne(
            drone_n,
            drone_e,
            heading_deg,
            forward_body,
            right_body,
        )

        # Warm-up: use the raw ChArUco observation for the first N frames while
        # the KF converges, then switch to the filtered state.
        if not planner.initialized:
            kf_frame_count = 0
            planner.initialize(target_n, target_e)
            use_n = target_n
            use_e = target_e
            use_vn = 0.0
            use_ve = 0.0
            last_kf_time = now
        else:
            kf_frame_count += 1
            if last_kf_time is None:
                kf_dt = dt
            else:
                kf_dt = min(max(now - last_kf_time, 0.01), KF_MAX_DT_S)
            last_kf_time = now
            planner.predict(kf_dt)
            planner.update(target_n, target_e)

        if kf_frame_count <= KF_WARMUP_FRAMES:
            use_n = target_n
            use_e = target_e
            use_vn = 0.0
            use_ve = 0.0
        else:
            use_n = planner.pos_n
            use_e = planner.pos_e
            use_vn = clamp(
                planner.vel_n, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S
            )
            use_ve = clamp(
                planner.vel_e, -MAX_PREDICT_SPEED_M_S, MAX_PREDICT_SPEED_M_S
            )

        # Predicted target point, N/E.
        predict_n = use_n + use_vn * PLANNER_PREDICT_TIME_S
        predict_e = use_e + use_ve * PLANNER_PREDICT_TIME_S

        err_n = predict_n - drone_n
        err_e = predict_e - drone_e
        target_heading_deg = math.degrees(math.atan2(err_e, err_n))
        heading_err = angle_diff_deg(target_heading_deg, heading_deg)

        target_yaw = clamp(
            PLANNER_KP_YAW * heading_err,
            -MAX_YAW_RATE_DEG_S,
            MAX_YAW_RATE_DEG_S,
        )

        distance_m = detector.target_distance_m
        distance_err = distance_m - desired_distance_m

        if abs(distance_err) > PLANNER_DEADBAND_M:
            target_forward = clamp(
                PLANNER_KP_DIST * distance_err,
                -PLANNER_MAX_RETREAT,
                PLANNER_MAX_FORWARD,
            )
        else:
            target_forward = 0.0

        # Velocity feedforward into the body frame so the drone keeps up with
        # the moving vehicle (forward and lateral / circling motion).
        yaw_rad = math.radians(heading_deg)
        ff_forward = (
            use_vn * math.cos(yaw_rad)
            + use_ve * math.sin(yaw_rad)
        )
        ff_right = (
            -use_vn * math.sin(yaw_rad)
            + use_ve * math.cos(yaw_rad)
        )

        # Distance safety always overrides tracking.
        if distance_m < PLANNER_SAFE_DISTANCE_M:
            retreat_speed = clamp(
                URGENT_RETREAT_KP
                * (PLANNER_SAFE_DISTANCE_M - distance_m),
                URGENT_RETREAT_MIN_SPEED,
                URGENT_RETREAT_MAX_SPEED,
            )
            target_forward = -retreat_speed
            target_right = PLANNER_FF_GAIN * ff_right
            mode = "KF_URGENT_RETREAT"
        else:
            target_forward += PLANNER_FF_GAIN * ff_forward
            target_right = PLANNER_FF_GAIN * ff_right
            mode = "KF_TRACK"

        forward = slew_limit(
            target_forward, last_forward, FORWARD_SLEW_M_S2, dt
        )
        right = slew_limit(
            target_right, last_right, LATERAL_SLEW_M_S2, dt
        )
        yaw = slew_limit(target_yaw, last_yaw, YAW_SLEW_DEG_S2, dt)

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(forward, right, 0.0, yaw)
        )
        last_forward = forward
        last_right = right
        last_yaw = yaw

        if now - last_print > 0.5:
            last_print = now
            print(
                f"[{mode}] d={distance_m:.2f}m "
                f"d_err={distance_err:+.2f}m "
                f"h_err={heading_err:+.1f}deg "
                f"vn={use_vn:+.2f} ve={use_ve:+.2f} "
                f"F={forward:+.2f} R={right:+.2f} Y={yaw:+.1f}"
            )

        await asyncio.sleep(dt)

    return "stopped"


async def main_async():
    rclpy.init()

    detector = CharucoTargetDetector()
    drone = System()

    planner = CVKalmanFilter() if PLANNER_ENABLED else None

    stop_event = asyncio.Event()
    land_event = asyncio.Event()

    loop = asyncio.get_running_loop()

    try:
        loop.add_signal_handler(signal.SIGINT, stop_event.set)
    except NotImplementedError:
        pass

    ros_task = asyncio.create_task(spin_ros_node(detector))
    cmd_task = None
    flight_started = False

    try:
        ok = await connect_and_takeoff(drone)
        if not ok:
            return

        flight_started = True

        # Start Offboard once, then keep zero-velocity setpoints while waiting
        # for START / home / land commands. This avoids ending the program after
        # a single tracking task.
        offboard_ok = await start_offboard_velocity(drone)
        if not offboard_ok:
            return

        # Record local-coordinate reference after takeoff. This is the same idea
        # as the original uavG home command: north=0, east=0, down=-takeoff_alt.
        pos = await drone.telemetry.position().__anext__()
        ref_lat, ref_lon = pos.latitude_deg, pos.longitude_deg
        print(
            f"[HOME参考点] lat={pos.latitude_deg:.7f} lon={pos.longitude_deg:.7f} "
            f"abs_alt={pos.absolute_altitude_m:.2f}m rel_alt={pos.relative_altitude_m:.2f}m"
        )

        # Asynchronous N/E/heading cache for the KF planner. Each synchronous
        # read triggers one async refresh and returns the previous estimate.
        _pos_cache = {
            "n": 0.0, "e": 0.0, "h": 0.0,
            "pending": False, "ready": False,
        }

        async def _update_pos_cache():
            try:
                pos = await drone.telemetry.position().__anext__()
                heading = await drone.telemetry.heading().__anext__()
                dlat = pos.latitude_deg - ref_lat
                dlon = pos.longitude_deg - ref_lon
                _pos_cache["n"] = dlat * 111320.0
                _pos_cache["e"] = dlon * 111320.0 * math.cos(math.radians(ref_lat))
                _pos_cache["h"] = heading.heading_deg
                _pos_cache["ready"] = True
            except Exception:
                pass
            finally:
                _pos_cache["pending"] = False

        def drone_position_provider():
            if not _pos_cache["pending"]:
                _pos_cache["pending"] = True
                asyncio.ensure_future(_update_pos_cache())
            if not _pos_cache["ready"]:
                return None
            return _pos_cache["n"], _pos_cache["e"], _pos_cache["h"]

        while not stop_event.is_set() and not land_event.is_set():
            action = await hover_wait_for_command(
                drone,
                stop_event,
                land_event,
                ref_lat,
                ref_lon,
            )

            if action != "start":
                break

            # Search stage: allow land/stop/0 while the aircraft is actively yaw-searching.
            cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

            found = await search_until_target_found(
                drone,
                detector,
                stop_event,
                land_event
            )

            if cmd_task is not None:
                cmd_task.cancel()
                try:
                    await cmd_task
                except asyncio.CancelledError:
                    pass
                cmd_task = None

            if land_event.is_set() or stop_event.is_set():
                break

            if not found:
                print("[WAIT] 本次 search 没有进入 tracking，返回悬停等待。")
                continue

            # Target found: ask task parameters. No command_listener here, because
            # it would compete with these interactive inputs.
            track_duration_s = await ask_track_duration(stop_event, land_event)
            if track_duration_s is None:
                if land_event.is_set():
                    break
                continue

            desired_distance_m = await ask_float(
                "请输入板法向正前方期望距离 desired_distance_m，单位 m，例如 7.5",
                default_value=DEFAULT_DESIRED_DISTANCE_M,
                min_value=1.0,
                max_value=MAX_VALID_MONO_DISTANCE_M,
                stop_event=stop_event
            )

            if desired_distance_m is None:
                if land_event.is_set():
                    break
                continue

            # Tracking stage: allow land/stop/0 while actively tracking.
            cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

            if planner is not None:
                planner.reset()
                track_result = await visual_kf_tracking(
                    drone,
                    detector,
                    track_duration_s,
                    desired_distance_m,
                    stop_event,
                    land_event,
                    planner,
                    drone_position_provider,
                )
            else:
                track_result = await visual_orbit_control(
                    drone,
                    detector,
                    track_duration_s,
                    desired_distance_m,
                    stop_event,
                    land_event,
                )

            if cmd_task is not None:
                cmd_task.cancel()
                try:
                    await cmd_task
                except asyncio.CancelledError:
                    pass
                cmd_task = None

            if land_event.is_set() or stop_event.is_set():
                break

            print(f"[WAIT] tracking 结束原因: {track_result}，返回悬停等待。")

    except Exception as e:
        print(f"[ERROR] 发生异常: {e}")
        land_event.set()

    finally:
        stop_event.set()

        if cmd_task is not None:
            cmd_task.cancel()
            try:
                await cmd_task
            except asyncio.CancelledError:
                pass

        if flight_started:
            await safe_stop_and_land(drone, reason="程序结束/异常/用户停止")

        ros_task.cancel()
        try:
            await ros_task
        except asyncio.CancelledError:
            pass

        detector.destroy_node()
        rclpy.shutdown()

        print("程序已安全退出")

def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()


