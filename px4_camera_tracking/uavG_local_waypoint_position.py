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
from sensor_msgs.msg import Image

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed, PositionNedYaw


# ================= 基本连接参数 =================
UDP_ADDR = "udpin://0.0.0.0:14540"

IMAGE_TOPIC = "/camera/image_raw"
TAKEOFF_ALT_M = 5.0
DEPTH_TOPIC = "/camera/depth/image_raw"

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
MAX_BACKWARD_SPEED = 0.80
ORBIT_RIGHT_SPEED = 0.35
MAX_YAW_RATE_DEG_S = 75.0
CENTER_ERROR_LIMIT = 60.0

KP_YAW_NORM = 12.0
MAX_TRACK_YAW_RATE_DEG_S = 15.0
YAW_DEADBAND_PX = 90.0
YAW_SLEW_RATE_DEG_S2 = 25.0
# ================= Depth distance control =================
# RGB is used to find the ArUco / ChArUco target.
# Depth is used to control forward/backward distance when a valid distance exists.
USE_DEPTH_DISTANCE_CONTROL = True
DEFAULT_DESIRED_DISTANCE_M = 5.0
DISTANCE_DEADBAND_M = 0.4
KP_DISTANCE = 0.18
MAX_VALID_DEPTH_M = 15.0
DEPTH_TIMEOUT_S = 0.8

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
USE_CHARUCO_TARGET = True
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_100
TARGET_MARKER_ID = None
MIN_MARKER_AREA = 150.0
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

# Keep B position tracking stable during short visual/depth dropouts.
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
        self.error_x = 0.0
        self.error_y = 0.0
        self.area = 0.0
        self.last_seen_time = 0.0
        self.frame_count = 0

        # Debug / tracking information for the selected ArUco / ChArUco target.
        self.bbox_x = 0
        self.bbox_y = 0
        self.bbox_w = 0
        self.bbox_h = 0
        self.target_cx = 0.0
        self.target_cy = 0.0
        self.image_w = 0
        self.image_h = 0
        self.target_distance_m = None
        self.last_depth_time = 0.0

        # Previous-frame tracking state.
        # If we saw a target in the previous frames, choose the new contour
        # closest to that previous target rather than blindly choosing
        # the largest red contour in the whole image.
        self.has_previous_target = False
        self.prev_target_cx = 0.0
        self.prev_target_cy = 0.0
        self.lost_frame_count = 0
        self.max_lost_frames = 10
        self.previous_target_distance_weight = 5.0
        self.last_aruco_debug_time = 0.0

        # ArUco / ChArUco detector
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

        self.get_logger().info(
            f"Using ArUco/ChArUco detector, dictionary={ARUCO_DICTIONARY}, "
            f"min_marker_area={MIN_MARKER_AREA}"
        )

        self.sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            10
        )

        self.depth_sub = self.create_subscription(
            Image,
            DEPTH_TOPIC,
            self.depth_callback,
            10
        )

        self.get_logger().info(f"Subscribing to RGB: {IMAGE_TOPIC}")
        self.get_logger().info(f"Subscribing to depth: {DEPTH_TOPIC}")

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
                f"Vision acceleration check: "
                f"CUDA devices={cuda_count}, "
                f"OpenCL available={opencl_available}, "
                f"OpenCL enabled={opencl_enabled}, "
                f"OpenCV={cv2.__version__}"
            )

    def _mark_target_lost(self):
        self.target_found = False
        self.target_distance_m = None
        self.lost_frame_count += 1

        # Do not immediately forget the previous target.
        # This allows short detection dropouts without jumping to another object.
        if self.lost_frame_count > self.max_lost_frames:
            self.has_previous_target = False

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
                parameters=self.aruco_parameters
            )

        now = time.time()

        if ids is None or len(corners) == 0:
            if DEBUG_ARUCO_DETECTION and now - self.last_aruco_debug_time > 1.0:
                self.last_aruco_debug_time = now
                self.get_logger().info(
                    f"No ArUco marker detected in image {msg.width}x{msg.height}"
                )
            self._mark_target_lost()
            return

        if DEBUG_ARUCO_DETECTION and now - self.last_aruco_debug_time > 1.0:
            self.last_aruco_debug_time = now
            self.get_logger().info(
                f"Detected ArUco IDs: {ids.flatten().tolist()}"
            )

        all_points = []

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            if TARGET_MARKER_ID is not None and int(marker_id) != TARGET_MARKER_ID:
                continue

            pts = marker_corners.reshape((4, 2)).astype(np.float32)

            if cv2.contourArea(pts) >= MIN_MARKER_AREA:
                all_points.append(pts)

        if len(all_points) == 0:
            self._mark_target_lost()
            return

        # One marker works, multiple markers become one ChArUco board target.
        all_points = np.vstack(all_points)

        x, y, w, h = cv2.boundingRect(
            all_points.astype(np.int32)
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
        self.target_found = True
        self.last_seen_time = time.time()

        self.prev_target_cx = self.target_cx
        self.prev_target_cy = self.target_cy
        self.has_previous_target = True
        self.lost_frame_count = 0

    def depth_callback(self, msg):
        # Depth is only useful after the RGB detector has selected a target.
        if not self.target_found:
            self.target_distance_m = None
            return

        try:
            depth = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().warn(f"Depth convert failed: {e}")
            return

        if self.image_w <= 0 or self.image_h <= 0:
            return

        # RGB bbox -> depth bbox coordinate scaling.
        # This works even if RGB and depth resolutions are different.
        scale_x = msg.width / float(self.image_w)
        scale_y = msg.height / float(self.image_h)

        x = int(self.bbox_x * scale_x)
        y = int(self.bbox_y * scale_y)
        w = int(self.bbox_w * scale_x)
        h = int(self.bbox_h * scale_y)

        if w <= 0 or h <= 0:
            return

        # Use the central part of the bbox instead of one center pixel.
        # This is much more stable when the target is small or partially visible.
        margin_x = max(1, int(w * 0.20))
        margin_y = max(1, int(h * 0.20))

        x1 = max(0, x + margin_x)
        x2 = min(msg.width, x + w - margin_x)
        y1 = max(0, y + margin_y)
        y2 = min(msg.height, y + h - margin_y)

        # If the bbox is very small, fall back to a small center patch.
        if x2 <= x1 or y2 <= y1:
            cx = int(self.target_cx * scale_x)
            cy = int(self.target_cy * scale_y)

            if cx < 0 or cy < 0 or cx >= msg.width or cy >= msg.height:
                return

            half = 4
            x1 = max(0, cx - half)
            x2 = min(msg.width, cx + half + 1)
            y1 = max(0, cy - half)
            y2 = min(msg.height, cy + half + 1)

        patch = depth[y1:y2, x1:x2]

        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0.1]
        valid = valid[valid < MAX_VALID_DEPTH_M]

        if valid.size == 0:
            # Do not keep an old distance forever.
            # The control loop also checks last_depth_time.
            self.target_distance_m = None
            return

        # Your depth encoding is 32FC1, so the value is normally in metres.
        # Use a lower percentile rather than pure median so that if the bbox
        # includes some background, the closer target surface is preferred.
        distance_m = float(np.percentile(valid, 30))

        self.target_distance_m = distance_m
        self.last_depth_time = time.time()

def reset_detector_tracking_state(detector):
    """Clear stale visual/depth target state before a new search.
    This prevents search from using a target that was only seen shortly before
    the new START/search command.
    """
    detector.target_found = False
    detector.target_distance_m = None

    detector.last_seen_time = 0.0
    detector.last_depth_time = 0.0

    detector.lost_frame_count = 0
    detector.has_previous_target = False

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
    print("现在请输入 B版 local waypoint position 跟踪飞行时长，单位秒。")
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
    desired_area,
    approach_area,
    orbit_right_speed,
    desired_distance_m,
    stop_event,
    land_event,
    ref_lat,
    ref_lon,
):
    print("\n[TRACK-B] 开始 local waypoint position tracking")
    print(f"[TRACK-B] 设定跟踪时长：{track_duration_s:.1f} s")
    print("[TRACK-B] 逻辑：估计 target_N/E → 滤波/预测 → 生成 follow_N/E → PositionNedYaw")
    print("[TRACK-B] 该版本保留 MAX_POSITION_SETPOINT_STEP_M 作为跳变保护，速度主要由 PX4 MPC 参数限制")

    dt = 1.0 / CONTROL_HZ
    track_start = time.time()
    last_print = 0.0
    last_yaw_rate = 0.0

    filtered_target_n = None
    filtered_target_e = None
    prev_target_n = None
    prev_target_e = None
    prev_target_time = None

    # Active PX4 position setpoint.
    # We do not generate a tiny waypoint every loop. Instead, we generate the
    # full follow point, optionally limit sudden jumps, and only update this
    # active setpoint when enough time has passed and the change is meaningful.
    active_follow_n = None
    active_follow_e = None
    active_yaw_deg = 0.0
    last_setpoint_time = 0.0

    position_tracking_active = False

    # When the target is lost long enough to trigger yaw search, the old
    # filtered target and active waypoint may no longer represent the moving
    # target. Reset them when the target is seen again.
    reset_position_state_after_reacquire = False

    while not stop_event.is_set():
        now = time.time()

        if land_event.is_set():
            print("[TRACK-B] 用户要求降落")
            return "land"

        if now - track_start > track_duration_s:
            print("[FINISH] 达到设定跟踪时长，停止 tracking 并进入悬停等待指令")
            return "finished"

        recently_seen = (
            detector.target_found and
            now - detector.last_seen_time < TARGET_LOST_HOVER_S
        )

        if not recently_seen:
            lost_time = now - detector.last_seen_time

            # Stage 1: very short target loss. Keep the latest B waypoint so
            # one or two missed detections do not disturb the tracking motion.
            if (
                position_tracking_active and
                active_follow_n is not None and
                active_follow_e is not None and
                lost_time < POSITION_TRACK_HOLD_AFTER_TARGET_LOST_S
            ):
                await drone.offboard.set_position_ned(
                    PositionNedYaw(
                        active_follow_n,
                        active_follow_e,
                        -TAKEOFF_ALT_M,
                        active_yaw_deg
                    )
                )
                await asyncio.sleep(dt)
                continue

            # Stage 2: after the short hold, yaw-search in place. With the
            # default 8 deg/s this gives a full 360 deg search in 45 s.
            yaw_search_time = max(
                0.0,
                lost_time - POSITION_TRACK_HOLD_AFTER_TARGET_LOST_S
            )

            if yaw_search_time > LOST_YAW_SEARCH_TIMEOUT_S:
                print("[LOST] 目标丢失太久，停止 tracking 并进入悬停等待指令")
                return "lost_timeout"

            if lost_time > TARGET_LOST_LAND_S:
                print("[LOST] 目标丢失太久，停止 tracking 并进入悬停等待指令")
                return "lost_timeout"

            if now - last_print > 1.0:
                last_print = now
                print("[LOST] 目标短暂丢失 → 保持最近 waypoint")

            reset_position_state_after_reacquire = True

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    0.0,
                    0.0,
                    0.0,
                    LOST_YAW_SEARCH_RATE_DEG_S
                )
            )

            await asyncio.sleep(dt)
            continue

        if reset_position_state_after_reacquire:
            filtered_target_n = None
            filtered_target_e = None
            prev_target_n = None
            prev_target_e = None
            prev_target_time = None
            active_follow_n = None
            active_follow_e = None
            active_yaw_deg = 0.0
            last_setpoint_time = 0.0
            position_tracking_active = False
            reset_position_state_after_reacquire = False

        if detector.image_w > 0:
            half_w = detector.image_w / 2.0
        else:
            half_w = 960.0

        if abs(detector.error_x) < YAW_DEADBAND_PX:
            raw_yaw_rate = 0.0
        else:
            error_norm = detector.error_x / half_w
            raw_yaw_rate = KP_YAW_NORM * error_norm

        raw_yaw_rate = clamp(
            raw_yaw_rate,
            -MAX_TRACK_YAW_RATE_DEG_S,
            MAX_TRACK_YAW_RATE_DEG_S
        )

        max_yaw_delta = YAW_SLEW_RATE_DEG_S2 * dt
        yaw_rate_for_fallback = clamp(
            raw_yaw_rate,
            last_yaw_rate - max_yaw_delta,
            last_yaw_rate + max_yaw_delta
        )
        last_yaw_rate = yaw_rate_for_fallback

        abs_error_x = abs(detector.error_x)

        if detector.image_w > 0:
            center_error_limit = max(60.0, detector.image_w * 0.08)
        else:
            center_error_limit = CENTER_ERROR_LIMIT

        target_low = (
            detector.image_h > 0 and
            detector.target_cy > detector.image_h * TARGET_LOW_RATIO
        )

        bbox_touch_bottom = (
            detector.image_h > 0 and
            detector.bbox_y + detector.bbox_h > detector.image_h * BBOX_BOTTOM_RATIO
        )

        depth_valid = (
            USE_DEPTH_DISTANCE_CONTROL and
            detector.target_distance_m is not None and
            now - detector.last_depth_time < DEPTH_TIMEOUT_S and
            0.5 < detector.target_distance_m < MAX_VALID_DEPTH_M
        )

        mode = "NONE"
        distance_source = "NONE"
        forward_speed = 0.0
        right_speed = 0.0
        yaw_command_deg = 0.0
        debug_extra = ""

        if (target_low or bbox_touch_bottom) and not position_tracking_active:
            mode = "BACK_OFF_IMAGE_EDGE"
            distance_source = "EDGE"
            forward_speed = IMAGE_EDGE_BACKOFF_SPEED
            right_speed = 0.0

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    forward_speed,
                    right_speed,
                    0.0,
                    yaw_rate_for_fallback
                )
            )

        elif depth_valid:
            body_est = estimate_target_body_from_image(detector)
            pos = await get_position_once(drone, timeout_s=0.7)
            heading_deg = await get_heading_once(drone, timeout_s=0.5)

            if body_est is None or pos is None or heading_deg is None:
                mode = "POSITION_WAYPOINT_WAIT"
                distance_source = "POSE_WAIT"
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
            else:
                target_forward_body, target_right_body = body_est

                drone_n, drone_e = meters_from_gps(
                    ref_lat,
                    ref_lon,
                    pos.latitude_deg,
                    pos.longitude_deg
                )

                target_n_raw, target_e_raw = body_to_local_ne(
                    drone_n,
                    drone_e,
                    heading_deg,
                    target_forward_body,
                    target_right_body
                )

                if filtered_target_n is None:
                    filtered_target_n = target_n_raw
                    filtered_target_e = target_e_raw
                else:
                    alpha = TARGET_POSITION_FILTER_ALPHA
                    filtered_target_n = (
                        alpha * target_n_raw + (1.0 - alpha) * filtered_target_n
                    )
                    filtered_target_e = (
                        alpha * target_e_raw + (1.0 - alpha) * filtered_target_e
                    )

                target_n_used = filtered_target_n
                target_e_used = filtered_target_e
                target_v_n = 0.0
                target_v_e = 0.0
                target_speed = 0.0

                if (
                    prev_target_n is not None and
                    prev_target_e is not None and
                    prev_target_time is not None
                ):
                    sample_dt = max(now - prev_target_time, 1e-3)
                    target_v_n = (filtered_target_n - prev_target_n) / sample_dt
                    target_v_e = (filtered_target_e - prev_target_e) / sample_dt
                    target_speed = math.sqrt(
                        target_v_n * target_v_n + target_v_e * target_v_e
                    )

                    # Bound prediction speed so one noisy visual update does not
                    # create a huge predicted target jump.
                    if target_speed > MAX_PREDICT_SPEED_M_S:
                        scale = MAX_PREDICT_SPEED_M_S / target_speed
                        target_v_n *= scale
                        target_v_e *= scale
                        target_speed = MAX_PREDICT_SPEED_M_S

                    target_n_used = (
                        filtered_target_n + target_v_n * TARGET_PREDICT_TIME_S
                    )
                    target_e_used = (
                        filtered_target_e + target_v_e * TARGET_PREDICT_TIME_S
                    )

                prev_target_n = filtered_target_n
                prev_target_e = filtered_target_e
                prev_target_time = now

                follow_n_raw, follow_e_raw, target_range = generate_follow_point(
                    drone_n,
                    drone_e,
                    target_n_used,
                    target_e_used,
                    desired_distance_m
                )

                # Keep this limit, but use it only as a jump guard.
                # It limits sudden changes relative to the active setpoint,
                # not the physical step that the UAV is asked to move.
                follow_n, follow_e = limit_position_setpoint_step(
                    active_follow_n,
                    active_follow_e,
                    follow_n_raw,
                    follow_e_raw,
                    MAX_POSITION_SETPOINT_STEP_M
                )

                yaw_candidate_deg = math.degrees(
                    math.atan2(target_e_used - drone_e, target_n_used - drone_n)
                )

                setpoint_updated = False

                if active_follow_n is None or active_follow_e is None:
                    active_follow_n = follow_n
                    active_follow_e = follow_e
                    active_yaw_deg = yaw_candidate_deg
                    last_setpoint_time = now
                    setpoint_updated = True
                    position_tracking_active = True

                elif now - last_setpoint_time >= POSITION_SETPOINT_UPDATE_S:
                    position_change = distance_ne(
                        follow_n,
                        follow_e,
                        active_follow_n,
                        active_follow_e
                    )
                    yaw_change = abs(
                        angle_diff_deg(yaw_candidate_deg, active_yaw_deg)
                    )

                    if (
                        position_change >= POSITION_UPDATE_DISTANCE_M or
                        yaw_change >= YAW_UPDATE_DEG
                    ):
                        active_follow_n = follow_n
                        active_follow_e = follow_e
                        active_yaw_deg = yaw_candidate_deg
                        last_setpoint_time = now
                        setpoint_updated = True

                # Keep sending the active setpoint every control loop so Offboard
                # remains alive, but do not keep changing the target point at 10 Hz.
                yaw_command_deg = active_yaw_deg
                await drone.offboard.set_position_ned(
                    PositionNedYaw(
                        active_follow_n,
                        active_follow_e,
                        -TAKEOFF_ALT_M,
                        yaw_command_deg
                    )
                )

                mode = "POSITION_WAYPOINT"
                distance_source = "DEPTH+LOCAL"

                debug_extra = (
                    f" target_body=({target_forward_body:.1f},{target_right_body:+.1f})"
                    f" drone_NE=({drone_n:+.1f},{drone_e:+.1f})"
                    f" target_NE=({target_n_used:+.1f},{target_e_used:+.1f})"
                    f" follow_raw_NE=({follow_n_raw:+.1f},{follow_e_raw:+.1f})"
                    f" active_NE=({active_follow_n:+.1f},{active_follow_e:+.1f})"
                    f" range={target_range:.1f}m"
                    f" target_v={target_speed:.2f}m/s"
                    f" yaw_cmd={yaw_command_deg:+.1f}"
                    f" updated={int(setpoint_updated)}"
                    f" heading={heading_deg:.1f}"
                )

        else:
            mode = "AREA_FALLBACK"
            distance_source = "AREA"

            area_error = desired_area - detector.area
            forward_speed = KP_FORWARD * area_error
            forward_speed = clamp(
                forward_speed,
                -MAX_BACKWARD_SPEED,
                MAX_FORWARD_SPEED
            )

            if detector.area < approach_area * AREA_FAST_APPROACH_RATIO:
                forward_speed = MAX_FORWARD_SPEED
            elif detector.area < approach_area and forward_speed > 0.0:
                forward_speed = max(forward_speed, MIN_AREA_APPROACH_SPEED)

            if abs_error_x > center_error_limit:
                forward_speed *= 0.65
                right_speed = 0.0
                mode = "AREA_TURNING"
            elif detector.area < approach_area:
                right_speed = 0.0
                mode = "AREA_APPROACH"
            else:
                right_speed = orbit_right_speed
                mode = "AREA_SIDE_TRACK"

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    forward_speed,
                    right_speed,
                    0.0,
                    yaw_rate_for_fallback
                )
            )

        if now - last_print > 0.5:
            last_print = now

            dist_text = "None"
            if detector.target_distance_m is not None:
                dist_text = f"{detector.target_distance_m:.2f}m"

            print(
                f"[{mode}] src={distance_source} "
                f"area={detector.area:.0f} "
                f"dist={dist_text} "
                f"desired_area={desired_area:.0f} "
                f"desired_dist={desired_distance_m:.1f}m "
                f"error_x={detector.error_x:+.1f} "
                f"forward={forward_speed:+.2f} "
                f"right={right_speed:+.2f} "
                f"down=+0.00 "
                f"yaw_rate={yaw_rate_for_fallback:+.1f}"
                f"{debug_extra}"
            )

        await asyncio.sleep(dt)

    return "stopped"

async def main_async():
    rclpy.init()

    detector = CharucoTargetDetector()
    drone = System()

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

            desired_area = await ask_float(
                "请输入期望目标面积 desired_area，数值越大离目标越近，例如 1800",
                default_value=1800.0,
                min_value=300.0,
                max_value=300000.0,
                stop_event=stop_event
            )

            if desired_area is None:
                if land_event.is_set():
                    break
                continue

            approach_area = await ask_float(
                "请输入开始绕飞的最小目标面积 approach_area，例如 1200",
                default_value=1200.0,
                min_value=100.0,
                max_value=300000.0,
                stop_event=stop_event
            )

            if approach_area is None:
                if land_event.is_set():
                    break
                continue

            orbit_right_speed = await ask_float(
                "请输入额外横向速度 m/s，通常填 0；正负可给轻微侧向偏置，例如 0.00",
                default_value=0.0,
                min_value=-1.0,
                max_value=1.0,
                stop_event=stop_event
            )

            if orbit_right_speed is None:
                if land_event.is_set():
                    break
                continue

            desired_distance_m = await ask_float(
                "请输入期望跟随距离 desired_distance_m，单位 m，例如 5.0",
                default_value=DEFAULT_DESIRED_DISTANCE_M,
                min_value=1.0,
                max_value=MAX_VALID_DEPTH_M,
                stop_event=stop_event
            )

            if desired_distance_m is None:
                if land_event.is_set():
                    break
                continue

            # Tracking stage: allow land/stop/0 while the aircraft is actively tracking.
            cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

            track_result = await visual_orbit_control(
                drone,
                detector,
                track_duration_s,
                desired_area,
                approach_area,
                orbit_right_speed,
                desired_distance_m,
                stop_event,
                land_event,
                ref_lat,
                ref_lon,
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


