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
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed


# ================= 基本连接参数 =================
UDP_ADDR = "udpin://0.0.0.0:14540"

IMAGE_TOPIC = "/camera/image_raw"
TAKEOFF_ALT_M = 5.0
DEPTH_TOPIC = "/camera/depth/image_raw"

# ================= 搜索 / 绕飞参数 =================
CONTROL_HZ = 10.0

# 起飞后先不搜索，用户输入 START 后才开始搜索
SEARCH_YAW_RATE_DEG_S = 15.0
SEARCH_TIMEOUT_S = 45.0

# orbit 中目标丢失保护
TARGET_LOST_HOVER_S = 1.5
TARGET_LOST_LAND_S = 20.0

# OpenCV 视觉控制参数
KP_YAW = 0.10
KP_FORWARD = 0.00060

DESIRED_AREA = 3500.0
MIN_TARGET_AREA = 90.0

MAX_FORWARD_SPEED = 0.90
MAX_BACKWARD_SPEED = 0.60
ORBIT_RIGHT_SPEED = 0.35
MAX_YAW_RATE_DEG_S = 75.0
CENTER_ERROR_LIMIT = 60.0

# ================= Depth distance control =================
# RGB is still used to find the red target.
# Depth is used to control forward/backward distance when a valid distance exists.
USE_DEPTH_DISTANCE_CONTROL = True
DEFAULT_DESIRED_DISTANCE_M = 5.0
DISTANCE_DEADBAND_M = 0.4
KP_DISTANCE = 0.18
MAX_VALID_DEPTH_M = 15.0
DEPTH_TIMEOUT_S = 0.5

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
ARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
TARGET_MARKER_ID = None
MIN_MARKER_AREA = 80.0

# Image-edge safety: if the target is going out of the bottom of the camera view,
# do not trust area; back off horizontally.
TARGET_LOW_RATIO = 0.78
BBOX_BOTTOM_RATIO = 0.95
IMAGE_EDGE_BACKOFF_SPEED = -0.25

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


class RedTargetDetector(Node):
    def __init__(self):
        super().__init__("charuco_target_detector")

        self.bridge = CvBridge()

        self.target_found = False
        self.error_x = 0.0
        self.error_y = 0.0
        self.area = 0.0
        self.last_seen_time = 0.0
        self.frame_count = 0

        # Debug / tracking information for the selected red object.
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

        if ids is None or len(corners) == 0:
            self._mark_target_lost()
            return

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

        
async def spin_ros_node(detector):
    while rclpy.ok():
        rclpy.spin_once(detector, timeout_sec=0.01)
        await asyncio.sleep(0.001)


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
    await asyncio.sleep(8.0)

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
    print("确认开始 search 红色目标请输入 START")
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
    print("\n[SEARCH] 开始搜索红色目标")
    print(f"[SEARCH] 最多搜索 {SEARCH_TIMEOUT_S:.1f} s")
    print("[SEARCH] 搜索阶段会原地慢速 yaw 旋转")

    search_start = time.time()
    last_print = 0.0
    dt = 1.0 / CONTROL_HZ

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
            print("[SEARCH] 已找到红色目标")

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
                f"error_y={detector.error_y:+.1f}"
            )

            # 找到目标后先停住，等待用户输入飞行时长
            for _ in range(10):
                await drone.offboard.set_velocity_body(
                    VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)
                )
                await asyncio.sleep(0.1)

            return True

        if now - search_start > SEARCH_TIMEOUT_S:
            print("[SEARCH] 搜索超时，未找到红色目标，准备降落")
            land_event.set()
            return False

        if now - last_print > 1.0:
            last_print = now
            print("[SEARCH] target not found → 原地 yaw search")

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
    print("现在请输入 visual orbit 飞行时长，单位秒。")
    print("例如输入 20 表示绕飞 20 秒。")
    print("如果不想绕飞，输入 land。")

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


async def visual_orbit_control(drone, detector, track_duration_s, desired_area, approach_area, orbit_right_speed, desired_distance_m, stop_event, land_event):
    print("\n[TRACK] 开始 visual orbit")
    print(f"[TRACK] 设定绕飞时长：{track_duration_s:.1f} s")
    print("[TRACK] 逻辑：RGB 找目标/yaw，depth 优先控前后距离，area 作为 fallback")

    dt = 1.0 / CONTROL_HZ
    track_start = time.time()
    last_print = 0.0

    # Minimal latch: once the aircraft has entered orbit, small depth drift
    # should be corrected while continuing to orbit, not by stopping right_speed.
    in_depth_orbit = False

    while not stop_event.is_set():
        now = time.time()

        if land_event.is_set():
            print("[TRACK] 用户要求降落")
            return

        if now - track_start > track_duration_s:
            print("[FINISH] 达到设定飞行时长，准备降落")
            land_event.set()
            return

        recently_seen = (
            detector.target_found and
            now - detector.last_seen_time < TARGET_LOST_HOVER_S
        )

        if not recently_seen:
            lost_time = now - detector.last_seen_time

            in_depth_orbit = False

            if lost_time > TARGET_LOST_LAND_S:
                print("[LOST] 目标丢失太久，准备降落")
                land_event.set()
                return

            if now - last_print > 1.0:
                last_print = now
                print("[LOST] 目标短暂丢失 → 慢速 yaw search")

            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(
                    0.0,
                    0.0,
                    0.0,
                    SEARCH_YAW_RATE_DEG_S * 0.6
                )
            )

            await asyncio.sleep(dt)
            continue

        yaw_rate = KP_YAW * detector.error_x
        yaw_rate = clamp(
            yaw_rate,
            -MAX_YAW_RATE_DEG_S,
            MAX_YAW_RATE_DEG_S
        )

        # Fixed-altitude horizontal tracking only:
        # - RGB error_x controls yaw
        # - depth distance controls horizontal forward/backward speed when valid
        # - area controls horizontal forward/backward speed only as fallback
        # - target low / bbox touching bottom means horizontal back-off
        # - the third MAVSDK velocity argument is always 0.0, so no commanded descent/ascent
        abs_error_x = abs(detector.error_x)

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

        distance_source = "NONE"

        if target_low or bbox_touch_bottom:
            mode = "BACK_OFF_IMAGE_EDGE"
            distance_source = "EDGE"
            forward_speed = IMAGE_EDGE_BACKOFF_SPEED
            right_speed = 0.0

        elif depth_valid:
            mode = "DEPTH_FOLLOW"
            distance_source = "DEPTH"

            distance_error = detector.target_distance_m - desired_distance_m

            # First approach: use full speed authority while far away.
            # After entering orbit: keep tangential motion unless the distance
            # is seriously wrong. This avoids side-fly -> stop -> approach loops.
            if not in_depth_orbit:
                forward_speed = depth_approach_speed(distance_error)

                if abs_error_x > CENTER_ERROR_LIMIT:
                    forward_speed *= 0.65
                    right_speed = 0.0
                    mode = "DEPTH_TURNING"
                elif abs(distance_error) > DISTANCE_DEADBAND_M:
                    right_speed = 0.0
                    mode = "DEPTH_APPROACH"
                else:
                    in_depth_orbit = True
                    forward_speed = 0.0
                    right_speed = orbit_right_speed
                    mode = "DEPTH_ORBIT"
            else:
                # Already orbiting. Do not leave orbit for ordinary range drift.
                if abs_error_x > CENTER_ERROR_LIMIT * 1.8:
                    # Target is near the image edge: reduce orbit but do not
                    # fully cancel it unless the target is badly lost.
                    right_speed = orbit_right_speed * 0.60
                    forward_speed = 0.0
                    mode = "DEPTH_ORBIT_TURNING"
                elif distance_error > ORBIT_REAPPROACH_ERROR_M:
                    # Only a clearly far target switches back to approach.
                    in_depth_orbit = False
                    forward_speed = depth_approach_speed(distance_error)
                    right_speed = 0.0
                    mode = "DEPTH_REAPPROACH"
                elif abs(distance_error) > ORBIT_DISTANCE_BAND_M:
                    # Slow radial correction while still orbiting.
                    forward_speed = clamp(
                        KP_DISTANCE * distance_error,
                        -ORBIT_CORRECTION_SPEED,
                        ORBIT_CORRECTION_SPEED
                    )
                    right_speed = orbit_right_speed
                    mode = "DEPTH_ORBIT_CORRECT"
                else:
                    forward_speed = 0.0
                    right_speed = orbit_right_speed
                    mode = "DEPTH_ORBIT"

        else:
            mode = "AREA_FALLBACK"
            distance_source = "AREA"

            # Use the user-entered desired_area, not the global default.
            area_error = desired_area - detector.area
            forward_speed = KP_FORWARD * area_error
            forward_speed = clamp(
                forward_speed,
                -MAX_BACKWARD_SPEED,
                MAX_FORWARD_SPEED
            )

            # Before depth becomes valid, RGB/area is only for fast approach.
            # Keep the original area logic, but use decisive speed when the target
            # is still much smaller than the approach threshold.
            if detector.area < approach_area * AREA_FAST_APPROACH_RATIO:
                forward_speed = MAX_FORWARD_SPEED
            elif detector.area < approach_area and forward_speed > 0.0:
                forward_speed = max(forward_speed, MIN_AREA_APPROACH_SPEED)

            # 如果目标偏得很厉害，只降低前进速度，不完全停住。
            if abs_error_x > CENTER_ERROR_LIMIT:
                forward_speed *= 0.65
                right_speed = 0.0
                mode = "AREA_TURNING"

            elif detector.area < approach_area:
                right_speed = 0.0
                mode = "AREA_APPROACH"

            else:
                # Area fallback can start a visual orbit before depth exists.
                # Once depth appears later, the depth branch will take over.
                in_depth_orbit = False
                right_speed = orbit_right_speed
                mode = "AREA_ORBIT"

        await drone.offboard.set_velocity_body(
            VelocityBodyYawspeed(
                forward_speed,
                right_speed,
                0.0,
                yaw_rate
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
                f"yaw_rate={yaw_rate:+.1f}"
            )

        await asyncio.sleep(dt)


async def main_async():
    rclpy.init()

    detector = RedTargetDetector()
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

        # 起飞阶段不进入 Offboard；先让 PX4 自己完成 takeoff/hover。
        # 用户输入 START 后，才进入 Offboard velocity 模式开始 search/tracking。
        start_search = await ask_start_search(stop_event, land_event)
        if not start_search:
            return

        offboard_ok = await start_offboard_velocity(drone)
        if not offboard_ok:
            return

        # search 阶段允许输入 land/stop/0
        cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

        found = await search_until_target_found(
            drone,
            detector,
            stop_event,
            land_event
        )

        # 找到目标后，停止 command_listener，避免它抢走“飞行时长”的输入
        if cmd_task is not None:
            cmd_task.cancel()
            try:
                await cmd_task
            except asyncio.CancelledError:
                pass
            cmd_task = None

        if not found:
            return

        # 找到目标后再问飞行时长
        track_duration_s = await ask_track_duration(stop_event, land_event)
        if track_duration_s is None:
            return

        desired_area = await ask_float(
            "请输入期望目标面积 desired_area，数值越大离目标越近，例如 1800",
            default_value=1800.0,
            min_value=300.0,
            max_value=30000.0,
            stop_event=stop_event
        )

        if desired_area is None:
            land_event.set()
            return

        approach_area = await ask_float(
            "请输入开始绕飞的最小目标面积 approach_area，例如 1200",
            default_value=1200.0,
            min_value=100.0,
            max_value=30000.0,
            stop_event=stop_event
        )

        if approach_area is None:
            land_event.set()
            return

        orbit_right_speed = await ask_float(
            "请输入绕飞横向速度 m/s，正负决定绕飞方向，例如 0.12",
            default_value=0.03,
            min_value=-1.0,
            max_value=1.0,
            stop_event=stop_event
        )

        if orbit_right_speed is None:
            land_event.set()
            return

        desired_distance_m = await ask_float(
            "请输入期望跟随距离 desired_distance_m，单位 m，例如 5.0",
            default_value=DEFAULT_DESIRED_DISTANCE_M,
            min_value=1.0,
            max_value=MAX_VALID_DEPTH_M,
            stop_event=stop_event
        )

        if desired_distance_m is None:
            land_event.set()
            return

        # orbit 阶段重新启动 command_listener，允许飞行中 land/stop/0
        cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

        await visual_orbit_control(
            drone,
            detector,
            track_duration_s,
            desired_area,
            approach_area,
            orbit_right_speed,
            desired_distance_m,
            stop_event,
            land_event
        )

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

