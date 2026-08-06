"""
检测器
YOLO 检测 + bbox 张角法测距 + Looming 测速
（需要接入正确的yolo模型）
"""

import asyncio
import math
import time

import numpy as np
import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

from test_params import (
    IMAGE_TOPIC, CAMERA_INFO_TOPIC,
    YOLO_MODEL_PATH, YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD, YOLO_TARGET_CLASSES,
    VIEW_SIZE_TABLE, DEFAULT_TARGET_SIZE,
    LOOMING_ALPHA, LOOMING_MIN_AREA,
)


# ================================================================
# 张角法测距
# ================================================================
def interpolate_target_size(bbox_w, bbox_h):
    """根据 bbox 宽高比插值目标真实投影尺寸。"""
    if bbox_h < 1:
        return DEFAULT_TARGET_SIZE

    ratio = bbox_w / bbox_h
    sorted_ratios = sorted(VIEW_SIZE_TABLE.keys())

    if ratio <= sorted_ratios[0]:
        return VIEW_SIZE_TABLE[sorted_ratios[0]]
    if ratio >= sorted_ratios[-1]:
        return VIEW_SIZE_TABLE[sorted_ratios[-1]]

    for i in range(len(sorted_ratios) - 1):
        r0 = sorted_ratios[i]
        r1 = sorted_ratios[i + 1]
        if r0 <= ratio <= r1:
            w0, h0 = VIEW_SIZE_TABLE[r0]
            w1, h1 = VIEW_SIZE_TABLE[r1]
            t = (ratio - r0) / (r1 - r0)
            return (w0 + t * (w1 - w0), h0 + t * (h1 - h0))

    return DEFAULT_TARGET_SIZE


def compute_distance_from_bbox(bbox_w, bbox_h, fx, fy):
    """
    bbox 张角法测距。宽高比插值 → 真实尺寸 → 距离。
    返回: (distance_m, confidence)
    """
    if bbox_w < 3 or bbox_h < 3:
        return None, 0.0

    real_w, real_h = interpolate_target_size(bbox_w, bbox_h)

    distances = []
    confidences = []

    if bbox_h > 3 and fy > 0:
        d_h = real_h * fy / bbox_h
        distances.append(d_h)
        confidences.append(0.5 if bbox_h > 30 else (0.35 if bbox_h > 15 else 0.2))

    if bbox_w > 3 and fx > 0:
        d_w = real_w * fx / bbox_w
        distances.append(d_w)
        confidences.append(0.5 if bbox_w > 30 else (0.35 if bbox_w > 15 else 0.2))

    if not distances:
        return None, 0.0

    total_w = sum(confidences)
    if total_w < 1e-6:
        return None, 0.0

    distance = sum(d * c for d, c in zip(distances, confidences)) / total_w
    confidence = max(confidences)

    return distance, confidence


# ================================================================
# YOLO 检测器节点
# ================================================================
class YOLODetectorNode(Node):
    """
    YOLO 检测 + bbox 张角法测距 + Looming 测速。

    对外属性（tracker 每帧读取）:
        target_found: bool      - 是否检测到目标
        bbox: (x, y, w, h)     - YOLO 检测框
        distance_m: float       - EMA 滤波后距离
        raw_distance_m: float   - 原始距离
        bearing_deg: float     - 目标水平偏角
        confidence: float       - 检测置信度
        range_rate: float       - Looming 测速 (m/s)
        area: float             - 当前 bbox 面积
        image_size: (w, h)      - 图像尺寸
        error_x: float          - 图像中心误差(px)，兼容 uavG
        image_w: int            - 图像宽度，兼容 uavG
        target_distance_m: float|None  - 兼容 uavG 的接口
        target_bearing_deg: float       - 兼容 uavG 的接口
    """

    def __init__(self):
        super().__init__("yolo_detector_node")

        self._yolo_model = None
        self._yolo_model_path = YOLO_MODEL_PATH

        self._bridge = CvBridge()
        self._camera_info_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self._camera_info_callback, 10
        )
        self._image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self._image_callback, 10
        )

        # 相机内参
        self._fx = None
        self._fy = None
        self._camera_info_received = False

        # ── 对外状态 ──
        self.target_found = False
        self.bbox = (0, 0, 0, 0)
        self.distance_m = 0.0
        self.raw_distance_m = 0.0
        self.bearing_deg = 0.0
        self.confidence = 0.0
        self.range_rate = 0.0
        self.area = 0.0
        self.image_size = (640, 480)

        # uavG 兼容属性
        self.error_x = 0.0
        self.image_w = 640
        self.target_distance_m = None
        self.target_bearing_deg = 0.0

        # EMA 距离滤波
        self._filtered_distance = None
        self._alpha_base = 0.4

        # Looming
        self._prev_area = None
        self._prev_area_time = None

        self.lost_frame_count = 0
        self.get_logger().info("YOLODetectorNode 已初始化")

    # ── YOLO 模型懒加载 ──
    def _ensure_yolo_model(self):
        if self._yolo_model is None:
            try:
                from ultralytics import YOLO
                self._yolo_model = YOLO(self._yolo_model_path)
                self.get_logger().info(f"YOLO 模型已加载: {self._yolo_model_path}")
            except Exception as e:
                self.get_logger().error(f"YOLO 模型加载失败 [{self._yolo_model_path}]: {e}")
                raise

    # ── 相机内参 ──
    def _camera_info_callback(self, msg: CameraInfo):
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self._fx = float(K[0, 0])
        self._fy = float(K[1, 1])
        self._camera_info_received = True
        self.destroy_subscription(self._camera_info_sub)

    def _ensure_camera_fallback(self, img_w, img_h):
        if self._camera_info_received:
            return
        self._fx = img_w * 0.5 / math.tan(math.radians(45))
        self._fy = self._fx
        self._camera_info_received = True
        self.get_logger().warn("无 camera_info，用 90° FOV 估算内参")

    # ── 图像回调（核心） ──
    def _image_callback(self, msg: Image):
        if self._yolo_model is None:
            try:
                self._ensure_yolo_model()
            except Exception:
                return  # 模型加载失败，跳过本帧

        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"图像转换失败: {e}")
            return

        img_h, img_w = frame.shape[:2]
        self.image_size = (img_w, img_h)
        self.image_w = img_w
        self._ensure_camera_fallback(img_w, img_h)

        # YOLO 检测
        results = self._yolo_model(
            frame, conf=YOLO_CONF_THRESHOLD, iou=YOLO_IOU_THRESHOLD, verbose=False
        )

        if not results or len(results[0].boxes) == 0:
            self._mark_target_lost()
            return

        best_box = None
        best_conf = 0.0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if YOLO_TARGET_CLASSES and cls_id not in YOLO_TARGET_CLASSES:
                continue
            conf = float(box.conf[0])
            if conf > best_conf:
                best_conf = conf
                best_box = box

        if best_box is None:
            self._mark_target_lost()
            return

        xyxy = best_box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
        w, h = x2 - x1, y2 - y1

        if w < 5 or h < 5:
            self._mark_target_lost()
            return

        self.bbox = (x1, y1, w, h)
        self.target_found = True
        self.lost_frame_count = 0

        area = float(w * h)
        self.area = area

        target_cx = x1 + w / 2.0
        self.error_x = target_cx - img_w / 2.0

        # ── 张角法测距 ──
        raw_distance, conf = compute_distance_from_bbox(w, h, self._fx, self._fy)

        if raw_distance is None:
            self._mark_target_lost()
            return

        self.raw_distance_m = raw_distance
        self.confidence = conf

        if self._filtered_distance is None:
            self._filtered_distance = raw_distance
        else:
            self._filtered_distance = (
                self._alpha_base * raw_distance
                + (1.0 - self._alpha_base) * self._filtered_distance
            )
        self.distance_m = self._filtered_distance

        self.target_distance_m = self.distance_m

        # ── Bearing ──
        if self._fx and self._fx > 0:
            self.bearing_deg = math.degrees(math.atan2(self.error_x, self._fx))
        else:
            self.bearing_deg = 0.0
        self.target_bearing_deg = self.bearing_deg

        # ── Looming → range_rate ──
        # bbox 出画面边界时跳过 Looming
        bbox_touches_edge = (
            x1 <= 2 or y1 <= 2 or x2 >= img_w - 2 or y2 >= img_h - 2
        )

        now = time.time()
        if bbox_touches_edge:
            self._prev_area = None
            self._prev_area_time = None
        else:
            if (self._prev_area is not None
                    and area > LOOMING_MIN_AREA
                    and self._prev_area > LOOMING_MIN_AREA):
                dt_area = now - self._prev_area_time if self._prev_area_time else 0.1
                if dt_area > 1e-6:
                    area_rate = (area - self._prev_area) / dt_area
                    relative_rate = area_rate / self._prev_area
                    raw_range_rate = -(relative_rate / 2.0) * raw_distance
                    self.range_rate = (
                        LOOMING_ALPHA * raw_range_rate
                        + (1.0 - LOOMING_ALPHA) * self.range_rate
                    )

            self._prev_area = area
            self._prev_area_time = now

    def _mark_target_lost(self):
        self.target_found = False
        self.lost_frame_count += 1
        if self.lost_frame_count > 3:
            self._filtered_distance = None
            self._prev_area = None
            self._prev_area_time = None
            self.area = 0.0
            self.range_rate = 0.0
            self.target_distance_m = None

    def reset_tracking_state(self):
        self.target_found = False
        self.bbox = (0, 0, 0, 0)
        self.distance_m = 0.0
        self.raw_distance_m = 0.0
        self.bearing_deg = 0.0
        self.confidence = 0.0
        self.range_rate = 0.0
        self.lost_frame_count = 0
        self._filtered_distance = None
        self._prev_area = None
        self._prev_area_time = None
        self.area = 0.0
        self.error_x = 0.0
        self.target_distance_m = None
        self.target_bearing_deg = 0.0


# ================================================================
# ROS2 spin（异步后台运行）
# ================================================================
async def spin_ros_node(node):
    while rclpy.ok():
        try:
            rclpy.spin_once(node, timeout_sec=0.01)
        except Exception as e:
            node.get_logger().error(f"ROS spin 异常: {e}")
            await asyncio.sleep(1.0)
            continue
        await asyncio.sleep(0.005)
