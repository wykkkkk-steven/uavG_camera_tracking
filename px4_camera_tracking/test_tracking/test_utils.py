"""
工具函数

clamp, slew_limit, 坐标转换, GPS计算, 控制台交互等
"""

import asyncio
import math
import select
import sys

import numpy as np


# ================================================================
# 数值工具
# ================================================================
def clamp(value, low, high):
    return max(low, min(high, value))


def slew_limit(target, previous, max_rate_per_s, dt):
    max_step = max_rate_per_s * dt
    return clamp(target, previous - max_step, previous + max_step)


# ================================================================
# 向量 / 角度工具（预留，B 版位置跟踪可能用到）
# ================================================================
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


def angle_diff_deg(a, b):
    """最小有符号角度差"""
    diff = (a - b) % 360.0
    if diff > 180.0:
        diff -= 360.0
    return diff


# ================================================================
# GPS / 坐标转换（预留，B 版位置跟踪可能用到）
# ================================================================
def meters_from_gps(lat0_deg, lon0_deg, lat_deg, lon_deg):
    dn = (lat_deg - lat0_deg) * 111_320.0
    de = (lon_deg - lon0_deg) * 111_320.0 * math.cos(math.radians(lat0_deg))
    return dn, de


def distance_ne(n1, e1, n2, e2):
    return math.sqrt((n2 - n1) ** 2 + (e2 - e1) ** 2)


def body_to_local_ne(drone_n, drone_e, heading_deg, forward_body, right_body):
    yaw_rad = math.radians(heading_deg)
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
    if prev_n is None or prev_e is None:
        return new_n, new_e
    dn = new_n - prev_n
    de = new_e - prev_e
    dist = math.sqrt(dn * dn + de * de)
    if dist <= max_step_m or dist < 1e-6:
        return new_n, new_e
    ratio = max_step_m / dist
    return prev_n + dn * ratio, prev_e + de * ratio


# ================================================================
# 控制台交互
# ================================================================
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
            f"{prompt} 默认 {default_value}: ", stop_event=stop_event
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
