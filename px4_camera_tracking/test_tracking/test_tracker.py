"""
搜索 + 跟踪状态机
"""

import asyncio
import math
import select
import sys
import time

from mavsdk.offboard import VelocityBodyYawspeed

from .test_params import (
    CONTROL_HZ,
    SEARCH_YAW_RATE_DEG_S, SEARCH_TIMEOUT_S, SEARCH_CONFIRM_MIN_FRAMES,
    APPROACH_YAW_RATE_DEG_S, APPROACH_FORWARD_SPEED, APPROACH_ENTER_RANGE_M,
    DISTANCE_MIN_SAFE_M, DISTANCE_OPTIMAL_MIN_M, DISTANCE_OPTIMAL_MAX_M,
    DISTANCE_MAX_VALID_M,
    KP_DISTANCE_CLOSE, KP_DISTANCE_OPTIMAL, KP_DISTANCE_FAR, DISTANCE_DEADBAND_M,
    MAX_TRACK_FORWARD_SPEED, MAX_TRACK_BACKWARD_SPEED,
    KP_BEARING_YAW, BEARING_DEADBAND_DEG,
    MAX_YAW_RATE_CLOSE_DEG_S, MAX_YAW_RATE_OPTIMAL_DEG_S, MAX_YAW_RATE_FAR_DEG_S,
    FORWARD_SLEW_CLOSE_M_S2, FORWARD_SLEW_OPTIMAL_M_S2, FORWARD_SLEW_FAR_M_S2,
    YAW_SLEW_CLOSE_DEG_S2, YAW_SLEW_OPTIMAL_DEG_S2, YAW_SLEW_FAR_DEG_S2,
    ALIGN_BEARING_DEG, ALIGN_CONFIRM_FRAMES, REALIGN_BEARING_DEG,
    DISTANCE_TOLERANCE_M, BEARING_TOLERANCE_DEG, ARRIVAL_CONFIRM_FRAMES,
    URGENT_TOO_CLOSE_MARGIN_M, URGENT_CLOSING_SPEED_M_S, URGENT_CONFIRM_FRAMES,
    URGENT_RETREAT_MIN_SPEED, URGENT_RETREAT_MAX_SPEED, URGENT_RETREAT_KP,
    AREA_JUMP_RATIO, PREVIEW_MAX_DURATION_S, PREVIEW_RECOVER_RATIO,
    TARGET_LOST_HOVER_S, TARGET_LOST_LAND_S,
    LOST_YAW_SEARCH_RATE_DEG_S, LOOMING_MIN_AREA,
    PLANNER_KP_YAW,
    PLANNER_KP_DIST,
    PLANNER_MAX_FORWARD, PLANNER_MAX_RETREAT,
    TARGET_PREDICT_TIME_S,
)
from .test_utils import clamp, slew_limit


class _VelocitySender:
    """发送速度指令，连续失败超过阈值时触发紧急降落。"""
    def __init__(self, land_event=None):
        self._fail_count = 0
        self._land_event = land_event

    async def send(self, drone, forward, yaw_rate):
        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(forward, 0, 0, yaw_rate)
            )
            self._fail_count = 0
        except Exception as e:
            self._fail_count += 1
            if self._fail_count % 10 == 1:
                print(f"[WARN] Offboard 速度指令失败 ({self._fail_count}次): {e}")
            if self._fail_count >= 30:
                print("[ERROR] Offboard 速度指令连续失败 30 次，触发紧急降落")
                if self._land_event:
                    self._land_event.set()


def _check_stdin_command(stop_event, land_event):
    """非阻塞检查 stdin，如果用户输入 land/stop/0 则设置事件。"""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        line = sys.stdin.readline().strip().lower()
        if line in ("land", "stop", "0"):
            land_event.set()
            stop_event.set()
            return True
    return False


async def search_until_target_found(drone, detector, stop_event=None, land_event=None):
    print("[搜索] 开始原地旋转搜索...")
    dt = 1.0 / CONTROL_HZ
    confirm_count = 0
    t0 = time.time()
    sender = _VelocitySender(land_event)

    while True:
        if time.time() - t0 > SEARCH_TIMEOUT_S:
            print("[搜索] 超时，未找到目标")
            return "timeout"

        if land_event is not None and land_event.is_set():
            return "land"
        if stop_event is not None and stop_event.is_set():
            return "land"

        if _check_stdin_command(stop_event, land_event):
            return "land"

        if detector.target_found:
            confirm_count += 1
            if confirm_count >= SEARCH_CONFIRM_MIN_FRAMES:
                dist_str = f"{detector.distance_m:.1f}m" if detector.distance_m > 0 else f">{DISTANCE_MAX_VALID_M}m(远距)"
                print(f"[搜索] 目标确认! bearing={detector.bearing_deg:.1f}° dist={dist_str}")
                await sender.send(drone, 0, 0)
                return "found"
        else:
            confirm_count = 0

        await sender.send(drone, 0, SEARCH_YAW_RATE_DEG_S)
        await asyncio.sleep(dt)


async def approach_until_in_range(drone, detector, stop_event=None, land_event=None):
    """
    远距识别到目标后，向目标飞近直到进入测距范围。
    返回: 'in_range' / 'lost' / 'land'
    """
    print("[接近] 目标远距，向目标飞近...")
    dt = 1.0 / CONTROL_HZ
    sender = _VelocitySender(land_event)

    while True:
        if land_event is not None and land_event.is_set():
            return "land"
        if stop_event is not None and stop_event.is_set():
            return "land"

        if _check_stdin_command(stop_event, land_event):
            return "land"

        if not detector.target_found:
            print("[接近] 目标丢失")
            return "lost"

        if detector.distance_m > 0 and detector.distance_m <= APPROACH_ENTER_RANGE_M:
            print(f"[接近] 进入测距范围 dist={detector.distance_m:.1f}m")
            return "in_range"

        bearing_deg = detector.bearing_deg
        yaw_rate = clamp(
            KP_BEARING_YAW * bearing_deg,
            -APPROACH_YAW_RATE_DEG_S,
            APPROACH_YAW_RATE_DEG_S,
        )
        forward = APPROACH_FORWARD_SPEED

        await sender.send(drone, forward, yaw_rate)
        await asyncio.sleep(dt)


async def visual_tracking_control(drone, detector, desired_distance_m, track_duration_s,
                                  stop_event=None, land_event=None,
                                  planner=None, drone_position_provider=None):
    """精细跟踪核心（状态机）。"""

    dt = 1.0 / CONTROL_HZ
    t0 = time.time()

    state = "ALIGN"
    align_confirm = 0
    arrival_confirm = 0
    urgent_confirm = 0

    preview_active = False
    preview_start_time = None
    preview_frozen_distance = 0.0
    preview_frozen_range_rate = 0.0
    preview_frozen_area = 0.0

    prev_forward = 0.0
    prev_yaw_rate = 0.0
    lost_time_start = None
    prev_area_for_jump = None
    last_log_time = 0.0

    sender = _VelocitySender(land_event)

    print(f"[跟踪] 开始! 期望距离={desired_distance_m:.1f}m 时长={track_duration_s:.0f}s"
          + (" [PLANNER模式]" if planner else " [REACTIVE模式]"))

    while True:
        if track_duration_s > 0 and time.time() - t0 > track_duration_s:
            print("[跟踪] 时长到")
            return "done"

        if land_event is not None and land_event.is_set():
            return "land"
        if stop_event is not None and stop_event.is_set():
            return "land"

        if _check_stdin_command(stop_event, land_event):
            return "land"

        target_found = detector.target_found
        ema_distance_m = detector.distance_m
        bearing_deg = detector.bearing_deg
        range_rate = detector.range_rate
        current_area = detector.area

        if planner is not None and drone_position_provider is not None and target_found:
            try:
                dn, de, dh = drone_position_provider()
                planner.update(bearing_deg, ema_distance_m, dn, de, dh)
            except Exception as e:
                if time.time() - last_log_time > 2.0:
                    print(f"[WARN] Planner update 失败: {e}")

        if not target_found:
            if planner is not None:
                planner.mark_target_lost()
            preview_active = False
            if lost_time_start is None:
                lost_time_start = time.time()
            lost_duration = time.time() - lost_time_start

            if lost_duration < TARGET_LOST_HOVER_S:
                forward = yaw_rate = 0.0
            elif lost_duration < TARGET_LOST_LAND_S:
                forward = 0.0
                yaw_rate = LOST_YAW_SEARCH_RATE_DEG_S
            else:
                print("[跟踪] 目标丢失超时")
                return "lost"

            prev_forward = slew_limit(forward, prev_forward, FORWARD_SLEW_OPTIMAL_M_S2, dt)
            prev_yaw_rate = slew_limit(yaw_rate, prev_yaw_rate, YAW_SLEW_OPTIMAL_DEG_S2, dt)

            await sender.send(drone, prev_forward, prev_yaw_rate)
            prev_area_for_jump = None
            await asyncio.sleep(dt)
            continue
        else:
            lost_time_start = None

        distance_valid = ema_distance_m > 0
        if distance_valid and ema_distance_m > DISTANCE_MAX_VALID_M:
            yaw_rate = clamp(KP_BEARING_YAW * bearing_deg,
                             -APPROACH_YAW_RATE_DEG_S, APPROACH_YAW_RATE_DEG_S)
            forward = APPROACH_FORWARD_SPEED
            prev_forward = slew_limit(forward, prev_forward, FORWARD_SLEW_OPTIMAL_M_S2, dt)
            prev_yaw_rate = slew_limit(yaw_rate, prev_yaw_rate, YAW_SLEW_OPTIMAL_DEG_S2, dt)
            await sender.send(drone, prev_forward, prev_yaw_rate)
            await asyncio.sleep(dt)
            continue

        area_jump_detected = False
        if prev_area_for_jump is not None and prev_area_for_jump > LOOMING_MIN_AREA:
            area_ratio = current_area / prev_area_for_jump
            if area_ratio > AREA_JUMP_RATIO or area_ratio < 1.0 / AREA_JUMP_RATIO:
                area_jump_detected = True
        prev_area_for_jump = current_area

        if area_jump_detected and not preview_active:
            preview_active = True
            preview_start_time = time.time()
            preview_frozen_distance = ema_distance_m
            preview_frozen_range_rate = range_rate
            preview_frozen_area = current_area
            print(f"[PREVIEW] 面积突变，冻结距离={preview_frozen_distance:.2f}m")

        if preview_active:
            preview_elapsed = time.time() - preview_start_time

            area_recovered = False
            if not area_jump_detected and preview_frozen_area > 0:
                area_ratio_to_frozen = current_area / preview_frozen_area
                if 1.0 - PREVIEW_RECOVER_RATIO < area_ratio_to_frozen < 1.0 + PREVIEW_RECOVER_RATIO:
                    area_recovered = True

            if area_recovered:
                preview_active = False
                print("[PREVIEW] 面积恢复，退出预瞄")
            elif preview_elapsed > PREVIEW_MAX_DURATION_S:
                preview_active = False
                print("[PREVIEW] 超时，退出预瞄")

            if preview_active:
                distance_m = preview_frozen_distance + preview_frozen_range_rate * preview_elapsed
                distance_m = clamp(distance_m, DISTANCE_MIN_SAFE_M, DISTANCE_MAX_VALID_M)
                range_rate_for_urgent = preview_frozen_range_rate
            else:
                distance_m = ema_distance_m
                range_rate_for_urgent = range_rate
        else:
            distance_m = ema_distance_m
            range_rate_for_urgent = range_rate

        if distance_m > DISTANCE_MAX_VALID_M:
            distance_m = DISTANCE_MAX_VALID_M

        if distance_m < DISTANCE_OPTIMAL_MIN_M:
            kp_dist = KP_DISTANCE_CLOSE
            max_yaw = MAX_YAW_RATE_CLOSE_DEG_S
            fwd_slew = FORWARD_SLEW_CLOSE_M_S2
            yaw_slew = YAW_SLEW_CLOSE_DEG_S2
        elif distance_m > DISTANCE_OPTIMAL_MAX_M:
            kp_dist = KP_DISTANCE_FAR
            max_yaw = MAX_YAW_RATE_FAR_DEG_S
            fwd_slew = FORWARD_SLEW_FAR_M_S2
            yaw_slew = YAW_SLEW_FAR_DEG_S2
        else:
            kp_dist = KP_DISTANCE_OPTIMAL
            max_yaw = MAX_YAW_RATE_OPTIMAL_DEG_S
            fwd_slew = FORWARD_SLEW_OPTIMAL_M_S2
            yaw_slew = YAW_SLEW_OPTIMAL_DEG_S2

        dangerously_close = distance_m < DISTANCE_MIN_SAFE_M
        too_close = distance_m < (desired_distance_m - URGENT_TOO_CLOSE_MARGIN_M)
        closing_fast = range_rate_for_urgent < -URGENT_CLOSING_SPEED_M_S

        if dangerously_close or (too_close and closing_fast):
            urgent_confirm += 1
        else:
            urgent_confirm = 0

        abs_bearing = abs(bearing_deg)

        if urgent_confirm >= URGENT_CONFIRM_FRAMES:
            state = "URGENT_RETREAT"
        elif state == "URGENT_RETREAT":
            if not (dangerously_close or (too_close and closing_fast)):
                state = "ALIGN"
                align_confirm = 0
        elif abs_bearing > REALIGN_BEARING_DEG:
            state = "ALIGN"
            align_confirm = 0
        elif state == "ALIGN":
            if abs_bearing < ALIGN_BEARING_DEG:
                align_confirm += 1
                if align_confirm >= ALIGN_CONFIRM_FRAMES:
                    if planner is not None and planner.direction_valid:
                        state = "PLANNER_TRACK"
                        arrival_confirm = 0
                    else:
                        state = "TRACK"
            else:
                align_confirm = 0
        elif state == "PLANNER_TRACK":
            if planner is None or not planner.direction_valid:
                state = "TRACK"
                arrival_confirm = 0
            else:
                in_distance = abs(distance_m - desired_distance_m) < DISTANCE_TOLERANCE_M
                if in_distance and abs_bearing < BEARING_TOLERANCE_DEG:
                    arrival_confirm += 1
                    if arrival_confirm >= ARRIVAL_CONFIRM_FRAMES:
                        state = "HOLD"
                else:
                    arrival_confirm = 0
        elif state == "TRACK":
            in_distance = abs(distance_m - desired_distance_m) < DISTANCE_TOLERANCE_M
            in_bearing = abs_bearing < BEARING_TOLERANCE_DEG
            if in_distance and in_bearing:
                arrival_confirm += 1
                if arrival_confirm >= ARRIVAL_CONFIRM_FRAMES:
                    state = "HOLD"
            else:
                arrival_confirm = 0
        elif state == "HOLD":
            in_distance = abs(distance_m - desired_distance_m) < DISTANCE_TOLERANCE_M
            in_bearing = abs_bearing < BEARING_TOLERANCE_DEG
            if not (in_distance and in_bearing):
                if planner is not None and planner.direction_valid:
                    state = "PLANNER_TRACK"
                else:
                    state = "TRACK"
                arrival_confirm = 0

        forward = 0.0
        yaw_rate = 0.0

        if state == "URGENT_RETREAT":
            retreat_speed = clamp(
                URGENT_RETREAT_KP * (desired_distance_m - distance_m),
                URGENT_RETREAT_MIN_SPEED,
                URGENT_RETREAT_MAX_SPEED,
            )
            forward = -retreat_speed

        elif state == "ALIGN":
            distance_error = distance_m - desired_distance_m
            if abs(distance_error) > DISTANCE_DEADBAND_M:
                forward = clamp(
                    kp_dist * distance_error,
                    -MAX_TRACK_BACKWARD_SPEED,
                    MAX_TRACK_FORWARD_SPEED,
                )
            if abs_bearing > BEARING_DEADBAND_DEG:
                yaw_rate = clamp(KP_BEARING_YAW * bearing_deg, -max_yaw, max_yaw)

        elif state == "TRACK":
            distance_error = distance_m - desired_distance_m
            if abs(distance_error) > DISTANCE_DEADBAND_M:
                forward = clamp(
                    kp_dist * distance_error,
                    -MAX_TRACK_BACKWARD_SPEED,
                    MAX_TRACK_FORWARD_SPEED,
                )
            if abs_bearing > BEARING_DEADBAND_DEG:
                yaw_rate = clamp(KP_BEARING_YAW * bearing_deg, -max_yaw, max_yaw)

        elif state == "PLANNER_TRACK":
            if planner is not None and planner.direction_valid:
                predict_n = planner.target_n + planner.target_vel_n * TARGET_PREDICT_TIME_S
                predict_e = planner.target_e + planner.target_vel_e * TARGET_PREDICT_TIME_S
                if drone_position_provider is not None:
                    try:
                        dn, de, dh = drone_position_provider()
                        err_n = predict_n - dn
                        err_e = predict_e - de
                        target_heading = math.degrees(math.atan2(err_e, err_n))
                        heading_err = target_heading - dh
                        heading_err = (heading_err + 180) % 360 - 180
                        yaw_rate = clamp(PLANNER_KP_YAW * heading_err, -max_yaw, max_yaw)
                    except Exception:
                        if abs_bearing > BEARING_DEADBAND_DEG:
                            yaw_rate = clamp(KP_BEARING_YAW * bearing_deg, -max_yaw, max_yaw)
                else:
                    if abs_bearing > BEARING_DEADBAND_DEG:
                        yaw_rate = clamp(KP_BEARING_YAW * bearing_deg, -max_yaw, max_yaw)

                distance_error = distance_m - desired_distance_m
                if distance_m > 0 and abs(distance_error) > DISTANCE_DEADBAND_M:
                    forward = clamp(
                        PLANNER_KP_DIST * distance_error,
                        -PLANNER_MAX_RETREAT, PLANNER_MAX_FORWARD
                    )
            else:
                if abs_bearing > BEARING_DEADBAND_DEG:
                    yaw_rate = clamp(KP_BEARING_YAW * bearing_deg, -max_yaw, max_yaw)

        elif state == "HOLD":
            forward = 0.0
            yaw_rate = 0.0

        prev_forward = slew_limit(forward, prev_forward, fwd_slew, dt)
        prev_yaw_rate = slew_limit(yaw_rate, prev_yaw_rate, yaw_slew, dt)

        await sender.send(drone, prev_forward, prev_yaw_rate)

        now = time.monotonic()
        if now - last_log_time > 0.5:
            last_log_time = now
            state_str = "PREVIEW" if preview_active else state
            rr_str = f" rr={range_rate:.2f}" if abs(range_rate) > 0.01 else ""
            planner_str = ""
            if state == "PLANNER_TRACK" and planner is not None and planner.direction_valid:
                try:
                    dn, de, _ = drone_position_provider()
                    tgt_d = math.sqrt((planner.target_n - dn)**2 + (planner.target_e - de)**2)
                    planner_str = f" tgt_d={tgt_d:.2f}m"
                except Exception:
                    pass
            print(
                f"[{state_str}] d={distance_m:.2f}m b={bearing_deg:.1f}° "
                f"fwd={prev_forward:.2f} yaw={prev_yaw_rate:.1f}°/s "
                f"conf={detector.confidence:.2f}{rr_str}{planner_str}"
            )

        await asyncio.sleep(dt)
