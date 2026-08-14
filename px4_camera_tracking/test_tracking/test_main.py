"""
主入口
"""

import asyncio
import math
import signal
import sys

import rclpy
from mavsdk import System

from .test_params import (
    DISTANCE_DEFAULT_M, DISTANCE_MAX_VALID_M,
    PLANNER_ENABLED,
)
from .test_utils import ask_float, read_console_line
from .test_flight import (
    connect_and_takeoff,
    start_offboard_velocity,
    hover_wait_for_command,
    return_home_and_hover,
    safe_stop_and_land,
)
from .test_detector import YOLODetectorNode, spin_ros_node
from .test_tracker import (
    search_until_target_found,
    visual_tracking_control,
)
from .test_planner import TargetPositionEstimator

async def command_listener(stop_event, land_event):
    import select

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

async def ask_track_duration(stop_event, land_event):
    print("\n[目标已找到] 输入跟踪飞行时长（秒），例如 60")
    print("不想继续跟踪，输入 land")

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
            print("请输入数字，例如 60；或者输入 land")


async def ask_desired_distance(stop_event, land_event):
    value = await ask_float(
        "请输入期望跟踪距离 desired_distance_m，单位 m，例如 3.5",
        default_value=DISTANCE_DEFAULT_M,
        min_value=1.0,
        max_value=DISTANCE_MAX_VALID_M,
        stop_event=stop_event,
    )
    if value is None:
        land_event.set()
        return None
    return value

async def main_async():
    rclpy.init()

    detector = YOLODetectorNode()
    drone = System()

    # Planner 初始化
    planner = TargetPositionEstimator(DISTANCE_DEFAULT_M) if PLANNER_ENABLED else None

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
        await connect_and_takeoff(drone)
        flight_started = True

        await start_offboard_velocity(drone)

        try:
            pos = await asyncio.wait_for(
                drone.telemetry.position().__anext__(), timeout=10.0
            )
        except asyncio.TimeoutError:
            print("[ERROR] 等待位置数据超时，无法获取 Home 参考点")
            raise
        ref_lat, ref_lon = pos.latitude_deg, pos.longitude_deg
        print(
            f"[HOME参考点] lat={pos.latitude_deg:.7f} lon={pos.longitude_deg:.7f} "
            f"abs_alt={pos.absolute_altitude_m:.2f}m rel_alt={pos.relative_altitude_m:.2f}m"
        )

        while not stop_event.is_set() and not land_event.is_set():

            action = await hover_wait_for_command(
                drone, stop_event, land_event,
            )

            if action == "home":
                await return_home_and_hover(drone, ref_lat, ref_lon)
                continue
            if action != "start":
                break

            cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

            # 位置提供器：供 planner 使用
            # pipeline 模式：每帧调用 ensure_future 触发异步更新，
            # 读上一次更新的结果（延迟一帧≈100ms，10Hz下可接受）
            _pos_cache = {"n": 0.0, "e": 0.0, "h": 0.0, "pending": False}

            async def _update_pos_cache():
                try:
                    pos = await drone.telemetry.position().__anext__()
                    heading = await drone.telemetry.heading().__anext__()
                    dlat = pos.latitude_deg - ref_lat
                    dlon = pos.longitude_deg - ref_lon
                    _pos_cache["n"] = dlat * 111320.0
                    _pos_cache["e"] = dlon * 111320.0 * math.cos(math.radians(ref_lat))
                    _pos_cache["h"] = heading.heading_deg
                except Exception:
                    pass
                finally:
                    _pos_cache["pending"] = False

            def drone_position_provider():
                """同步回调，返回 (north_m, east_m, heading_deg)。
                每次调用触发一次异步更新，返回上次更新结果。"""
                if not _pos_cache["pending"]:
                    _pos_cache["pending"] = True
                    asyncio.ensure_future(_update_pos_cache())
                return _pos_cache["n"], _pos_cache["e"], _pos_cache["h"]

            detector.reset_tracking_state()

            search_result = await search_until_target_found(
                drone, detector, stop_event, land_event,
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

            if search_result != "found":
                print("[WAIT] 搜索未找到目标，返回悬停等待")
                continue

            track_duration_s = await ask_track_duration(stop_event, land_event)
            if track_duration_s is None:
                if land_event.is_set():
                    break
                continue

            desired_distance_m = await ask_desired_distance(stop_event, land_event)
            if desired_distance_m is None:
                if land_event.is_set():
                    break
                continue

            cmd_task = asyncio.create_task(command_listener(stop_event, land_event))

            # 如果 planner 启用，更新期望距离并重置
            if planner is not None:
                planner._desired_distance_m = desired_distance_m
                planner.reset()

            track_result = await visual_tracking_control(
                drone,
                detector,
                desired_distance_m,
                track_duration_s,
                stop_event,
                land_event,
                planner=planner,
                drone_position_provider=drone_position_provider,
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

            print(f"[WAIT] 跟踪结束原因: {track_result}，返回悬停等待")

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
