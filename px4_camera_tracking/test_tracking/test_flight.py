"""
飞控
"""

import asyncio
import math
import select
import sys
import time

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityBodyYawspeed

from test_params import (
    UDP_ADDR, TAKEOFF_ALT_M,
    REACHED_XY_THR_M, REACHED_ALT_THR_M, WAIT_REACHED_TIMEOUT_S,
)
from test_utils import (
    clamp, meters_from_gps, distance_ne,
)

async def get_in_air_once(drone, timeout_s=2.0):
    try:
        return await asyncio.wait_for(
            drone.telemetry.in_air().__anext__(), timeout=timeout_s
        )
    except asyncio.TimeoutError:
        return None


async def get_position_once(drone, timeout_s=2.0):
    try:
        return await asyncio.wait_for(
            drone.telemetry.position().__anext__(), timeout=timeout_s
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
            drone.telemetry.heading().__anext__(), timeout=timeout_s
        )
        return heading.heading_deg
    except (asyncio.TimeoutError, Exception):
        return None

async def connect_and_takeoff(drone):
    print(f"[连接] {UDP_ADDR}")
    await drone.connect(system_address=UDP_ADDR)

    print("[连接] 等待无人机就绪...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[连接] 已连接")
            break

    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("[连接] GPS/Home 就绪")
            break

    answer = input("[起飞] 确认起飞? (YES/其他=退出): ").strip()
    if answer != "YES":
        raise RuntimeError("用户取消起飞")

    print("[起飞] Arm...")
    await drone.action.arm()
    await asyncio.sleep(1.0)

    print(f"[起飞] Takeoff → {TAKEOFF_ALT_M}m")
    await drone.action.takeoff()
    await asyncio.sleep(2.0)

    t0 = time.time()
    while time.time() - t0 < 20.0:
        pos = await get_position_once(drone)
        if pos and pos.relative_altitude_m >= TAKEOFF_ALT_M * 0.9:
            print(f"[起飞] 已到达 {pos.relative_altitude_m:.1f}m")
            break
        await asyncio.sleep(0.5)

    await asyncio.sleep(2.0)


async def start_offboard_velocity(drone):
    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    try:
        await drone.offboard.start()
        print("[Offboard] 已进入速度控制模式")
    except OffboardError as e:
        print(f"[Offboard] 启动失败: {e}")
        raise

async def wait_reached_offboard(drone, target_n, target_e, target_d,
                                ref_lat, ref_lon,
                                xy_thr=REACHED_XY_THR_M,
                                alt_thr=REACHED_ALT_THR_M,
                                timeout=WAIT_REACHED_TIMEOUT_S):
    t0 = time.time()
    while True:
        pos = await get_position_once(drone)
        if pos is None:
            await asyncio.sleep(0.2)
            continue

        dn, de = meters_from_gps(ref_lat, ref_lon,
                                  pos.latitude_deg, pos.longitude_deg)
        dd = pos.relative_altitude_m - target_d
        dist_xy = distance_ne(dn, de, target_n, target_e)

        if dist_xy < xy_thr and abs(dd) < alt_thr:
            print(f"[到达] N={dn:.1f} E={de:.1f} alt={pos.relative_altitude_m:.1f}m")
            return True

        if time.time() - t0 > timeout:
            print(f"[到达] 超时 (xy={dist_xy:.1f}m d={dd:.1f}m)")
            return False

        await asyncio.sleep(0.3)

async def hover_wait_for_command(drone, stop_event=None, land_event=None):
    """悬停等待用户输入指令。返回: 'start' / 'home' / 'land'"""
    print("\n[指令] START=开始搜索, home=返航, land=降落, 0=停止")

    while True:
        try:
            await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        except Exception as e:
            print(f"[WARN] 悬停指令失败: {e}")

        if land_event is not None and land_event.is_set():
            return "land"
        if stop_event is not None and stop_event.is_set():
            return "land"

        if select.select([sys.stdin], [], [], 0.0)[0]:
            line = sys.stdin.readline().strip().lower()
            if line == "start":
                return "start"
            elif line == "home":
                return "home"
            elif line in ("land", "stop", "0"):
                land_event.set()
                stop_event.set()
                return "land"

        await asyncio.sleep(0.1)

async def return_home_and_hover(drone, ref_lat, ref_lon, hover_alt=TAKEOFF_ALT_M,
                                timeout_s=WAIT_REACHED_TIMEOUT_S):
    """返回 Home 点上方悬停。超时则停止并打印警告。"""
    print("[返航] 开始返回 Home...")
    deadline = time.time() + timeout_s

    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0.3, 0))
    except Exception as e:
        print(f"[WARN] 爬升指令失败: {e}")
    await asyncio.sleep(3.0)

    while True:
        if time.time() > deadline:
            print("[返航] 超时，停止返航")
            try:
                await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
            except Exception:
                pass
            return

        pos = await get_position_once(drone)
        if pos is None:
            await asyncio.sleep(0.2)
            continue

        dn, de = meters_from_gps(ref_lat, ref_lon,
                                  pos.latitude_deg, pos.longitude_deg)
        dist = distance_ne(0, 0, dn, de)

        if dist < REACHED_XY_THR_M:
            print("[返航] 已到达 Home 上方")
            break

        # 误差 = 目标位置(Home=0,0) - 当前位置
        north_err = 0.0 - dn
        east_err = 0.0 - de

        heading = await get_heading_once(drone)
        if heading is None:
            heading = 0.0

        yaw_rad = math.radians(heading)
        # 速度方向：北东误差 → body 系
        vel_n = north_err / max(dist, 0.1)
        vel_e = east_err / max(dist, 0.1)
        fwd = vel_n * math.cos(yaw_rad) + vel_e * math.sin(yaw_rad)
        rgt = -vel_n * math.sin(yaw_rad) + vel_e * math.cos(yaw_rad)

        speed = clamp(dist * 0.5, 0.3, 2.0)
        fwd_cmd = clamp(fwd * speed, -2.0, 2.0)
        rgt_cmd = clamp(rgt * speed, -2.0, 2.0)

        try:
            await drone.offboard.set_velocity_body(
                VelocityBodyYawspeed(fwd_cmd, rgt_cmd, 0, 0)
            )
        except Exception as e:
            print(f"[WARN] 返航速度指令失败: {e}")

        await asyncio.sleep(0.1)

    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
    except Exception:
        pass  # 安全降落时不需要重试
    print("[返航] 到达 Home，悬停等待指令")

async def safe_stop_and_land(drone, reason=""):
    if reason:
        print(f"[安全] {reason}")
    try:
        await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0, 0, 0, 0))
        await asyncio.sleep(0.3)
    except Exception:
        pass
    try:
        await drone.offboard.stop()
    except OffboardError:
        pass
    try:
        await drone.action.land()
    except Exception as e:
        print(f"[安全] 降落指令异常: {e}")

    for _ in range(60):
        in_air = await get_in_air_once(drone, timeout_s=1.0)
        if in_air is not None and not in_air:
            print("[安全] 已着陆")
            return
        await asyncio.sleep(1.0)
    print("[安全] 等待着陆超时，请手动检查")
