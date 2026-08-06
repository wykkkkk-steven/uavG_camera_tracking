import asyncio
from mavsdk import System


TARGET_MAVLINK_PORT = 14541

# 必须与 YOLO 脚本使用不同的 mavsdk_server gRPC 端口
TARGET_MAVSDK_SERVER_PORT = 50052


async def main():
    # port=50052 是本地 mavsdk_server 的 gRPC 端口
    # sysid=246 是 mavsdk_server 自身的 MAVLink system ID，
    # 不是目标飞机的 MAV_SYS_ID。
    target = System(
        port=TARGET_MAVSDK_SERVER_PORT,
        sysid=246,
    )

    print(
        f"Connecting to target drone: "
        f"MAVLink UDP {TARGET_MAVLINK_PORT}, "
        f"MAVSDK server {TARGET_MAVSDK_SERVER_PORT}"
    )

    await target.connect(
        system_address=f"udpin://0.0.0.0:{TARGET_MAVLINK_PORT}"
    )

    print("Waiting for target drone connection...")

    async for state in target.core.connection_state():
        if state.is_connected:
            print("Target drone connected on UDP 14541")
            break

    print("Waiting for local position and armable state...")

    async for health in target.telemetry.health():
        if health.is_local_position_ok and health.is_armable:
            print("Target drone is ready")
            break

    await target.action.set_takeoff_altitude(5.0)

    print("Arming target drone...")
    await target.action.arm()

    print("Taking off target drone...")
    await target.action.takeoff()

    print("Target drone taking off and holding at 5 m")

    try:
        while True:
            async for position in target.telemetry.position():
                print(
                    f"[TARGET] altitude="
                    f"{position.relative_altitude_m:.1f} m"
                )
                break

            await asyncio.sleep(1.0)

    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    asyncio.run(main())
