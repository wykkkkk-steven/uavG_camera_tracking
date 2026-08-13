"""ArduCopter 4.6 GUIDED velocity/yaw-rate adapter using pymavlink."""

import math
import threading
import time

try:
    from . import config
except ImportError:  # Allow direct execution for diagnostics.
    import config


PYMAVLINK_ERROR = (
    "pymavlink is not installed. Install it in the Python environment used "
    "to run this program."
)


def _load_mavutil():
    try:
        from pymavlink import mavutil
    except ImportError as exc:
        raise RuntimeError(PYMAVLINK_ERROR) from exc
    return mavutil


def build_velocity_yaw_rate_mask(mavlink):
    """Ignore position, acceleration and yaw; command velocity and yaw rate."""
    return (
        mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
        | mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE
    )


class ArduPilotController:
    """Resend the latest body velocity command at 10 Hz with a watchdog.

    ArduCopter 4.6 accepts SET_POSITION_TARGET_LOCAL_NED in GUIDED mode.
    MAV_FRAME_BODY_NED defines velocity X/Y/Z as aircraft forward/right/down.
    The type mask keeps vx/vy/vz and yaw_rate active while ignoring position,
    acceleration and yaw angle. Active transmission never changes mode, arms,
    takes off, returns home, or lands.
    """

    def __init__(self, dry_run=config.DRY_RUN):
        self.dry_run = bool(dry_run)
        self._connection = None
        self._mavutil = None
        self._target_system = None
        self._target_component = None
        self._last_heartbeat = 0.0
        self._last_command = 0.0
        self._command = (0.0, 0.0, 0.0, 0.0)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._last_fault = None

    def connect(self):
        if self.dry_run:
            print("[AP][DRY-RUN] MAVLink transmission disabled")
            return

        config.validate_live_config()
        self._mavutil = _load_mavutil()
        kwargs = {}
        if config.MAVLINK_BAUD is not None:
            kwargs["baud"] = config.MAVLINK_BAUD
        self._connection = self._mavutil.mavlink_connection(
            config.MAVLINK_CONNECTION,
            **kwargs,
        )
        heartbeat = self._connection.wait_heartbeat(
            timeout=config.HEARTBEAT_TIMEOUT_S
        )
        if heartbeat is None:
            raise RuntimeError("Timed out waiting for an ArduPilot heartbeat.")
        if heartbeat.autopilot != self._mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            raise RuntimeError(
                "Heartbeat is not from an ArduPilot autopilot "
                f"(autopilot={heartbeat.autopilot})."
            )
        self._target_system = self._connection.target_system
        self._target_component = self._connection.target_component
        self._last_heartbeat = time.monotonic()
        print(
            f"[AP] heartbeat system={self._target_system} "
            f"component={self._target_component}"
        )

    def start(self):
        if not self.dry_run and self._connection is None:
            raise RuntimeError("Call connect() before start().")
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._send_loop,
            name="ardupilot-command-loop",
            daemon=True,
        )
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._send_zero_if_allowed()

    def update_command(self, forward_m_s, right_m_s, down_m_s, yaw_rate_rad_s):
        command = (
            max(-config.MAX_BACKWARD_M_S, min(config.MAX_FORWARD_M_S, float(forward_m_s))),
            max(-config.MAX_RIGHT_M_S, min(config.MAX_RIGHT_M_S, float(right_m_s))),
            max(-config.MAX_DOWN_M_S, min(config.MAX_DOWN_M_S, float(down_m_s))),
            max(-config.MAX_YAW_RATE_RAD_S, min(config.MAX_YAW_RATE_RAD_S, float(yaw_rate_rad_s))),
        )
        if not all(math.isfinite(value) for value in command):
            command = (0.0, 0.0, 0.0, 0.0)
            print("[AP][SAFE] non-finite command rejected; holding")
        with self._lock:
            self._command = command
            self._last_command = time.monotonic()

    def _send_loop(self):
        period = 1.0 / config.COMMAND_SEND_HZ
        while not self._stop.is_set():
            cycle_start = time.monotonic()
            self._poll_heartbeat()
            with self._lock:
                fresh = cycle_start - self._last_command <= config.COMMAND_TIMEOUT_S
                command = self._command if fresh else (0.0, 0.0, 0.0, 0.0)
            self._send(command)
            self._stop.wait(max(0.0, period - (time.monotonic() - cycle_start)))

    def _poll_heartbeat(self):
        if self.dry_run:
            return
        while True:
            heartbeat = self._connection.recv_match(type="HEARTBEAT", blocking=False)
            if heartbeat is None:
                break
            if heartbeat.get_srcSystem() == self._target_system:
                self._last_heartbeat = time.monotonic()

    def _guided_mode_active(self):
        if self.dry_run:
            return True
        if time.monotonic() - self._last_heartbeat > config.HEARTBEAT_TIMEOUT_S:
            self._report_fault("MAVLink heartbeat lost; active commands stopped")
            return False
        mapping = self._connection.mode_mapping() or {}
        guided_id = mapping.get(config.REQUIRED_MODE)
        if guided_id is None:
            self._report_fault("GUIDED mode is not present in the vehicle mode mapping")
            return False
        heartbeat = self._connection.messages.get("HEARTBEAT")
        if heartbeat is None or heartbeat.custom_mode != guided_id:
            self._report_fault("Vehicle is not in GUIDED; active commands suppressed")
            return False
        self._last_fault = None
        return True

    def _report_fault(self, message):
        if message != self._last_fault:
            print(f"[AP][SAFE] {message}")
            self._last_fault = message

    def _send(self, command):
        if self.dry_run:
            forward, right, down, yaw_rate = command
            print(
                "[AP][DRY-RUN] "
                f"F={forward:+.2f} R={right:+.2f} D={down:+.2f} "
                f"Y={yaw_rate:+.3f}rad/s"
            )
            return
        if not self._guided_mode_active():
            return

        mavlink = self._mavutil.mavlink
        type_mask = build_velocity_yaw_rate_mask(mavlink)
        self._connection.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000) & 0xFFFFFFFF,
            self._target_system,
            self._target_component,
            mavlink.MAV_FRAME_BODY_NED,
            type_mask,
            0.0,
            0.0,
            0.0,
            command[0],
            command[1],
            command[2],
            0.0,
            0.0,
            0.0,
            0.0,
            command[3],
        )

    def _send_zero_if_allowed(self):
        if self.dry_run:
            print("[AP][DRY-RUN] F=+0.00 R=+0.00 D=+0.00 Y=+0.000rad/s")
        elif self._connection is not None and self._guided_mode_active():
            self._send((0.0, 0.0, 0.0, 0.0))
