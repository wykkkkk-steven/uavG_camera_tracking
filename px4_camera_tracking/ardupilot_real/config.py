"""Configuration for the real ArduCopter tracking implementation."""

# TODO(real hardware): configure the actual serial device, USB serial port, or
# mavlink-router UDP endpoint. This placeholder is never passed to pymavlink.
MAVLINK_CONNECTION = "TODO_SET_REAL_ARDUPILOT_CONNECTION"
MAVLINK_BAUD = None

# TODO(real hardware): configure the topics from the calibrated camera driver.
IMAGE_TOPIC = "TODO_SET_REAL_CAMERA_IMAGE_TOPIC"
CAMERA_INFO_TOPIC = "TODO_SET_REAL_CAMERA_INFO_TOPIC"

# TODO(real hardware): insert the measured camera-optical-frame to aircraft-body
# rotation. None deliberately prevents pretending the physical mount is known.
CAMERA_TO_BODY_ROTATION = None

# Safe validation default: commands are logged and never transmitted.
DRY_RUN = True
COMMAND_SEND_HZ = 10.0
COMMAND_TIMEOUT_S = 0.5
HEARTBEAT_TIMEOUT_S = 3.0

# Second-layer adapter clamps. These do not increase tracker limits.
MAX_FORWARD_M_S = 0.75
MAX_BACKWARD_M_S = 0.75
MAX_RIGHT_M_S = 0.30
MAX_DOWN_M_S = 0.40
MAX_YAW_RATE_RAD_S = 0.2617993877991494  # 15 deg/s

REQUIRED_MODE = "GUIDED"


def validate_camera_config():
    missing = []
    if IMAGE_TOPIC.startswith("TODO_"):
        missing.append("IMAGE_TOPIC")
    if CAMERA_INFO_TOPIC.startswith("TODO_"):
        missing.append("CAMERA_INFO_TOPIC")
    if missing:
        raise RuntimeError(
            "Configure the real camera interfaces before running: "
            + ", ".join(missing)
        )


def validate_live_config():
    if MAVLINK_CONNECTION.startswith("TODO_"):
        raise RuntimeError(
            "Configure MAVLINK_CONNECTION for the real ArduPilot link before "
            "disabling dry-run mode."
        )
