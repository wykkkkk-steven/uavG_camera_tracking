"""
全部参数

职责：所有调参都在这一个文件里，其余模块 import 引用
"""

# ================================================================
# 基本连接参数
# ================================================================
UDP_ADDR = "udpin://0.0.0.0:14540"
IMAGE_TOPIC = "/camera/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera_info"
TAKEOFF_ALT_M = 5.0

# ================================================================
# 搜索参数
# ================================================================
CONTROL_HZ = 10.0
SEARCH_YAW_RATE_DEG_S = 5.0
SEARCH_TIMEOUT_S = 80.0
SEARCH_CONFIRM_MIN_FRAMES = 2

# 目标丢失保护
TARGET_LOST_HOVER_S = 2.0
TARGET_LOST_LAND_S = 20.0

# ================================================================
# 远距识别 → 接近参数（50m 识别后向目标飞近）
# ================================================================
APPROACH_YAW_RATE_DEG_S = 10.0     # 远距接近时 yaw 限速
APPROACH_FORWARD_SPEED = 1.5       # 远距接近时前进速度
APPROACH_ENTER_RANGE_M = 7.5       # 进入此距离后切换为精细跟踪

# ================================================================
# 距离分层参数
# ================================================================
DISTANCE_MIN_SAFE_M = 1.0          # 最低安全距离
DISTANCE_OPTIMAL_MIN_M = 2.0       # 最佳区间下限
DISTANCE_OPTIMAL_MAX_M = 5.0       # 最佳区间上限
DISTANCE_MAX_VALID_M = 7.5         # 最大有效追踪距离
DISTANCE_DEFAULT_M = 3.5           # 默认期望距离

# ================================================================
# 距离控制参数
# ================================================================
KP_DISTANCE_CLOSE = 0.20           # <2m
KP_DISTANCE_OPTIMAL = 0.12         # 2-5m
KP_DISTANCE_FAR = 0.08             # >5m
DISTANCE_DEADBAND_M = 0.5

# 速度限幅
MAX_FORWARD_SPEED = 2.0
MAX_BACKWARD_SPEED = 1.8
MAX_TRACK_FORWARD_SPEED = 0.75
MAX_TRACK_BACKWARD_SPEED = 0.75

# ================================================================
# yaw 控制参数
# ================================================================
KP_BEARING_YAW = 1.10
BEARING_DEADBAND_DEG = 1.5

# yaw 限速分级
MAX_YAW_RATE_CLOSE_DEG_S = 60.0    # <2m
MAX_YAW_RATE_OPTIMAL_DEG_S = 45.0  # 2-5m
MAX_YAW_RATE_FAR_DEG_S = 25.0      # >5m

# ================================================================
# 加速度限幅
# ================================================================
FORWARD_SLEW_CLOSE_M_S2 = 0.50    # <2m
FORWARD_SLEW_OPTIMAL_M_S2 = 0.90  # 2-5m
FORWARD_SLEW_FAR_M_S2 = 1.20     # >5m

YAW_SLEW_CLOSE_DEG_S2 = 15.0     # <2m
YAW_SLEW_OPTIMAL_DEG_S2 = 30.0   # 2-5m
YAW_SLEW_FAR_DEG_S2 = 40.0       # >5m

# ================================================================
# ALIGN 参数
# ================================================================
ALIGN_BEARING_DEG = 2.0
ALIGN_CONFIRM_FRAMES = 5
REALIGN_BEARING_DEG = 10.0

# ================================================================
# 到达判定（HOLD）
# ================================================================
DISTANCE_TOLERANCE_M = 0.50
BEARING_TOLERANCE_DEG = 2.0
ARRIVAL_CONFIRM_FRAMES = 5

# ================================================================
# URGENT 后退
# ================================================================
URGENT_TOO_CLOSE_MARGIN_M = 0.80
URGENT_CLOSING_SPEED_M_S = 0.70
URGENT_CONFIRM_FRAMES = 3
URGENT_RETREAT_MIN_SPEED = 0.20
URGENT_RETREAT_MAX_SPEED = 0.75
URGENT_RETREAT_KP = 0.55

# ================================================================
# YOLO 检测参数
# ================================================================
YOLO_MODEL_PATH = "/home/xueyang/uav_yolo_runs/x500_2872_yolo11s_1920-3/weights/best.pt"  # 必须替换！默认值会导致加载失败
YOLO_CONF_THRESHOLD = 0.5
YOLO_IOU_THRESHOLD = 0.45
YOLO_TARGET_CLASSES = []         # 可调，选了全部接受

# ================================================================
# 张角法测距参数（x500 无人机已知尺寸）
# ================================================================
VIEW_SIZE_TABLE = {
    0.0:  (0.18, 0.22),           # 纯侧面
    1.0:  (0.26, 0.26),           # 俯视
    2.4:  (0.52, 0.22),           # 正面/背面
}
DEFAULT_TARGET_SIZE = (0.35, 0.22)

# ================================================================
# Looming 参数
# ================================================================
LOOMING_ALPHA = 0.3
LOOMING_MIN_AREA = 50.0

# ================================================================
# PREVIEW 参数（面积突变 → 预瞄模式）
# ================================================================
AREA_JUMP_RATIO = 1.5              # 面积1帧内变化超过此比例 → 进入PREVIEW
PREVIEW_MAX_DURATION_S = 1.5       # PREVIEW 最长持续时间
PREVIEW_RECOVER_RATIO = 0.30      # 面积回到 ±30% 内 → 退出PREVIEW

# ================================================================
# 丢失搜索
# ================================================================
POSE_LOST_HOVER_S = 0.60
LOST_YAW_SEARCH_RATE_DEG_S = 8.0

# ================================================================
# Home / hover 参数（与 uavG 一致）
# ================================================================
REACHED_XY_THR_M = 1.0
REACHED_ALT_THR_M = 0.5
WAIT_REACHED_TIMEOUT_S = 45.0

# ================================================================
# B 版位置跟踪参数（保留，与 uavG 一致，后续可用）
# ================================================================
TARGET_POSITION_FILTER_ALPHA = 0.30
TARGET_PREDICT_TIME_S = 0.25
MAX_PREDICT_SPEED_M_S = 1.20
POSITION_SETPOINT_UPDATE_S = 0.30
POSITION_UPDATE_DISTANCE_M = 0.30
YAW_UPDATE_DEG = 5.0
MAX_POSITION_SETPOINT_STEP_M = 1.20
POSITION_TRACK_HOLD_AFTER_TARGET_LOST_S = 1.5
