文件说明

test_tracking/
├── test_main.py       ← 主入口
├── test_params.py     ← 所有参数，调参只改这里
├── test_utils.py      ← 工具函数
├── test_flight.py     ← 飞控
├── test_detector.py   ← YOLO检测 + bbox张角法测距 + Looming测速
└── test_tracker.py    ← 搜索 + 远距接近 + 精细跟踪状态机


运行前必改的 3 个参数
test_params.py

1. YOLO 模型路径
YOLO_MODEL_PATH = "REPLACE_WITH_YOUR_MODEL.pt改成训练好的模型路径

2. 目标类别 ID
当前默认：接受所有类别
YOLO_TARGET_CLASSES = []
 改成模型中无人机对应的类别 ID

3. 目标尺寸表
张角法测距的核心：告诉系统目标无人机在不同朝向下有多大。需要从仿真中量测校准。
# 默认值（x500 大致估计，可能不准）
VIEW_SIZE_TABLE = {
    0.0:  (0.18, 0.22),    # 宽高比 ≈ 0.82 → 纯侧面（宽,高）单位：米
    1.0:  (0.26, 0.26),    # 宽高比 ≈ 1.0  → 俯视
    2.4:  (0.52, 0.22),    # 宽高比 ≈ 2.4  → 正面/背面
}
DEFAULT_TARGET_SIZE = (0.35, 0.22)   # 兜底默认值

校准：在仿真中把目标机放在已知距离（比如 5m），用相机截图量 YOLO bbox 像素宽高，反算真实尺寸：
真实宽度 = bbox_像素宽 × 距离 / fx
真实高度 = bbox_像素高 × 距离 / fy

其中 fx/fy 是相机内参（从 `/camera/camera_info` topic 获取）。分别在侧面、俯视、正面三个朝向量一次即可。

常用调参

都在 `test_params.py` 里，按需改：

| 需求 | 改什么 |
|------|--------|
| 跟踪距离太近/太远 | `DISTANCE_DEFAULT_M`（运行时也会问） |
| 跟踪抖动 | 减小 `KP_DISTANCE_*` 或增大 `*_SLEW_*` |
| 搜索转太慢 | 增大 `SEARCH_YAW_RATE_DEG_S` |
| 远距识别不到 | 检查 YOLO 模型和 `YOLO_CONF_THRESHOLD` |
| 距离估不准 | 校准 `VIEW_SIZE_TABLE` |
| URGENT 频繁触发 | 增大 `URGENT_TOO_CLOSE_MARGIN_M` |
| 目标丢失太快降落 | 增大 `TARGET_LOST_LAND_S` |
| 远距接近太慢 | 增大 `APPROACH_FORWARD_SPEED` |