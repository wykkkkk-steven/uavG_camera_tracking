"""
五面 ArUco 正方体面关系配置（绝对方向方案）

实体：无底板五面正方体（小车搭载，仅作测试平台）
- 顶面: ID 0（不参与水平方向选择）
- 四个侧面: ID 1, 2, 3, 4
- 俯视顺时针环绕顺序: 1 -> 4 -> 3 -> 2 -> 1
- 对面关系: 1 <-> 3, 2 <-> 4
- 侧面临边（站在该面外侧、面向该面）:
    left  = 逆时针相邻面
    right = 顺时针相邻面
  由用户给定: 2 的左边是 3，右边是 1
- 侧面到顶面: up = 0；顶面到侧面: down

目标面: 运行时锁定第一个检测到的侧面（不在配置中写死）。
TARGET_RELATION 由 face_graph 在锁定时预计算为静态映射表。

板子尺寸: 板 390x390 mm, ArUco 320x320 mm, 白边 35 mm
markerLength = 0.320 m
"""

TOP_FACE_ID = 0

FACE_GRAPH = {
    0: {"down": [1, 2, 3, 4]},
    1: {"left": 2, "right": 4, "up": 0, "opposite": 3},
    2: {"left": 3, "right": 1, "up": 0, "opposite": 4},
    3: {"left": 4, "right": 2, "up": 0, "opposite": 1},
    4: {"left": 1, "right": 3, "up": 0, "opposite": 2},
}

BOARD_SIZE_M = 0.390
MARKER_SIZE_M = 0.320
WHITE_BORDER_M = 0.035
MARKER_LENGTH_M = 0.320

# 目标面锁定后确认/丢失帧数
TARGET_CONFIRM_FRAMES = 3
LOST_CONFIRM_FRAMES = 5

# 当前面切换滞回：新候选面积必须超过当前面面积 * 该阈值才切换
SWITCH_THRESHOLD = 1.2
