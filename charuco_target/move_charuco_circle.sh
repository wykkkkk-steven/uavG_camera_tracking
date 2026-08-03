#!/usr/bin/env bash
set -euo pipefail

# ================================================================
# 圆周绕行移动脚本
#
# 用法: ./move_charuco_circle.sh [WORLD] [MODEL]
# 示例: ./move_charuco_circle.sh baylands charuco_target
#
# 目标沿水平面圆周轨迹运动，朝向始终指向圆心（即面向追踪无人机）
# ================================================================

WORLD="${1:-baylands}"
MODEL="${2:-charuco_target}"

# ── 圆周参数 ──
CENTER_X=25.0       # 圆心 X
CENTER_Y=-3.0       # 圆心 Y
Z=3.5               # 固定高度
RADIUS=8.0          # 圆周半径（米）
SPEED=0.50          # 线速度（m/s）
DT=0.50             # 步进间隔（秒）

# ── 初始角度 ──
ANGLE=0.0           # 从 0° 开始

# ── 角速度 ──
# ω = v / r（弧度/秒）
OMEGA=$(python3 -c "print(${SPEED} / ${RADIUS})")

echo "[CIRCLE] world=${WORLD}, model=${MODEL}"
echo "[CIRCLE] center=(${CENTER_X}, ${CENTER_Y}), z=${Z}, radius=${RADIUS}m"
echo "[CIRCLE] speed=${SPEED} m/s, omega=${OMEGA} rad/s, dt=${DT}s"
echo "[CIRCLE] Ctrl+C to stop"

while true; do
  # ── 计算圆周上的位置 ──
  RESULT=$(python3 - <<PY
import math
angle = float("${ANGLE}")
cx = float("${CENTER_X}")
cy = float("${CENTER_Y}")
r = float("${RADIUS}")
z = float("${Z}")

x = cx + r * math.cos(angle)
y = cy + r * math.sin(angle)

# 朝向圆心：yaw = atan2(center_y - y, center_x - x)
# GZ 四元数：绕 Z 轴旋转 yaw 角
yaw = math.atan2(cy - y, cx - x)
# 四元数 (x, y, z, w) 绕 Z 轴旋转 yaw
qx = 0.0
qy = 0.0
qz = math.sin(yaw / 2.0)
qw = math.cos(yaw / 2.0)

print(f"{x:.6f} {y:.6f} {z:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}")
PY
)

  X=$(echo "${RESULT}" | awk '{print $1}')
  Y=$(echo "${RESULT}" | awk '{print $2}')
  POS_Z=$(echo "${RESULT}" | awk '{print $3}')
  QX=$(echo "${RESULT}" | awk '{print $4}')
  QY=$(echo "${RESULT}" | awk '{print $5}')
  QZ=$(echo "${RESULT}" | awk '{print $6}')
  QW=$(echo "${RESULT}" | awk '{print $7}')

  # ── 设置模型位姿 ──
  gz service -s "/world/${WORLD}/set_pose" \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 5000 \
    --req "name: \"${MODEL}\" position {x: ${X} y: ${Y} z: ${POS_Z}} orientation {x: ${QX} y: ${QY} z: ${QZ} w: ${QW}}" >/dev/null || true

  # ── 更新角度 ──
  ANGLE=$(python3 -c "
angle = float('${ANGLE}')
omega = float('${OMEGA}')
dt = float('${DT}')
# 归一化到 [0, 2π)
new_angle = (angle + omega * dt) % (2.0 * math.pi)
print(f'{new_angle:.10f}')
")

  sleep "${DT}"
done
