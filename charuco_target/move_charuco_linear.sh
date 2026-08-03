#!/usr/bin/env bash
set -euo pipefail

WORLD="${1:-baylands}"
MODEL="${2:-charuco_target}"

Y=-3.0
Z=3.5
X_MIN=8.0
X_MAX=45.0
SPEED=0.40
DT=0.60

# Keep board upright.
ORIENTATION='orientation {y: 0.7071 w: 0.7071}'

# If your drone views the board from the X direction, use this instead:
# ORIENTATION='orientation {z: 0.7071 w: 0.7071}'

echo "[MOVE] world=${WORLD}, model=${MODEL}"
echo "[MOVE] x=${X_MIN} to x=${X_MAX}, speed=${SPEED} m/s, dt=${DT}s"
echo "[MOVE] Ctrl+C to stop"

X="${X_MIN}"
DIR=1

while true; do
  gz service -s "/world/${WORLD}/set_pose" \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 5000 \
    --req "name: \"${MODEL}\" position {x: ${X} y: ${Y} z: ${Z}} ${ORIENTATION}" >/dev/null || true

  X_NEXT=$(python3 - <<PY
x = float("${X}")
dir_ = int("${DIR}")
speed = float("${SPEED}")
dt = float("${DT}")
print(x + dir_ * speed * dt)
PY
)

  CHECK=$(python3 - <<PY
x = float("${X_NEXT}")
xmin = float("${X_MIN}")
xmax = float("${X_MAX}")
print("ok" if xmin <= x <= xmax else "flip")
PY
)

  if [ "${CHECK}" = "flip" ]; then
    if [ "${DIR}" = "1" ]; then
      DIR=-1
      X="${X_MAX}"
    else
      DIR=1
      X="${X_MIN}"
    fi
  else
    X="${X_NEXT}"
  fi

  sleep "${DT}"
done

