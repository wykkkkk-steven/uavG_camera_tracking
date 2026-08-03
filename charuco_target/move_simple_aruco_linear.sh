#!/usr/bin/env bash
set -euo pipefail

WORLD="${1:-baylands}"
MODEL="${2:-simple_aruco_target}"

X=8.0
Z=1.5
Y_MIN=14.0
Y_MAX=43.0
SPEED=1.20
DT=0.50

# Keep same orientation as the recommended spawn command.
ORIENTATION='orientation {y: 0.7071 w: 0.7071}'

echo "[MOVE] world=${WORLD}, model=${MODEL}"
echo "[MOVE] y=${Y_MIN} to y=${Y_MAX}, speed=${SPEED} m/s, dt=${DT}s"
echo "[MOVE] Ctrl+C to stop"

Y="${Y_MIN}"
DIR=1

while true; do
  gz service -s "/world/${WORLD}/set_pose" \
    --reqtype gz.msgs.Pose \
    --reptype gz.msgs.Boolean \
    --timeout 5000 \
    --req "name: \"${MODEL}\" position {x: ${X} y: ${Y} z: ${Z}} ${ORIENTATION}" >/dev/null || true

  Y_NEXT=$(python3 - <<PY
y = float("${Y}")
dir_ = int("${DIR}")
speed = float("${SPEED}")
dt = float("${DT}")
print(y + dir_ * speed * dt)
PY
)

  CHECK=$(python3 - <<PY
y = float("${Y_NEXT}")
ymin = float("${Y_MIN}")
ymax = float("${Y_MAX}")
print("ok" if ymin <= y <= ymax else "flip")
PY
)

  if [ "${CHECK}" = "flip" ]; then
    if [ "${DIR}" = "1" ]; then
      DIR=-1
      Y="${Y_MAX}"
    else
      DIR=1
      Y="${Y_MIN}"
    fi
  else
    Y="${Y_NEXT}"
  fi

  sleep "${DT}"
done
