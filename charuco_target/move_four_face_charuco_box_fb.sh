#!/usr/bin/env bash
set -euo pipefail

WORLD="${1:-baylands}"
MODEL="${2:-four_face_charuco_box}"

Y=0.0
Z=3.5
X_MIN=15.0
X_MAX=40.0
SPEED=1.50
DT=0.60

# The box is upright internally.
ORIENTATION='orientation {w: 1}'

echo "[MOVE FB] world=${WORLD}, model=${MODEL}"
echo "[MOVE FB] x=${X_MIN} to ${X_MAX}, y=${Y}, z=${Z}, speed=${SPEED} m/s"
echo "[MOVE FB] Ctrl+C to stop"

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
