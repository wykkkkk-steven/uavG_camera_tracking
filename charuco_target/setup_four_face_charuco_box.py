#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


def get_aruco_dictionary(name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Use OpenCV with opencv_contrib.")

    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown ArUco dictionary: {name}")

    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def generate_marker(dictionary, marker_id: int, size_px: int):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)

    return cv2.aruco.drawMarker(dictionary, marker_id, size_px)


def make_charuco_style_board(
    dictionary,
    start_id: int,
    squares_x: int,
    squares_y: int,
    square_px: int,
    marker_ratio: float,
):
    """
    Generate a ChArUco-style board texture manually:
    - checkerboard background
    - ArUco markers centered in white squares
    - sequential marker IDs starting from start_id

    The flight code only needs cv2.aruco.detectMarkers(), so this is enough and
    more controllable than depending on OpenCV CharucoBoard API differences.
    """
    width = squares_x * square_px
    height = squares_y * square_px

    img = np.full((height, width), 255, dtype=np.uint8)

    marker_px = int(square_px * marker_ratio)
    marker_px = max(20, min(marker_px, square_px - 8))
    offset = (square_px - marker_px) // 2

    marker_id = start_id
    used_ids = []

    for y in range(squares_y):
        for x in range(squares_x):
            x0 = x * square_px
            y0 = y * square_px

            # Black chess squares.
            if (x + y) % 2 == 1:
                img[y0:y0 + square_px, x0:x0 + square_px] = 0
                continue

            # White square with centered ArUco marker.
            marker = generate_marker(dictionary, marker_id, marker_px)

            mx0 = x0 + offset
            my0 = y0 + offset
            img[my0:my0 + marker_px, mx0:mx0 + marker_px] = marker

            used_ids.append(marker_id)
            marker_id += 1

    # Add a white margin around the whole board. This helps marker detection.
    margin = max(20, square_px // 4)
    img = cv2.copyMakeBorder(
        img,
        margin,
        margin,
        margin,
        margin,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return img, used_ids


def write_mtl(mesh_dir: Path):
    lines = []

    for face in ("front", "back", "left", "right"):
        lines += [
            f"newmtl {face}_mat",
            "Ka 1.000 1.000 1.000",
            "Kd 1.000 1.000 1.000",
            "Ks 0.000 0.000 0.000",
            "d 1.0",
            "illum 1",
            f"map_Kd charuco_{face}.png",
            "",
        ]

    (mesh_dir / "charuco_box.mtl").write_text("\n".join(lines), encoding="utf-8")


def write_obj(mesh_dir: Path, width_m: float, depth_m: float, height_m: float):
    hx = width_m / 2.0
    hy = depth_m / 2.0
    hz = height_m / 2.0

    # Coordinate convention:
    # X front/back, Y left/right, Z up/down.
    # Four vertical faces only.
    obj = f"""mtllib charuco_box.mtl
o four_face_charuco_box

# front face, +X
v {hx:.6f} {-hy:.6f} {-hz:.6f}
v {hx:.6f} {hy:.6f} {-hz:.6f}
v {hx:.6f} {hy:.6f} {hz:.6f}
v {hx:.6f} {-hy:.6f} {hz:.6f}

# back face, -X
v {-hx:.6f} {hy:.6f} {-hz:.6f}
v {-hx:.6f} {-hy:.6f} {-hz:.6f}
v {-hx:.6f} {-hy:.6f} {hz:.6f}
v {-hx:.6f} {hy:.6f} {hz:.6f}

# left face, +Y
v {hx:.6f} {hy:.6f} {-hz:.6f}
v {-hx:.6f} {hy:.6f} {-hz:.6f}
v {-hx:.6f} {hy:.6f} {hz:.6f}
v {hx:.6f} {hy:.6f} {hz:.6f}

# right face, -Y
v {-hx:.6f} {-hy:.6f} {-hz:.6f}
v {hx:.6f} {-hy:.6f} {-hz:.6f}
v {hx:.6f} {-hy:.6f} {hz:.6f}
v {-hx:.6f} {-hy:.6f} {hz:.6f}

# texture coordinates
vt 0 1
vt 1 1
vt 1 0
vt 0 0

# normals
vn 1 0 0
vn -1 0 0
vn 0 1 0
vn 0 -1 0

usemtl front_mat
f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1

usemtl back_mat
f 5/1/2 6/2/2 7/3/2
f 5/1/2 7/3/2 8/4/2

usemtl left_mat
f 9/1/3 10/2/3 11/3/3
f 9/1/3 11/3/3 12/4/3

usemtl right_mat
f 13/1/4 14/2/4 15/3/4
f 13/1/4 15/3/4 16/4/4
"""

    (mesh_dir / "charuco_box.obj").write_text(obj, encoding="utf-8")


def write_sdf(model_dir: Path, width_m: float, depth_m: float, height_m: float):
    sdf = f"""<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="four_face_charuco_box">
    <static>true</static>

    <link name="box_link">
      <visual name="box_visual">
        <geometry>
          <mesh>
            <uri>model://four_face_charuco_box/meshes/charuco_box.obj</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>

      <collision name="box_collision">
        <geometry>
          <box>
            <size>{width_m:.4f} {depth_m:.4f} {height_m:.4f}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
"""
    (model_dir / "model.sdf").write_text(sdf, encoding="utf-8")


def write_config(model_dir: Path):
    config = """<?xml version="1.0"?>
<model>
  <name>four_face_charuco_box</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>local</name>
  </author>
  <description>Four-face ChArUco-style box target for Gazebo Sim.</description>
</model>
"""
    (model_dir / "model.config").write_text(config, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-dir",
        default=str(Path.home() / ".gz" / "models" / "four_face_charuco_box"),
    )

    # DICT_4X4_100 is the default because four 5x5 boards need more than 50 IDs.
    parser.add_argument("--dict", default="DICT_4X4_100")

    # Physical box dimensions.
    parser.add_argument("--width-m", type=float, default=1.6)
    parser.add_argument("--depth-m", type=float, default=1.6)
    parser.add_argument("--height-m", type=float, default=1.6)

    # Board texture settings.
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=5)
    parser.add_argument("--square-px", type=int, default=240)
    parser.add_argument("--marker-ratio", type=float, default=0.72)
    parser.add_argument(
        "--no-mirror-compensation",
        action="store_true",
        help="Disable horizontal texture flip compensation. Use only if the board looks mirrored after testing.",
    )

    # Marker ID start for each face.
    parser.add_argument("--front-start-id", type=int, default=0)
    parser.add_argument("--back-start-id", type=int, default=20)
    parser.add_argument("--left-start-id", type=int, default=40)
    parser.add_argument("--right-start-id", type=int, default=60)

    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    mesh_dir = model_dir / "meshes"
    texture_dir = model_dir / "materials" / "textures"

    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_aruco_dictionary(args.dict)

    face_start_ids = {
        "front": args.front_start_id,
        "back": args.back_start_id,
        "left": args.left_start_id,
        "right": args.right_start_id,
    }

    print("Generating ChArUco-style textures...")

    all_used_ids = []

    for face, start_id in face_start_ids.items():
        board_img, used_ids = make_charuco_style_board(
            dictionary=dictionary,
            start_id=start_id,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_px=args.square_px,
            marker_ratio=args.marker_ratio,
        )

        # Gazebo/OBJ UV mapping can display outside-facing textures mirrored.
        # ArUco/ChArUco markers cannot be detected after mirror reflection.
        # Pre-flip the texture so the board seen by the camera is not mirrored.
        if not args.no_mirror_compensation:
            board_img = cv2.flip(board_img, 1)

        cv2.imwrite(str(mesh_dir / f"charuco_{face}.png"), board_img)
        cv2.imwrite(str(texture_dir / f"charuco_{face}.png"), board_img)

        all_used_ids.extend(used_ids)
        print(f"  {face}: IDs {used_ids[0]} to {used_ids[-1]}")

    write_mtl(mesh_dir)
    write_obj(mesh_dir, args.width_m, args.depth_m, args.height_m)
    write_sdf(model_dir, args.width_m, args.depth_m, args.height_m)
    write_config(model_dir)

    print("")
    print("Generated four-face ChArUco-style box target:")
    print(model_dir)
    print("")
    print("Box size:")
    print(f"  width  X: {args.width_m:.2f} m")
    print(f"  depth  Y: {args.depth_m:.2f} m")
    print(f"  height Z: {args.height_m:.2f} m")
    print("")
    print("Dictionary:", args.dict)
    print("Mirror compensation:", not args.no_mirror_compensation)
    print("Total marker IDs used:", len(all_used_ids))
    print("")
    print("IMPORTANT detector settings in visual_orbit_node.py:")
    print("  ARUCO_DICTIONARY = cv2.aruco." + args.dict)
    print("  TARGET_MARKER_ID = None")
    print("  MIN_MARKER_AREA = 150.0")
    print("")
    print("Remove old model if needed:")
    print(
        "gz service -s /world/baylands/remove "
        "--reqtype gz.msgs.Entity "
        "--reptype gz.msgs.Boolean "
        "--timeout 300 "
        "--req 'name: \"four_face_charuco_box\"'"
    )
    print("")
    print("Spawn command:")
    print(
        "gz service -s /world/baylands/create "
        "--reqtype gz.msgs.EntityFactory "
        "--reptype gz.msgs.Boolean "
        "--timeout 300 "
        "--req 'sdf_filename: \"" + str(model_dir / "model.sdf") + "\" "
        "name: \"four_face_charuco_box\" "
        "pose {position {x: 8 y: 0 z: 1.5} orientation {w: 1}}'"
    )


if __name__ == "__main__":
    main()
