#!/usr/bin/env python3
import argparse
from pathlib import Path
import cv2


def get_aruco_dictionary(name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco is missing. Build/install OpenCV with opencv_contrib.")
    if not hasattr(cv2.aruco, name):
        raise RuntimeError(f"Unknown dictionary: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_charuco_board(dictionary, squares_x, squares_y, square_px, marker_ratio):
    marker_px = int(square_px * marker_ratio)

    if hasattr(cv2.aruco, "CharucoBoard"):
        board = cv2.aruco.CharucoBoard(
            (squares_x, squares_y),
            float(square_px),
            float(marker_px),
            dictionary
        )
    else:
        board = cv2.aruco.CharucoBoard_create(
            squares_x,
            squares_y,
            float(square_px),
            float(marker_px),
            dictionary
        )

    image_size = (squares_x * square_px, squares_y * square_px)

    if hasattr(board, "generateImage"):
        return board.generateImage(image_size, marginSize=20, borderBits=1)

    return board.draw(image_size, marginSize=20, borderBits=1)


def write_obj(model_dir, board_width_m, board_height_m):
    mesh_dir = model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    half_w = board_width_m / 2.0
    half_h = board_height_m / 2.0
    t = 0.004

    obj = f'''mtllib charuco_board.mtl
o charuco_board_vertical_fixed

v {-half_w:.6f} {-half_h:.6f} 0
v {half_w:.6f} {-half_h:.6f} 0
v {half_w:.6f} {half_h:.6f} 0
v {-half_w:.6f} {half_h:.6f} 0

v {-half_w:.6f} {-half_h:.6f} {t:.6f}
v {half_w:.6f} {-half_h:.6f} {t:.6f}
v {half_w:.6f} {half_h:.6f} {t:.6f}
v {-half_w:.6f} {half_h:.6f} {t:.6f}

vt 0 1
vt 1 1
vt 1 0
vt 0 0

vn 0 0 -1
vn 0 0 1

usemtl charuco_texture

f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1

f 8/4/2 7/3/2 6/2/2
f 8/4/2 6/2/2 5/1/2
'''

    mtl = '''newmtl charuco_texture
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
d 1.0
illum 1
map_Kd charuco_board.png
'''

    (mesh_dir / "charuco_board.obj").write_text(obj)
    (mesh_dir / "charuco_board.mtl").write_text(mtl)


def write_model_files(model_dir, board_width_m, board_height_m):
    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="charuco_target">
    <static>true</static>

    <link name="board_link">
      <visual name="charuco_visual">
        <geometry>
          <mesh>
            <uri>model://charuco_target/meshes/charuco_board.obj</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>

      <collision name="charuco_collision">
        <geometry>
          <box>
            <size>{board_width_m:.4f} 0.03 {board_height_m:.4f}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
'''

    config = '''<?xml version="1.0"?>
<model>
  <name>charuco_target</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>local</name>
  </author>
  <description>Vertical ChArUco target board for Gazebo Sim.</description>
</model>
'''

    (model_dir / "model.sdf").write_text(sdf)
    (model_dir / "model.config").write_text(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / ".gz" / "models" / "charuco_target"))
    parser.add_argument("--dict", default="DICT_4X4_100")
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-px", type=int, default=180)
    parser.add_argument("--marker-ratio", type=float, default=0.70)
    parser.add_argument("--board-width-m", type=float, default=2.142857)
    parser.add_argument("--board-height-m", type=float, default=3.00)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    mesh_dir = model_dir / "meshes"
    texture_dir = model_dir / "materials" / "textures"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_aruco_dictionary(args.dict)
    img = make_charuco_board(
        dictionary,
        args.squares_x,
        args.squares_y,
        args.square_px,
        args.marker_ratio
    )

    cv2.imwrite(str(mesh_dir / "charuco_board.png"), img)
    cv2.imwrite(str(texture_dir / "charuco_board.png"), img)

    write_obj(model_dir, args.board_width_m, args.board_height_m)
    write_model_files(model_dir, args.board_width_m, args.board_height_m)

    print("Generated fixed vertical ChArUco target model:")
    print(model_dir)
    print("Dictionary:", args.dict)
    print("Default orientation: vertical board, face approximately +/-Y")
    print("If the drone views from X direction, use orientation {y: 0.7071 w: 0.7071}")


if __name__ == "__main__":
    main()

