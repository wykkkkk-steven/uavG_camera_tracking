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


def make_marker(dictionary, marker_id, size_px):
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)

    img = None
    img = cv2.aruco.drawMarker(dictionary, marker_id, size_px)
    return img


def write_obj(model_dir, board_size_m):
    mesh_dir = model_dir / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    half = board_size_m / 2.0
    t = 0.004

    # The marker is made in the OBJ X-Y plane.
    # Gazebo/OGRE conversion usually makes it stand vertically in Gazebo.
    # If it is still not vertical in your Gazebo, spawn with roll/pitch can fix it,
    # but this layout is the most reliable with Gazebo Sim mesh import.
    obj = f'''mtllib aruco_marker.mtl
o simple_aruco_marker

v {-half:.6f} {-half:.6f} 0
v {half:.6f} {-half:.6f} 0
v {half:.6f} {half:.6f} 0
v {-half:.6f} {half:.6f} 0

v {-half:.6f} {-half:.6f} {t:.6f}
v {half:.6f} {-half:.6f} {t:.6f}
v {half:.6f} {half:.6f} {t:.6f}
v {-half:.6f} {half:.6f} {t:.6f}

vt 0 1
vt 1 1
vt 1 0
vt 0 0

vn 0 0 -1
vn 0 0 1

usemtl aruco_texture

f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1

f 8/4/2 7/3/2 6/2/2
f 8/4/2 6/2/2 5/1/2
'''

    mtl = '''newmtl aruco_texture
Ka 1.000 1.000 1.000
Kd 1.000 1.000 1.000
Ks 0.000 0.000 0.000
d 1.0
illum 1
map_Kd aruco_marker.png
'''

    (mesh_dir / "aruco_marker.obj").write_text(obj)
    (mesh_dir / "aruco_marker.mtl").write_text(mtl)


def write_model_files(model_dir, board_size_m):
    sdf = f'''<?xml version="1.0" ?>
<sdf version="1.9">
  <model name="simple_aruco_target">
    <static>true</static>

    <link name="marker_link">
      <visual name="marker_visual">
        <geometry>
          <mesh>
            <uri>model://simple_aruco_target/meshes/aruco_marker.obj</uri>
            <scale>1 1 1</scale>
          </mesh>
        </geometry>
      </visual>

      <collision name="marker_collision">
        <geometry>
          <box>
            <size>{board_size_m:.4f} 0.03 {board_size_m:.4f}</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
'''

    config = '''<?xml version="1.0"?>
<model>
  <name>simple_aruco_target</name>
  <version>1.0</version>
  <sdf version="1.9">model.sdf</sdf>
  <author>
    <name>local</name>
  </author>
  <description>Simple large ArUco marker target for Gazebo Sim.</description>
</model>
'''

    (model_dir / "model.sdf").write_text(sdf)
    (model_dir / "model.config").write_text(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=str(Path.home() / ".gz" / "models" / "simple_aruco_target"))
    parser.add_argument("--dict", default="DICT_4X4_100")
    parser.add_argument("--id", type=int, default=0)
    parser.add_argument("--size-px", type=int, default=1000)
    parser.add_argument("--board-size-m", type=float, default=1.20)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    mesh_dir = model_dir / "meshes"
    texture_dir = model_dir / "materials" / "textures"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    dictionary = get_aruco_dictionary(args.dict)
    img = make_marker(dictionary, args.id, args.size_px)

    cv2.imwrite(str(mesh_dir / "aruco_marker.png"), img)
    cv2.imwrite(str(texture_dir / "aruco_marker.png"), img)

    write_obj(model_dir, args.board_size_m)
    write_model_files(model_dir, args.board_size_m)

    print("Generated simple ArUco target:")
    print(model_dir)
    print("Dictionary:", args.dict)
    print("Marker ID:", args.id)
    print("Recommended detector settings:")
    print("  ARUCO_DICTIONARY = cv2.aruco." + args.dict)
    print("  TARGET_MARKER_ID = " + str(args.id))
    print("")
    print("Spawn example:")
    print(
        "gz service -s /world/baylands/create "
        "--reqtype gz.msgs.EntityFactory "
        "--reptype gz.msgs.Boolean "
        "--timeout 300 "
        "--req 'sdf_filename: \"" + str(model_dir / "model.sdf") + "\" "
        "name: \"simple_aruco_target\" "
        "pose {position {x: 8 y: 0 z: 1.5} orientation {y: 0.7071 w: 0.7071}}'"
    )


if __name__ == "__main__":
    main()
