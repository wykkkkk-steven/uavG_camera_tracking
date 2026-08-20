"""
五面 ArUco 正方体目标面追踪
"""

import math

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from .face_config import (
        FACE_GRAPH,
        TOP_FACE_ID,
        TARGET_CONFIRM_FRAMES,
        LOST_CONFIRM_FRAMES,
        SWITCH_THRESHOLD,
    )
except ImportError:
    from face_config import (
        FACE_GRAPH,
        TOP_FACE_ID,
        TARGET_CONFIRM_FRAMES,
        LOST_CONFIRM_FRAMES,
        SWITCH_THRESHOLD,
    )


class TargetFaceTracker:

    def __init__(
        self,
        face_graph=FACE_GRAPH,
        top_face=TOP_FACE_ID,
        confirm_frames=TARGET_CONFIRM_FRAMES,
        lost_frames=LOST_CONFIRM_FRAMES,
        switch_threshold=SWITCH_THRESHOLD,
    ):
        self.face_graph = face_graph
        self.top_face = top_face
        self.confirm_frames = confirm_frames
        self.lost_frames = lost_frames
        self.switch_threshold = switch_threshold

        self.target_id = None
        self.target_relation = {}
        self.current_face = None
        self.state = "FIND_TARGET"
        self._target_confirm = 0
        self._lost_count = 0

    def _neighbors(self, face_id):
        return self.face_graph.get(face_id, {})

    def _first_step_direction(self, start, goal):
        """静态预计算用：从 start 出发到 goal 的最短路径第一步方向。"""
        if start == goal:
            return "CENTER"
        if start not in self.face_graph or goal not in self.face_graph:
            return "NONE"

        queue = [start]
        parent = {start: None}
        first_step = {start: None}
        head = 0

        while head < len(queue):
            current = queue[head]
            head += 1
            for direction, neighbor_list in self._neighbors(current).items():
                if isinstance(neighbor_list, int):
                    neighbor_list = [neighbor_list]
                for neighbor in neighbor_list:
                    if neighbor in parent:
                        continue
                    parent[neighbor] = current
                    first_step[neighbor] = (
                        direction if current == start else first_step[current]
                    )
                    if neighbor == goal:
                        return first_step[neighbor]
                    queue.append(neighbor)

        return "NONE"

    def _lock_target(self, face_id):
        self.target_id = face_id
        self.target_relation = {}
        for face in self.face_graph:
            if face == self.top_face:
                continue
            direction = self._first_step_direction(face, face_id)
            self.target_relation[face] = self._direction_label(direction)
        self.target_relation[face_id] = "CENTER"

    @staticmethod
    def _direction_label(direction):
        mapping = {
            "left": "LEFT",
            "right": "RIGHT",
            "opposite": "BACK",
            "up": "UP",
            "down": "DOWN",
            "CENTER": "CENTER",
            "NONE": "NONE",
        }
        return mapping.get(direction, "NONE")

    def _select_current_face(self, visible_ids, areas):
        """选择当前主要观察面。

        优先级:
        1. 目标面可见 -> 目标面
        2. 忽略顶面
        3. 剩余侧面中面积最大者；带滞回防止切换抖动。
        """
        if self.target_id is not None and self.target_id in visible_ids:
            self.current_face = self.target_id
            return self.target_id

        candidates = [
            face_id
            for face_id in visible_ids
            if face_id != self.top_face
        ]
        if not candidates:
            self.current_face = None
            return None

        best_face = max(
            candidates,
            key=lambda face_id: areas.get(face_id, 0.0),
        )

        if (
            self.current_face in candidates
            and best_face != self.current_face
        ):
            current_area = areas.get(self.current_face, 0.0)
            new_area = areas.get(best_face, 0.0)
            if new_area < current_area * self.switch_threshold:
                return self.current_face

        self.current_face = best_face
        return best_face

    def _target_pose(self, detections, image_w, fx):
        """从目标面检测结果计算 yaw_error / distance / angle_error。"""
        yaw_error = None
        distance = None
        angle_error = None

        for det in detections:
            if int(det["id"]) != self.target_id:
                continue

            corners = det.get("corners")
            if corners is not None and image_w and fx and fx > 0:
                corners_arr = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
                cx = float(corners_arr[:, 0].mean())
                yaw_error = math.atan((cx - image_w / 2.0) / fx)

            tvec = det.get("tvec")
            if tvec is not None:
                distance = float(math.sqrt(
                    float(tvec[0]) ** 2
                    + float(tvec[1]) ** 2
                    + float(tvec[2]) ** 2
                ))

            rvec = det.get("rvec")
            if rvec is not None and cv2 is not None:
                try:
                    rotation_matrix, _ = cv2.Rodrigues(
                        np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                    )
                    normal = rotation_matrix[:, 2]
                    angle_error = math.degrees(
                        math.acos(
                            max(
                                -1.0,
                                min(
                                    1.0,
                                    float(normal[2]),
                                ),
                            )
                        )
                    )
                except Exception:
                    angle_error = None
            break

        return yaw_error, distance, angle_error

    def update(self, detections, image_w=None, fx=None):
        visible_ids = []
        areas = {}
        for det in detections:
            face_id = int(det["id"])
            if face_id in self.face_graph:
                visible_ids.append(face_id)
                areas[face_id] = float(det.get("area", 0.0))

        if self.target_id is None:
            for face_id in visible_ids:
                if face_id != self.top_face:
                    self._lock_target(face_id)
                    break

        result = {
            "state": self.state,
            "target_visible": False,
            "target_id": self.target_id,
            "current_face": self.current_face,
            "target_direction": "NONE",
            "yaw_error": None,
            "distance": None,
            "angle_error": None,
        }

        if self.target_id is None:
            if visible_ids:
                self.state = "FIND_TARGET"
            result["state"] = self.state
            result["current_face"] = self.current_face
            return result

        target_visible = self.target_id in visible_ids

        if target_visible:
            self._target_confirm += 1
            self._lost_count = 0
        else:
            self._target_confirm = 0
            self._lost_count += 1

        if target_visible and self._target_confirm >= self.confirm_frames:
            self.state = "TARGET_TRACK"
            self.current_face = self.target_id
            yaw_error, distance, angle_error = self._target_pose(
                detections, image_w, fx
            )
            result.update(
                {
                    "state": "TARGET_TRACK",
                    "target_visible": True,
                    "current_face": self.target_id,
                    "target_direction": "CENTER",
                    "yaw_error": yaw_error,
                    "distance": distance,
                    "angle_error": angle_error,
                }
            )
            return result

        if target_visible:
            result["state"] = self.state
            result["target_visible"] = True
            result["current_face"] = self.current_face
            if self.state == "TARGET_TRACK":
                result["target_direction"] = "CENTER"
            return result

        if not visible_ids:
            if self._lost_count >= self.lost_frames:
                self.state = "FIND_TARGET"
            else:
                self.state = "LOST_CONFIRM"
            result["state"] = self.state
            result["current_face"] = self.current_face
            return result

        current_face = self._select_current_face(visible_ids, areas)
        direction = self.target_relation.get(current_face, "NONE")
        self.state = "FIND_TARGET"
        result.update(
            {
                "state": "FIND_TARGET",
                "current_face": current_face,
                "target_direction": direction,
            }
        )
        return result

    def reset(self):
        self.target_id = None
        self.target_relation = {}
        self.current_face = None
        self.state = "FIND_TARGET"
        self._target_confirm = 0
        self._lost_count = 0
