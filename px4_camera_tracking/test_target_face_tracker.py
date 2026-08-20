"""
TargetFaceTracker 单元测试（绝对方向方案）

覆盖:
1. 锁定第一个检测到的侧面
2. 目标面可见 -> TARGET_TRACK + CENTER
3. 目标面被遮挡 -> FIND_TARGET + 绝对方向
4. 顶面不参与方向选择
5. 侧面切换滞回（SWITCH_THRESHOLD）
6. 完全丢失 -> LOST_CONFIRM -> FIND_TARGET
7. 目标面姿态输出（yaw_error / distance / angle_error）
"""

import numpy as np

import sys
import types

if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    def _rodrigues(rvec):
        return np.eye(3, dtype=np.float64), None
    fake_cv2.Rodrigues = _rodrigues
    sys.modules["cv2"] = fake_cv2

from target_face_tracker import TargetFaceTracker


def det(ids, areas=None, corners=False, tvec=False):
    result = []
    for idx, face_id in enumerate(ids):
        item = {"id": face_id}
        if areas is not None:
            item["area"] = areas[idx]
        else:
            item["area"] = float(1000 + face_id * 100)
        if corners:
            item["corners"] = np.array(
                [[200.0, 100.0], [300.0, 100.0],
                 [300.0, 200.0], [200.0, 200.0]],
                dtype=np.float32,
            )
        if tvec:
            item["tvec"] = np.array([0.5, -0.2, 3.0], dtype=np.float64)
            item["rvec"] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        result.append(item)
    return result


def test_lock_first_side():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    assert tracker.target_id == 4
    assert tracker.target_relation[4] == "CENTER"
    print("PASS 1: 锁定第一个侧面 -> target_id=4")


def test_lock_ignores_top():
    tracker = TargetFaceTracker()
    tracker.update(det([0]))
    assert tracker.target_id is None
    r = tracker.update(det([0, 4]))
    assert tracker.target_id == 4
    print("PASS 2: 先看到顶面不锁定，出现侧面后锁定")


def test_target_visible():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    for _ in range(5):
        r = tracker.update(det([4, 0]))
    assert r["state"] == "TARGET_TRACK"
    assert r["target_visible"] is True
    assert r["target_direction"] == "CENTER"
    print("PASS 3: 目标面可见 -> TARGET_TRACK")


def test_absolute_direction():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    r = tracker.update(det([2]))
    # 锁定 target=4，从 2 出发：2 的 opposite 是 4 -> BACK
    assert r["state"] == "FIND_TARGET"
    assert r["target_direction"] == "BACK"
    assert r["current_face"] == 2
    print("PASS 4: 绝对方向表 2->4 = BACK")


def test_top_ignored_in_selection():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    r = tracker.update(det([0, 2], areas=[5000.0, 3000.0]))
    assert r["current_face"] == 2
    assert r["target_direction"] == "BACK"
    print("PASS 5: 顶面0面积大但被忽略，选侧面2")


def test_hysteresis_switch():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    # 锁定 target=4。当前看到 1 (面积 2000)，候选 2 (面积 2100)
    # 2100 < 2000*1.2=2400 -> 不切换，保持 1
    r = tracker.update(det([1], areas=[2000.0]))
    assert r["current_face"] == 1
    r = tracker.update(det([1, 2], areas=[2000.0, 2100.0]))
    assert r["current_face"] == 1
    # 候选面积 2600 > 2400 -> 切换
    r = tracker.update(det([1, 2], areas=[2000.0, 2600.0]))
    assert r["current_face"] == 2
    print("PASS 6: 滞回切换 SWITCH_THRESHOLD=1.2")


def test_lost_confirm_then_find():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    for _ in range(5):
        tracker.update(det([4]))
    r = tracker.update(det([]))
    assert r["state"] == "LOST_CONFIRM"
    for _ in range(5):
        r = tracker.update(det([]))
    assert r["state"] == "FIND_TARGET"
    print("PASS 7: 短暂丢失 LOST_CONFIRM -> 持续丢失 FIND_TARGET")


def test_target_pose_output():
    tracker = TargetFaceTracker()
    tracker.update(det([4]))
    for _ in range(5):
        r = tracker.update(det([4], corners=True, tvec=True), image_w=640, fx=320.0)
    assert r["target_visible"] is True
    assert r["distance"] is not None
    assert abs(r["distance"] - 3.048) < 0.05
    assert r["yaw_error"] is not None
    assert r["angle_error"] is not None
    print(f"PASS 8: 目标姿态 yaw={r['yaw_error']:.3f} d={r['distance']:.2f} a={r['angle_error']:.1f}")


if __name__ == "__main__":
    test_lock_first_side()
    test_lock_ignores_top()
    test_target_visible()
    test_absolute_direction()
    test_top_ignored_in_selection()
    test_hysteresis_switch()
    test_lost_confirm_then_find()
    test_target_pose_output()
    print("\n全部测试通过")
