"""Track a VR tracker and drive the wrist (root) joints in MuJoCo.

Usage:
    python tests/test_wrist_tracking.py
    python tests/test_wrist_tracking.py --tracker tracker_1

Notes:
  - Requires OpenVR runtime + connected tracker.
  - Drives the hand root via a mocap body + weld constraint (no Euler angles).

Terminology (frame + transform naming used in this script):
  - Frames:
      H = human frame (pelvis origin, +X right, +Y forward, +Z up) [absolute reference]
      V = VR base frame (SteamVR tracking space)
      T = tracker frame (tracker body frame reported by OpenVR)
  - Transform naming:
      T_ab means "from frame a to frame b", so p_b = T_ab * p_a
  - Live tracker data:
      OpenVR provides T_VT (tracker pose in VR base).
  - Calibration computes T_VH (VR base -> human) from pelvis origin + axis samples.
  - Optional mounting offset would be T_TA (tracker -> arm), not implemented here.
  - Applied chain (current):
      T_HT = T_VH * T_VT
      p_h, q_ht are the tracker pose expressed in human frame.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    print("Failed to import mujoco. Install with `pip install mujoco`.", file=sys.stderr)
    raise

try:
    import device.triad_openvr as triad_openvr
except Exception as exc:  # pragma: no cover
    print("Failed to import OpenVR bindings. Install `openvr` and ensure SteamVR is running.", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "asset" / "inspirehand" / "handright9253_simplified.xml"
DEFAULT_CALIB = Path(__file__).resolve().parent / "tracker_calibration.json"
DEFAULT_UPDATE_HZ = 90.0
CALIB_SAMPLES = 60
CALIB_HZ = 120.0
ZERO_AFTER_CALIB = False
ZERO_INITIAL_WITHOUT_CALIB = True
POS_SCALE = 1.0
POS_OFFSET = (0.0, 0.0, 0.0)

# Frame/transform naming:
#   T_ab means "from frame a to frame b", so p_b = T_ab * p_a.
#   V = VR base, T = tracker, H = human (pelvis).

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream tracker pose into MuJoCo wrist controls.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to MuJoCo XML.")
    parser.add_argument("--tracker", default="tracker_1", help="Tracker device name (triad_openvr).")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration and save transform, then exit.")
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB, help="Calibration file path.")
    parser.add_argument("--list-devices", action="store_true", help="List discovered OpenVR devices and exit.")
    return parser.parse_args()


def quat_mul(a: Tuple[float, float, float, float],
             b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_inv(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    w, x, y, z = q
    norm = w * w + x * x + y * y + z * z
    if norm == 0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, -x / norm, -y / norm, -z / norm)


def quat_to_euler_xyz(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    # roll (x-axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw (z-axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_to_mat(q: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float, float], ...]:
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return (
        (1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)),
        (2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)),
        (2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)),
    )


def mat_to_quat(m: Tuple[Tuple[float, float, float], ...]) -> Tuple[float, float, float, float]:
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


def dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = math.sqrt(dot(v, v))
    if n == 0.0:
        return (0.0, 0.0, 0.0)
    return (v[0] / n, v[1] / n, v[2] / n)


def avg_pose(tracker,
             samples: int,
             hz: float,
             label: Optional[str] = None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    interval = 1.0 / max(hz, 1e-6)
    pos_acc = [0.0, 0.0, 0.0]
    quat_acc = [0.0, 0.0, 0.0, 0.0]
    quat_ref = None
    count = 0
    last_print = 0.0
    for _ in range(samples):
        start = time.time()
        pose = tracker.get_pose_quaternion()
        if pose:
            if label and (time.time() - last_print) > 0.1:
                sys.stdout.write(
                    f"\r{label} raw xyz: {pose[0]: .4f} {pose[1]: .4f} {pose[2]: .4f} | "
                    f"quat: {pose[3]: .4f} {pose[4]: .4f} {pose[5]: .4f} {pose[6]: .4f}   "
                )
                sys.stdout.flush()
                last_print = time.time()
            pos_acc[0] += pose[0]
            pos_acc[1] += pose[1]
            pos_acc[2] += pose[2]
            quat = (pose[3], pose[4], pose[5], pose[6])
            if quat_ref is None:
                quat_ref = quat
            # Keep quaternions in same hemisphere for averaging.
            if quat_ref and (quat_ref[0] * quat[0] + quat_ref[1] * quat[1] +
                             quat_ref[2] * quat[2] + quat_ref[3] * quat[3]) < 0.0:
                quat = (-quat[0], -quat[1], -quat[2], -quat[3])
            quat_acc[0] += quat[0]
            quat_acc[1] += quat[1]
            quat_acc[2] += quat[2]
            quat_acc[3] += quat[3]
            count += 1
        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)
    if count == 0:
        raise RuntimeError("No tracker samples collected.")
    pos_avg = (pos_acc[0] / count, pos_acc[1] / count, pos_acc[2] / count)
    qn = math.sqrt(sum(q * q for q in quat_acc))
    if qn == 0.0:
        quat_avg = (1.0, 0.0, 0.0, 0.0)
    else:
        quat_avg = (quat_acc[0] / qn, quat_acc[1] / qn, quat_acc[2] / qn, quat_acc[3] / qn)
    return pos_avg, quat_avg


def calibrate_tracker(tracker, samples: int, hz: float, out_path: Path) -> None:
    print("Calibration uses robotics frame: +X right, +Y forward, +Z up.")
    input("Place tracker at pelvis origin with PALM DOWN, fingers forward, then press Enter...")
    p_v0, q_vt0 = avg_pose(tracker, samples, hz, label="Pelvis")
    input("Move tracker to +X (right) from pelvis, then press Enter...")
    p_vx, _ = avg_pose(tracker, samples, hz, label="+X")
    input("Move tracker to +Y (forward) from pelvis, then press Enter...")
    p_vy, _ = avg_pose(tracker, samples, hz, label="+Y")
    input("Move tracker to +Z (up) from pelvis, then press Enter...")
    p_vz, _ = avg_pose(tracker, samples, hz, label="+Z")

    # Human axes expressed in VR frame.
    x_h_in_v = normalize((p_vx[0] - p_v0[0], p_vx[1] - p_v0[1], p_vx[2] - p_v0[2]))
    z_h_in_v = normalize((p_vz[0] - p_v0[0], p_vz[1] - p_v0[1], p_vz[2] - p_v0[2]))
    y_h_in_v = normalize(cross(z_h_in_v, x_h_in_v))
    # Align sign with sampled +Y if needed
    y_raw = normalize((p_vy[0] - p_v0[0], p_vy[1] - p_v0[1], p_vy[2] - p_v0[2]))
    if dot(y_h_in_v, y_raw) < 0:
        y_h_in_v = (-y_h_in_v[0], -y_h_in_v[1], -y_h_in_v[2])
    x_h_in_v = normalize(cross(y_h_in_v, z_h_in_v))

    # Orientation offset: map the origin pose (palm down) to identity in human frame.
    r_vt0 = quat_to_mat(q_vt0)
    r_ht0 = (
        (dot(x_h_in_v, (r_vt0[0][0], r_vt0[1][0], r_vt0[2][0])),
         dot(x_h_in_v, (r_vt0[0][1], r_vt0[1][1], r_vt0[2][1])),
         dot(x_h_in_v, (r_vt0[0][2], r_vt0[1][2], r_vt0[2][2]))),
        (dot(y_h_in_v, (r_vt0[0][0], r_vt0[1][0], r_vt0[2][0])),
         dot(y_h_in_v, (r_vt0[0][1], r_vt0[1][1], r_vt0[2][1])),
         dot(y_h_in_v, (r_vt0[0][2], r_vt0[1][2], r_vt0[2][2]))),
        (dot(z_h_in_v, (r_vt0[0][0], r_vt0[1][0], r_vt0[2][0])),
         dot(z_h_in_v, (r_vt0[0][1], r_vt0[1][1], r_vt0[2][1])),
         dot(z_h_in_v, (r_vt0[0][2], r_vt0[1][2], r_vt0[2][2]))),
    )
    q_ht0 = mat_to_quat(r_ht0)

    # Rotation sign check: rotate +X (roll) and +Z (yaw) in human frame.
    input("Rotate tracker +X (right-hand rule), keep position, then press Enter...")
    _, q_vt_roll = avg_pose(tracker, samples, hz, label="+RollX")
    input("Rotate tracker +Z (right-hand rule), keep position, then press Enter...")
    _, q_vt_yaw = avg_pose(tracker, samples, hz, label="+YawZ")

    q_roll_h = apply_calibration((p_v0[0], p_v0[1], p_v0[2]), q_vt_roll, {
        "p_v0": p_v0,
        "x_h_in_v": x_h_in_v,
        "y_h_in_v": y_h_in_v,
        "z_h_in_v": z_h_in_v,
        "q_ht0": q_ht0,
    }, apply_rot_offset=True)[1]
    q_yaw_h = apply_calibration((p_v0[0], p_v0[1], p_v0[2]), q_vt_yaw, {
        "p_v0": p_v0,
        "x_h_in_v": x_h_in_v,
        "y_h_in_v": y_h_in_v,
        "z_h_in_v": z_h_in_v,
        "q_ht0": q_ht0,
    }, apply_rot_offset=True)[1]

    roll, _, _ = quat_to_euler_xyz(q_roll_h)
    _, _, yaw = quat_to_euler_xyz(q_yaw_h)
    rot_threshold = 0.2  # rad
    flip_rot_xz = (roll < -rot_threshold and yaw < -rot_threshold)

    calib = {
        # New terminology (preferred).
        "p_v0": [p_v0[0], p_v0[1], p_v0[2]],
        "x_h_in_v": [x_h_in_v[0], x_h_in_v[1], x_h_in_v[2]],
        "y_h_in_v": [y_h_in_v[0], y_h_in_v[1], y_h_in_v[2]],
        "z_h_in_v": [z_h_in_v[0], z_h_in_v[1], z_h_in_v[2]],
        "q_ht0": [q_ht0[0], q_ht0[1], q_ht0[2], q_ht0[3]],
        "flip_rot_xz": flip_rot_xz,
        "rot_check": {"roll": roll, "yaw": yaw},
        # Back-compat keys for older runs.
        "origin": [p_v0[0], p_v0[1], p_v0[2]],
        "axes": {
            "x": [x_h_in_v[0], x_h_in_v[1], x_h_in_v[2]],
            "y": [y_h_in_v[0], y_h_in_v[1], y_h_in_v[2]],
            "z": [z_h_in_v[0], z_h_in_v[1], z_h_in_v[2]],
        },
        "quat0_h": [q_ht0[0], q_ht0[1], q_ht0[2], q_ht0[3]],
    }
    out_path.write_text(json.dumps(calib, indent=2))
    print(f"Saved calibration to {out_path}")


def load_calibration(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def apply_calibration(pos: Tuple[float, float, float],
                      quat: Tuple[float, float, float, float],
                      calib: dict,
                      apply_rot_offset: bool = True) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    # Inputs are tracker pose in VR frame: T_VT.
    p_vt = pos
    q_vt = quat

    p_v0 = calib.get("p_v0", calib.get("origin"))
    if p_v0 is None:
        raise RuntimeError("Calibration missing p_v0/origin.")

    x_h_in_v = tuple(calib.get("x_h_in_v", calib.get("axes", {}).get("x")))
    y_h_in_v = tuple(calib.get("y_h_in_v", calib.get("axes", {}).get("y")))
    z_h_in_v = tuple(calib.get("z_h_in_v", calib.get("axes", {}).get("z")))

    dv = (p_vt[0] - p_v0[0], p_vt[1] - p_v0[1], p_vt[2] - p_v0[2])
    p_h = (dot(dv, x_h_in_v), dot(dv, y_h_in_v), dot(dv, z_h_in_v))

    r_vt = quat_to_mat(q_vt)
    r_ht = (
        (dot(x_h_in_v, (r_vt[0][0], r_vt[1][0], r_vt[2][0])),
         dot(x_h_in_v, (r_vt[0][1], r_vt[1][1], r_vt[2][1])),
         dot(x_h_in_v, (r_vt[0][2], r_vt[1][2], r_vt[2][2]))),
        (dot(y_h_in_v, (r_vt[0][0], r_vt[1][0], r_vt[2][0])),
         dot(y_h_in_v, (r_vt[0][1], r_vt[1][1], r_vt[2][1])),
         dot(y_h_in_v, (r_vt[0][2], r_vt[1][2], r_vt[2][2]))),
        (dot(z_h_in_v, (r_vt[0][0], r_vt[1][0], r_vt[2][0])),
         dot(z_h_in_v, (r_vt[0][1], r_vt[1][1], r_vt[2][1])),
         dot(z_h_in_v, (r_vt[0][2], r_vt[1][2], r_vt[2][2]))),
    )
    q_ht = mat_to_quat(r_ht)
    if apply_rot_offset:
        q_ht0 = calib.get("q_ht0", calib.get("quat0_h"))
        if q_ht0 is not None:
            q_ht = quat_mul(quat_inv((q_ht0[0], q_ht0[1], q_ht0[2], q_ht0[3])), q_ht)
    if calib.get("flip_rot_xz", False):
        # Flip rotation direction about X and Z while keeping axes aligned:
        # conjugate by 180° about +Y (R' = R_y(pi) * R * R_y(pi)).
        q_flip = (0.0, 0.0, 1.0, 0.0)  # 180° about +Y
        q_ht = quat_mul(q_flip, quat_mul(q_ht, q_flip))
    return p_h, q_ht


def main() -> Optional[int]:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found at {args.model}")

    vr = triad_openvr.triad_openvr()
    if args.list_devices:
        vr.print_discovered_objects()
        return 0

    tracker = vr.devices.get(args.tracker)
    if tracker is None:
        vr.print_discovered_objects()
        raise RuntimeError(f"Tracker '{args.tracker}' not found. See list above.")

    if args.calibrate:
        calibrate_tracker(tracker, CALIB_SAMPLES, CALIB_HZ, args.calib_file)
        return 0

    calib = None
    calib = load_calibration(args.calib_file)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    tracker_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tracker_marker")
    if tracker_body_id < 0:
        raise ValueError("Body 'tracker_marker' not found in model.")
    mocap_id = model.body_mocapid[tracker_body_id]
    if mocap_id < 0:
        raise ValueError("Body 'tracker_marker' is not marked as mocap.")

    human_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "human_frame")
    if human_body_id >= 0:
        human_mocap_id = model.body_mocapid[human_body_id]
        if human_mocap_id >= 0:
            data.mocap_pos[human_mocap_id] = (0.0, 0.0, 0.0)
            data.mocap_quat[human_mocap_id] = (1.0, 0.0, 0.0, 0.0)

    hand_mocap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand_mocap")
    if hand_mocap_body_id < 0:
        raise ValueError("Body 'hand_mocap' not found in model.")
    hand_mocap_id = model.body_mocapid[hand_mocap_body_id]
    if hand_mocap_id < 0:
        raise ValueError("Body 'hand_mocap' is not marked as mocap.")

    zero_p_h = None  # type: Optional[Tuple[float, float, float]]
    zero_q_ht = None  # type: Optional[Tuple[float, float, float, float]]
    use_zero = (calib is None and ZERO_INITIAL_WITHOUT_CALIB) or ZERO_AFTER_CALIB

    update_dt = 1.0 / max(DEFAULT_UPDATE_HZ, 1e-6)
    next_update = time.time()
    last_print = 0.0
    print_interval = 0.1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Close the window or press ESC to exit.")
        while viewer.is_running():
            now = time.time()
            if now >= next_update:
                pose = tracker.get_pose_quaternion()
                if pose:
                    p_vt = (pose[0], pose[1], pose[2])
                    q_vt = (pose[3], pose[4], pose[5], pose[6])
                    if calib:
                        p_h, q_ht = apply_calibration(p_vt, q_vt, calib, apply_rot_offset=True)
                    else:
                        p_h, q_ht = p_vt, q_vt

                    if zero_p_h is None and use_zero:
                        zero_p_h = p_h
                    if zero_q_ht is None and use_zero:
                        zero_q_ht = q_ht

                    # position (calibrated human frame)
                    px, py, pz = p_h
                    if zero_p_h:
                        px -= zero_p_h[0]
                        py -= zero_p_h[1]
                        pz -= zero_p_h[2]
                    px = px * POS_SCALE + POS_OFFSET[0]
                    py = py * POS_SCALE + POS_OFFSET[1]
                    pz = pz * POS_SCALE + POS_OFFSET[2]
                    data.mocap_pos[mocap_id] = (px, py, pz)

                    # rotation (quaternion in human frame)
                    if zero_q_ht:
                        quat_rel = quat_mul(quat_inv(zero_q_ht), q_ht)
                    else:
                        quat_rel = q_ht
                    data.mocap_quat[mocap_id] = quat_rel

                    # drive hand via mocap + weld (no Euler)
                    data.mocap_pos[hand_mocap_id] = (px, py, pz)
                    data.mocap_quat[hand_mocap_id] = quat_rel

                    if now - last_print >= print_interval:
                        sys.stdout.write(
                            f"\rtracker xyz: {p_h[0]: .4f} {p_h[1]: .4f} {p_h[2]: .4f} | "
                            f"quat: {q_ht[0]: .4f} {q_ht[1]: .4f} {q_ht[2]: .4f} {q_ht[3]: .4f}   "
                        )
                        sys.stdout.flush()
                        last_print = now

                next_update = now + update_dt

            mujoco.mj_step(model, data)
            viewer.sync()

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
