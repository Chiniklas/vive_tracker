"""Tracker-driven IK for the TG2 Inspirehand model.

Usage:
    python tests/test_tracker_ik.py
    python tests/test_tracker_ik.py --target-scale 1.0 --target-offset 0 0 0.9

Notes:
  - Uses OpenVR tracker pose (calibrated) as IK target position + orientation.
  - Position + orientation IK to a TCP site (default: hand_tcp).
  - Requires actuators on the model to hold the solved pose.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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
DEFAULT_MODEL = REPO_ROOT / "asset" / "tg2_inspirehand" / "tg2_inspirehand.xml"
DEFAULT_CALIB = Path(__file__).resolve().parent / "tracker_calibration.json"
DEFAULT_SITE = "hand_tcp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive IK target from tracker pose.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to MuJoCo XML.")
    parser.add_argument("--tracker", default="tracker_1", help="Tracker device name (triad_openvr).")
    parser.add_argument("--site", default=DEFAULT_SITE, help="Site name to control (TCP).")
    parser.add_argument("--calib-file", type=Path, default=DEFAULT_CALIB, help="Calibration file path.")
    parser.add_argument("--no-calib", action="store_true", help="Do not apply calibration transform.")
    parser.add_argument("--no-calib-rot", action="store_true", help="Do not apply calibration rotation offset.")
    parser.add_argument("--update-hz", type=float, default=90.0, help="Tracker sample rate.")
    parser.add_argument("--iters", type=int, default=50, help="IK iterations per update.")
    parser.add_argument("--tol", type=float, default=1e-3, help="IK tolerance (m).")
    parser.add_argument("--damping", type=float, default=0.05, help="Damping (lambda) for DLS.")
    parser.add_argument("--step", type=float, default=1.0, help="Step size multiplier.")
    parser.add_argument("--w-pos", type=float, default=1.0, help="Position error weight.")
    parser.add_argument("--w-rot", type=float, default=0.5, help="Rotation error weight.")
    parser.add_argument("--target-scale", type=float, default=1.0, help="Scale for tracker position.")
    parser.add_argument("--target-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        help="XYZ offset applied to target (world frame).")
    parser.add_argument("--print-hz", type=float, default=5.0, help="Debug print rate.")
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


def quat_to_axis_angle(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    w = max(-1.0, min(1.0, w))
    angle = 2.0 * math.acos(w)
    s = math.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-8:
        return (angle, 0.0, 0.0)
    return (x / s * angle, y / s * angle, z / s * angle)


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


def load_calibration(path: Path) -> Optional[dict]:
    if not path.is_file():
        return None
    import json

    return json.loads(path.read_text())


def apply_calibration(pos: Tuple[float, float, float],
                      quat: Tuple[float, float, float, float],
                      calib: dict,
                      apply_rot_offset: bool = True) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
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
        q_flip = (0.0, 0.0, 1.0, 0.0)  # 180° about +Y
        q_ht = quat_mul(q_flip, quat_mul(q_ht, q_flip))
    return p_h, q_ht


def build_actuated_hinge_dofs(model: mujoco.MjModel) -> Tuple[List[int], List[int], List[int], List[int]]:
    joint_ids: List[int] = []
    dof_ids: List[int] = []
    actuator_ids: List[int] = []
    qpos_addrs: List[int] = []
    seen = set()
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        j_id = int(model.actuator_trnid[a, 0])
        if model.jnt_type[j_id] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        if j_id not in seen:
            seen.add(j_id)
            joint_ids.append(j_id)
            dof_ids.append(int(model.jnt_dofadr[j_id]))
        actuator_ids.append(a)
        qpos_addrs.append(int(model.jnt_qposadr[j_id]))
    return joint_ids, dof_ids, actuator_ids, qpos_addrs


def clamp_joint_limits(model: mujoco.MjModel, data: mujoco.MjData, joint_ids: List[int]) -> None:
    for j in joint_ids:
        if model.jnt_limited[j]:
            qadr = model.jnt_qposadr[j]
            lo, hi = model.jnt_range[j]
            data.qpos[qadr] = max(lo, min(hi, data.qpos[qadr]))


def set_ctrls(model: mujoco.MjModel, data: mujoco.MjData,
              actuator_ids: List[int], values: List[float]) -> None:
    for act_id, value in zip(actuator_ids, values):
        if model.actuator_ctrllimited[act_id]:
            lo, hi = model.actuator_ctrlrange[act_id]
            value = max(lo, min(hi, value))
        data.ctrl[act_id] = value


def solve_ik(model: mujoco.MjModel,
             data: mujoco.MjData,
             site_id: int,
             target: Tuple[float, float, float],
             target_quat: Tuple[float, float, float, float],
             iters: int,
             tol: float,
             damping: float,
             step: float,
             w_pos: float,
             w_rot: float,
             joint_ids: List[int],
             dof_ids: List[int]) -> float:
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    last_err = float("inf")

    for _ in range(iters):
        mujoco.mj_forward(model, data)
        cur_pos = data.site_xpos[site_id]
        cur_xmat = data.site_xmat[site_id]
        cur_rot = (
            (float(cur_xmat[0]), float(cur_xmat[1]), float(cur_xmat[2])),
            (float(cur_xmat[3]), float(cur_xmat[4]), float(cur_xmat[5])),
            (float(cur_xmat[6]), float(cur_xmat[7]), float(cur_xmat[8])),
        )
        cur_quat = mat_to_quat(cur_rot)

        pos_err = np.array([target[0] - cur_pos[0],
                            target[1] - cur_pos[1],
                            target[2] - cur_pos[2]], dtype=np.float64)
        q_err = quat_mul(target_quat, quat_inv(cur_quat))
        rot_err_vec = np.array(quat_to_axis_angle(q_err), dtype=np.float64)

        err_vec = np.concatenate((pos_err * w_pos, rot_err_vec * w_rot))
        err_norm = float(np.linalg.norm(pos_err))
        last_err = err_norm
        if err_norm < tol:
            break

        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        ndof = len(dof_ids)
        j_reduced = np.zeros((6, ndof), dtype=np.float64)
        for k, d in enumerate(dof_ids):
            j_reduced[0, k] = jacp[0, d] * w_pos
            j_reduced[1, k] = jacp[1, d] * w_pos
            j_reduced[2, k] = jacp[2, d] * w_pos
            j_reduced[3, k] = jacr[0, d] * w_rot
            j_reduced[4, k] = jacr[1, d] * w_rot
            j_reduced[5, k] = jacr[2, d] * w_rot

        a = j_reduced @ j_reduced.T + (damping * damping) * np.eye(6)
        try:
            v = np.linalg.solve(a, err_vec)
        except np.linalg.LinAlgError:
            break
        dq = j_reduced.T @ v  # shape (ndof,)

        for idx, j in enumerate(joint_ids):
            qadr = model.jnt_qposadr[j]
            data.qpos[qadr] += step * float(dq[idx])

        clamp_joint_limits(model, data, joint_ids)

    mujoco.mj_forward(model, data)
    return last_err


def get_tcp_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    pos = data.site_xpos[site_id]
    xmat = data.site_xmat[site_id]
    rot = (
        (float(xmat[0]), float(xmat[1]), float(xmat[2])),
        (float(xmat[3]), float(xmat[4]), float(xmat[5])),
        (float(xmat[6]), float(xmat[7]), float(xmat[8])),
    )
    quat = mat_to_quat(rot)
    return (float(pos[0]), float(pos[1]), float(pos[2])), quat


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

    if not args.calib_file.is_file():
        raise FileNotFoundError(f"Calibration file not found at {args.calib_file}")
    calib = load_calibration(args.calib_file)

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
    if site_id < 0:
        raise ValueError(f"Site '{args.site}' not found.")

    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tracker_target")
    if target_body_id < 0:
        raise ValueError("Body 'tracker_target' not found in model.")
    target_mocap_id = model.body_mocapid[target_body_id]
    if target_mocap_id < 0:
        raise ValueError("Body 'tracker_target' is not marked as mocap.")

    joint_ids, dof_ids, actuator_ids, actuator_qpos = build_actuated_hinge_dofs(model)
    if not dof_ids:
        raise RuntimeError("No actuated hinge joints found for IK.")

    update_dt = 1.0 / max(args.update_hz, 1e-6)
    next_update = time.time()
    next_print = time.time()
    last_qpos_target: Optional[np.ndarray] = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Close the window or press ESC to exit.")
        while viewer.is_running():
            now = time.time()
            if now >= next_update:
                pose = tracker.get_pose_quaternion()
                if pose:
                    p_vt = (pose[0], pose[1], pose[2])
                    q_vt = (pose[3], pose[4], pose[5], pose[6])
                    p_h, q_ht = apply_calibration(p_vt, q_vt, calib, apply_rot_offset=True)

                    target = (
                        p_h[0] * args.target_scale + args.target_offset[0],
                        p_h[1] * args.target_scale + args.target_offset[1],
                        p_h[2] * args.target_scale + args.target_offset[2],
                    )
                    data.mocap_pos[target_mocap_id] = target
                    data.mocap_quat[target_mocap_id] = q_ht

                    qpos_start = data.qpos.copy()
                    err = solve_ik(
                        model,
                        data,
                        site_id=site_id,
                        target=target,
                        target_quat=q_ht,
                        iters=args.iters,
                        tol=args.tol,
                        damping=args.damping,
                        step=args.step,
                        w_pos=args.w_pos,
                        w_rot=args.w_rot,
                        joint_ids=joint_ids,
                        dof_ids=dof_ids,
                    )
                    last_qpos_target = data.qpos.copy()
                    data.qpos[:] = qpos_start
                    data.qvel[:] = 0.0
                    mujoco.mj_forward(model, data)
                next_update = now + update_dt

            if last_qpos_target is not None:
                ctrl_vals = [float(last_qpos_target[qadr]) for qadr in actuator_qpos]
                set_ctrls(model, data, actuator_ids, ctrl_vals)

            mujoco.mj_step(model, data)
            viewer.sync()

            if now >= next_print:
                mujoco.mj_forward(model, data)
                pos, quat = get_tcp_pose(model, data, site_id)
                print(
                    f"TCP (world): {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f} | "
                    f"quat: {quat[0]: .4f} {quat[1]: .4f} {quat[2]: .4f} {quat[3]: .4f}"
                )
                next_print = now + max(1e-6, 1.0 / args.print_hz)

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
