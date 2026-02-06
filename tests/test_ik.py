"""Simple Jacobian-based IK for the TG2 Inspirehand model.

Usage:
    python tests/test_ik.py --target 0.2 -0.2 0.2 --site hand_tcp --show --move-seconds 2 --hold-seconds 20

Notes:
  - Uses MuJoCo Jacobians (no external IK library).
  - Position + orientation IK with damped least squares.
"""

from __future__ import annotations

import argparse
import math
import time
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover
    print("Failed to import mujoco. Install with `pip install mujoco`.", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "asset" / "tg2_inspirehand" / "tg2_inspirehand.xml"
DEFAULT_BODY = "wrist_roll_r_link"
DEFAULT_SITE = "hand_tcp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DLS IK for a target body position.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Path to MuJoCo XML.")
    parser.add_argument("--body", default=DEFAULT_BODY, help="Body name to control.")
    parser.add_argument("--site", default=DEFAULT_SITE, help="Site name to control (preferred for TCP).")
    parser.add_argument("--target", type=float, nargs=3, action="append",
                        help="Target XYZ in world frame (repeatable).")
    parser.add_argument("--targets", type=float, nargs="+",
                        help="Flat list of XYZ targets (length multiple of 3).")
    parser.add_argument("--targets-file", type=Path,
                        help="JSON or text file with targets: each line 'x y z' or JSON [[x,y,z], ...].")
    parser.add_argument("--iters", type=int, default=200, help="Max IK iterations.")
    parser.add_argument("--tol", type=float, default=1e-4, help="Target position tolerance (m).")
    parser.add_argument("--damping", type=float, default=0.05, help="Damping (lambda) for DLS.")
    parser.add_argument("--step", type=float, default=1.0, help="Step size multiplier.")
    parser.add_argument("--target-quat", type=float, nargs=4,
                        help="Target orientation as w x y z (applied to all targets).")
    parser.add_argument("--target-rpy", type=float, nargs=3,
                        help="Target orientation as roll pitch yaw in radians (applied to all targets).")
    parser.add_argument("--w-pos", type=float, default=1.0, help="Position error weight.")
    parser.add_argument("--w-rot", type=float, default=0.5, help="Rotation error weight.")
    parser.add_argument("--show", action="store_true", help="Launch viewer after solving.")
    parser.add_argument("--hold-seconds", type=float, default=1.0,
                        help="Seconds to hold each target pose (viewer only).")
    parser.add_argument("--settle-seconds", type=float, default=1.0,
                        help="Seconds to simulate before starting IK (viewer only).")
    parser.add_argument("--move-seconds", type=float, default=1.0,
                        help="Seconds to move toward each target (viewer only).")
    parser.add_argument("--print-hz", type=float, default=5.0,
                        help="TCP position print rate (viewer only).")
    return parser.parse_args()


def build_actuated_hinge_dofs(model: mujoco.MjModel) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Return (joint_ids, dof_ids, actuator_ids, qpos_addrs) for actuated hinge joints."""
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


def build_hinge_dofs(model: mujoco.MjModel) -> Tuple[List[int], List[int]]:
    """Return (joint_ids, dof_ids) for all hinge joints."""
    joint_ids: List[int] = []
    dof_ids: List[int] = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            joint_ids.append(j)
            dof_ids.append(int(model.jnt_dofadr[j]))
    return joint_ids, dof_ids


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


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)


def solve_ik(model: mujoco.MjModel,
             data: mujoco.MjData,
             body_id: Optional[int],
             site_id: Optional[int],
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
        if site_id is not None:
            cur_pos = data.site_xpos[site_id]
            cur_xmat = data.site_xmat[site_id]
            cur_rot = (
                (float(cur_xmat[0]), float(cur_xmat[1]), float(cur_xmat[2])),
                (float(cur_xmat[3]), float(cur_xmat[4]), float(cur_xmat[5])),
                (float(cur_xmat[6]), float(cur_xmat[7]), float(cur_xmat[8])),
            )
            cur_quat = mat_to_quat(cur_rot)
        elif body_id is not None:
            cur_pos = data.xpos[body_id]
            cq = data.xquat[body_id]
            cur_quat = (float(cq[0]), float(cq[1]), float(cq[2]), float(cq[3]))
        else:
            raise RuntimeError("No body or site specified for IK.")
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

        if site_id is not None:
            mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        elif body_id is not None:
            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)

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
        dq = j_reduced.T @ v

        # Apply step to qpos (only hinge joints).
        for idx, j in enumerate(joint_ids):
            qadr = model.jnt_qposadr[j]
            data.qpos[qadr] += step * float(dq[idx])

        clamp_joint_limits(model, data, joint_ids)

    mujoco.mj_forward(model, data)
    return last_err


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


def load_targets(path: Path) -> List[Tuple[float, float, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Targets file not found at {path}")
    text = path.read_text().strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        import json

        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON targets must be a list of [x, y, z].")
        out: List[Tuple[float, float, float]] = []
        for item in data:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError("Each JSON target must be [x, y, z].")
            out.append((float(item[0]), float(item[1]), float(item[2])))
        return out

    out: List[Tuple[float, float, float]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p for p in line.replace(",", " ").split() if p]
        if len(parts) != 3:
            raise ValueError(f"Bad target line: '{line}' (expected 3 numbers)")
        out.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return out


def get_tcp_pose(model: mujoco.MjModel,
                 data: mujoco.MjData,
                 body_id: Optional[int],
                 site_id: Optional[int]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]:
    if site_id is not None:
        pos = data.site_xpos[site_id]
        xmat = data.site_xmat[site_id]
        rot = (
            (float(xmat[0]), float(xmat[1]), float(xmat[2])),
            (float(xmat[3]), float(xmat[4]), float(xmat[5])),
            (float(xmat[6]), float(xmat[7]), float(xmat[8])),
        )
        quat = mat_to_quat(rot)
        return (float(pos[0]), float(pos[1]), float(pos[2])), quat
    if body_id is not None:
        pos = data.xpos[body_id]
        quat = data.xquat[body_id]
        return (float(pos[0]), float(pos[1]), float(pos[2])), (
            float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
        )
    raise RuntimeError("No body or site specified for TCP pose.")


def main() -> Optional[int]:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found at {args.model}")

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    site_id = None
    if args.site:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site)
        if site_id < 0:
            site_id = None

    body_id = None
    if site_id is None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.body)
        if body_id < 0:
            raise ValueError(f"Body '{args.body}' not found and site '{args.site}' not found.")

    targets: List[Tuple[float, float, float]] = []
    if args.targets_file:
        targets.extend(load_targets(args.targets_file))
    if args.targets:
        if len(args.targets) % 3 != 0:
            raise ValueError("--targets must have a multiple of 3 values (x y z ...).")
        for i in range(0, len(args.targets), 3):
            targets.append((args.targets[i], args.targets[i + 1], args.targets[i + 2]))
    if args.target:
        targets.extend([(t[0], t[1], t[2]) for t in args.target])

    if not targets:
        mujoco.mj_forward(model, data)
        if site_id is not None:
            cur = data.site_xpos[site_id]
        elif body_id is not None:
            cur = data.xpos[body_id]
        else:
            raise RuntimeError("No body or site specified for IK.")
        targets = [(cur[0] + 0.05, cur[1] - 0.05, cur[2] + 0.05)]

    joint_ids, dof_ids = build_hinge_dofs(model)
    actuator_joint_ids, actuator_dof_ids, actuator_ids, actuator_qpos = build_actuated_hinge_dofs(model)
    if actuator_dof_ids:
        joint_ids, dof_ids = actuator_joint_ids, actuator_dof_ids
    if not dof_ids:
        raise RuntimeError("No hinge joints found to solve IK.")

    if args.show:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            print("Viewer running. Close the window or press ESC to exit.")
            next_print = time.time()
            # Settle the model before IK.
            if args.settle_seconds > 0:
                settle_until = time.time() + args.settle_seconds
                while viewer.is_running() and time.time() < settle_until:
                    mujoco.mj_step(model, data)
                    viewer.sync()
                    if time.time() >= next_print:
                        mujoco.mj_forward(model, data)
                        pos, quat = get_tcp_pose(model, data, body_id, site_id)
                        print(
                            f"TCP (world): {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f} | "
                            f"quat: {quat[0]: .4f} {quat[1]: .4f} {quat[2]: .4f} {quat[3]: .4f}"
                        )
                        next_print = time.time() + max(1e-6, 1.0 / args.print_hz)

            for idx, target in enumerate(targets, start=1):
                if args.target_quat and args.target_rpy:
                    raise ValueError("Use only one of --target-quat or --target-rpy.")
                if args.target_quat:
                    target_quat = (args.target_quat[0], args.target_quat[1],
                                   args.target_quat[2], args.target_quat[3])
                elif args.target_rpy:
                    target_quat = rpy_to_quat(args.target_rpy[0], args.target_rpy[1], args.target_rpy[2])
                else:
                    mujoco.mj_forward(model, data)
                    _, target_quat = get_tcp_pose(model, data, body_id, site_id)

                start_qpos = data.qpos.copy()
                err = solve_ik(
                    model,
                    data,
                    body_id=body_id,
                    site_id=site_id,
                    target=target,
                    target_quat=target_quat,
                    iters=args.iters,
                    tol=args.tol,
                    damping=args.damping,
                    step=args.step,
                    w_pos=args.w_pos,
                    w_rot=args.w_rot,
                    joint_ids=joint_ids,
                    dof_ids=dof_ids,
                )
                target_qpos = data.qpos.copy()
                data.qpos[:] = start_qpos
                data.qvel[:] = 0.0
                mujoco.mj_forward(model, data)

                print(
                    f"[{idx}/{len(targets)}] Target pos: ({target[0]: .4f}, {target[1]: .4f}, {target[2]: .4f}) | "
                    f"Target quat: ({target_quat[0]: .4f}, {target_quat[1]: .4f}, "
                    f"{target_quat[2]: .4f}, {target_quat[3]: .4f})"
                )
                print(f"[{idx}/{len(targets)}] Final pos error: {err:.6f} m")

                # Smoothly move toward the target using actuator setpoints if available.
                move_until = time.time() + max(0.0, args.move_seconds)
                while viewer.is_running() and time.time() < move_until:
                    if actuator_ids:
                        alpha = 1.0 - max(0.0, move_until - time.time()) / max(args.move_seconds, 1e-6)
                        ctrl_vals = [
                            start_qpos[qadr] + alpha * (target_qpos[qadr] - start_qpos[qadr])
                            for qadr in actuator_qpos
                        ]
                        set_ctrls(model, data, actuator_ids, ctrl_vals)
                        mujoco.mj_step(model, data)
                    else:
                        alpha = 1.0 - max(0.0, move_until - time.time()) / max(args.move_seconds, 1e-6)
                        data.qpos[:] = start_qpos + alpha * (target_qpos - start_qpos)
                        mujoco.mj_forward(model, data)
                    viewer.sync()
                    if time.time() >= next_print:
                        mujoco.mj_forward(model, data)
                        pos, quat = get_tcp_pose(model, data, body_id, site_id)
                        print(
                            f"TCP (world): {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f} | "
                            f"quat: {quat[0]: .4f} {quat[1]: .4f} {quat[2]: .4f} {quat[3]: .4f}"
                        )
                        next_print = time.time() + max(1e-6, 1.0 / args.print_hz)

                hold_until = time.time() + max(0.0, args.hold_seconds)
                while viewer.is_running() and time.time() < hold_until:
                    if actuator_ids:
                        ctrl_vals = [target_qpos[qadr] for qadr in actuator_qpos]
                        set_ctrls(model, data, actuator_ids, ctrl_vals)
                        mujoco.mj_step(model, data)
                    else:
                        mujoco.mj_step(model, data)
                    viewer.sync()
                    if time.time() >= next_print:
                        mujoco.mj_forward(model, data)
                        pos, quat = get_tcp_pose(model, data, body_id, site_id)
                        print(
                            f"TCP (world): {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f} | "
                            f"quat: {quat[0]: .4f} {quat[1]: .4f} {quat[2]: .4f} {quat[3]: .4f}"
                        )
                        next_print = time.time() + max(1e-6, 1.0 / args.print_hz)
            # Keep viewer open after last target
            while viewer.is_running():
                if actuator_ids:
                    ctrl_vals = [data.qpos[qadr] for qadr in actuator_qpos]
                    set_ctrls(model, data, actuator_ids, ctrl_vals)
                mujoco.mj_step(model, data)
                viewer.sync()
                if time.time() >= next_print:
                    mujoco.mj_forward(model, data)
                    pos, quat = get_tcp_pose(model, data, body_id, site_id)
                    print(
                        f"TCP (world): {pos[0]: .4f} {pos[1]: .4f} {pos[2]: .4f} | "
                        f"quat: {quat[0]: .4f} {quat[1]: .4f} {quat[2]: .4f} {quat[3]: .4f}"
                    )
                    next_print = time.time() + max(1e-6, 1.0 / args.print_hz)
    else:
        for idx, target in enumerate(targets, start=1):
            if args.target_quat and args.target_rpy:
                raise ValueError("Use only one of --target-quat or --target-rpy.")
            if args.target_quat:
                target_quat = (args.target_quat[0], args.target_quat[1],
                               args.target_quat[2], args.target_quat[3])
            elif args.target_rpy:
                target_quat = rpy_to_quat(args.target_rpy[0], args.target_rpy[1], args.target_rpy[2])
            else:
                mujoco.mj_forward(model, data)
                _, target_quat = get_tcp_pose(model, data, body_id, site_id)

            err = solve_ik(
                model,
                data,
                body_id=body_id,
                site_id=site_id,
                target=target,
                target_quat=target_quat,
                iters=args.iters,
                tol=args.tol,
                damping=args.damping,
                step=args.step,
                w_pos=args.w_pos,
                w_rot=args.w_rot,
                joint_ids=joint_ids,
                dof_ids=dof_ids,
            )
            print(
                f"[{idx}/{len(targets)}] Target pos: ({target[0]: .4f}, {target[1]: .4f}, {target[2]: .4f}) | "
                f"Target quat: ({target_quat[0]: .4f}, {target_quat[1]: .4f}, "
                f"{target_quat[2]: .4f}, {target_quat[3]: .4f})"
            )
            print(f"[{idx}/{len(targets)}] Final pos error: {err:.6f} m")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
