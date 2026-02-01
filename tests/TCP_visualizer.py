"""Real-time 3D visualization of tracker TCP position.

Usage:
    python tests/TCP_visualizer.py [hz]

- `hz` (optional) sets the sampling frequency; defaults to 50 Hz.
- Displays a 3D scatter plot of the tracker position in meters.
- Press Ctrl+C to quit.

Requirements: matplotlib, openvr, pyquaternion (same as the rest of the repo).
"""

import sys
import time
from collections import deque

import matplotlib.pyplot as plt
import signal
import numpy as np
from pyquaternion import Quaternion

import device.triad_openvr as triad_openvr
from tests.tracker_test import format_pose


DEFAULT_HZ = 50
WINDOW_SECONDS = 10  # seconds of history to show
# Remap: x' = -x, y' = z, z' = y
TRANSFORM_MAT = np.array(
    [
        [-1.0, 0.0, 0.0],  # x' = -x
        [0.0, 0.0, 1.0],   # y' = z
        [0.0, 1.0, 0.0],   # z' = y
    ]
)
TRANSFORM_QUAT = Quaternion(matrix=TRANSFORM_MAT)


def parse_args():
    if len(sys.argv) == 1:
        return DEFAULT_HZ
    if len(sys.argv) == 2:
        try:
            return float(sys.argv[1])
        except ValueError:
            print("Argument must be a number (Hz)")
            sys.exit(1)
    print("Usage: python tests/TCP_visualizer.py [hz]")
    sys.exit(1)


def main():
    hz = parse_args()
    interval = 1.0 / hz
    interval_ms = interval * 1000

    vr = triad_openvr.triad_openvr()

    max_samples = int(WINDOW_SECONDS * hz)
    xs, ys, zs = deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples)

    fig = plt.figure("TCP Visualizer")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (from -X) (m)")
    ax.set_ylabel("Y (from +Z) (m)")
    ax.set_zlabel("Z (from +Y) (m)")
    ax.view_init(elev=20, azim=45)
    ax.grid(True)
    ax.set_axis_on()
    # Keep a fixed -2–2 m cube so trajectories are easy to read with origin at an inner corner.
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    try:
        # Matplotlib ≥3.3
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        # Older matplotlib: already square by matching limits above.
        pass
    scatter = ax.scatter([], [], [], c="tab:blue", s=0) # default size 2

    # Static origin triad to debug world frame.
    origin_len = 0.4
    # Red points -X, green shows Z, blue shows Y (color swap + red flip)
    ax.quiver(0, 0, 0, -origin_len, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, origin_len, color="g", linewidth=2)  # Z
    ax.quiver(0, 0, 0, 0, origin_len, 0, color="b", linewidth=2)  # Y

    # Close the window cleanly on Ctrl+C.
    def handle_sigint(_sig, _frame):
        plt.close(fig)
    signal.signal(signal.SIGINT, handle_sigint)

    tracker_quivers = []

    def get_tracker():
        vr.poll_vr_events()
        return vr.devices.get("tracker_1")

    def update_tracker_axes(position, quat):
        nonlocal tracker_quivers
        # Remove old axes
        for q in tracker_quivers:
            q.remove()
        tracker_quivers = []
        # Build rotation matrix from quaternion (w, x, y, z)
        R = Quaternion(quat).rotation_matrix
        v_x, v_y, v_z = R[:, 0], R[:, 1], R[:, 2]
        length = 0.2
        # Red points -X, green shows Z, blue shows Y
        axis_defs = [
            (-v_x, "r"),
            (v_z, "g"),
            (v_y, "b"),
        ]
        for vec, color in axis_defs:
            tracker_quivers.append(
                ax.quiver(
                    position[0],
                    position[1],
                    position[2],
                    vec[0] * length,
                    vec[1] * length,
                    vec[2] * length,
                    color=color,
                    linewidth=2,
                )
            )

    plt.tight_layout()
    print(f"Streaming tracker_1 at ~{hz:.1f} Hz. Close the window or Ctrl+C to exit.")

    try:
        while plt.fignum_exists(fig.number):
            tracker = get_tracker()
            if tracker is None:
                sys.stdout.write("\rWaiting for tracker_1...")
                sys.stdout.flush()
                plt.pause(interval)
                continue

            pose_quat = tracker.get_pose_quaternion()
            if pose_quat:
                pos_world = np.array(pose_quat[:3])
                quat_world = Quaternion(pose_quat[3:])

                pos_disp = TRANSFORM_MAT.dot(pos_world)
                quat_disp = TRANSFORM_QUAT * quat_world * TRANSFORM_QUAT.inverse

                xs.append(pos_disp[0])
                ys.append(pos_disp[1])
                zs.append(pos_disp[2])
                scatter._offsets3d = (xs, ys, zs)

                # Update moving coordinate frame using transformed pose
                update_tracker_axes(pos_disp, quat_disp.elements)

                # Print live data (unchanged, original frame)
                sys.stdout.write("\r" + format_pose(pose_quat))
                sys.stdout.flush()
            plt.pause(interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
