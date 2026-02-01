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
from matplotlib.animation import FuncAnimation

import device.triad_openvr as triad_openvr


DEFAULT_HZ = 50
WINDOW_SECONDS = 10  # seconds of history to show


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

    vr = triad_openvr.triad_openvr()
    tracker = vr.devices.get("tracker_1")
    if tracker is None:
        print("No tracker_1 found. Make sure the Vive tracker is connected.")
        return

    max_samples = int(WINDOW_SECONDS * hz)
    xs, ys, zs = deque(maxlen=max_samples), deque(maxlen=max_samples), deque(maxlen=max_samples)

    fig = plt.figure("TCP Visualizer")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.view_init(elev=20, azim=45)
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
    scatter = ax.scatter([], [], [], c="tab:blue", s=20)

    def update(_frame):
        start = time.time()
        pose_quat = tracker.get_pose_quaternion()
        if pose_quat:
            x, y, z = pose_quat[:3]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            scatter._offsets3d = (xs, ys, zs)
        # throttle to desired interval
        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return scatter,

    ani = FuncAnimation(fig, update, interval=1, blit=False)
    plt.tight_layout()
    print(f"Streaming tracker_1 at ~{hz:.1f} Hz. Close the window or Ctrl+C to exit.")
    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
