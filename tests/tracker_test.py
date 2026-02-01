"""Simple CLI to stream live tracker pose to stdout.

Usage:
    python tests/tracker_test.py [hz]

If `hz` is omitted, defaults to 250 Hz. Output shows position (x y z, meters)
followed by quaternion (w x y z).
"""

import sys
import time

import device.triad_openvr as triad_openvr


def format_pose(pose):
    """Return a compact string for xyz + quaternion."""
    xyz = pose[:3]
    quat = pose[3:]
    xyz_txt = " ".join(f"{v:.4f}" for v in xyz)
    quat_txt = " ".join(f"{v:.4f}" for v in quat)
    return f"xyz: {xyz_txt} | quat: {quat_txt}"


def main():
    if len(sys.argv) == 1:
        interval = 1 / 250
    elif len(sys.argv) == 2:
        try:
            hz = float(sys.argv[1])
            interval = 1 / hz
        except ValueError:
            print("Argument must be a number (Hz)")
            return
    else:
        print("Usage: python tests/tracker_test.py [hz]")
        return

    vr = triad_openvr.triad_openvr()
    # vr.print_discovered_objects()

    while True:
        start = time.time()
        pose = vr.devices.get("tracker_1")
        if pose:
            pose_quat = pose.get_pose_quaternion()
            if pose_quat:
                print("\r" + format_pose(pose_quat), end="")
        sleep_time = interval - (time.time() - start)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()
