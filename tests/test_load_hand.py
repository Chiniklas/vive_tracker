"""Quick loader for the Inspire hand MJCF in MuJoCo.

Usage:
    python tests/test_load_hand.py           # launches interactive viewer
    python tests/test_load_hand.py --headless --steps 10

Requirements:
    - mujoco>=2.3 (PyPI package `mujoco`)
    - Optional viewer dependencies (GLFW) for interactive mode
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import mujoco
    import mujoco.viewer
except Exception as exc:  # pragma: no cover - import-time diagnostics for users
    print("Failed to import mujoco. Install with `pip install mujoco`.", file=sys.stderr)
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_PATH = REPO_ROOT / "asset" / "inspirehand" / "handright9253_fixed.mjcf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Inspire hand MJCF in MuJoCo.")
    parser.add_argument("--headless", action="store_true", help="Run simulation without opening a viewer.")
    parser.add_argument("--steps", type=int, default=500, help="Steps to simulate (headless).")
    parser.add_argument("--dt", type=float, default=0.002, help="Simulation timestep (seconds).")
    return parser.parse_args()


def load_model(path: Path) -> mujoco.MjModel:
    if not path.is_file():
        raise FileNotFoundError(f"MJCF not found at {path}")
    return mujoco.MjModel.from_xml_path(str(path))


def run_headless(model: mujoco.MjModel, steps: int, dt: float) -> None:
    data = mujoco.MjData(model)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    print(f"Headless run complete: {steps} steps at dt={dt}s")


def run_viewer(model: mujoco.MjModel) -> None:
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Viewer running. Close the window or press ESC to exit.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def main() -> Optional[int]:
    args = parse_args()
    model = load_model(ASSET_PATH)
    # Ensure the model's timestep matches requested dt if provided
    if args.dt is not None:
        model.opt.timestep = args.dt

    if args.headless:
        run_headless(model, steps=args.steps, dt=args.dt)
    else:
        run_viewer(model)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
