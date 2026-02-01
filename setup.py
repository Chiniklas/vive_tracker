from pathlib import Path

from setuptools import find_packages, setup


here = Path(__file__).parent
readme = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="steamvr-tracking",
    version="0.1.0",
    description="Extract and visualize HTC Vive tracker pose data via OpenVR.",
    long_description=readme,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    packages=find_packages(),
    install_requires=[
        "openvr",
        "pyquaternion",
        "matplotlib",
        "numpy",
        "mujoco>=2.3",
    ],
    keywords=["steamvr", "vive", "openvr", "tracking", "robotics"],
    entry_points={
        "console_scripts": [
            "tracker-test=tests.tracker_test:main",
            "tcp-visualizer=tests.TCP_visualizer:main",
        ]
    },
)
