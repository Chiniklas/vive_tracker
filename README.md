# Extracting data from SteamVR using HTC VIVE tracker

The goal of this project is to extract useful pose data using the HTC VIVE tracker to pass to the UR5 robot arm through Rosvita, so that the robot arm could imitate the motion of the tracker. The program is designed for windows and only works on windows due to the compatibility of SteamVR.

The program uses triad_openvr.py script from [triad_openvr](https://github.com/TriadSemi/triad_openvr).

## Getting Started

These instructions will illustrate how to set up the environment required for the program to run.

### Setting up the Base Station

Mount the base station in your room (recommend using a tripod). Use one base station should be enough. Here are some things that you need to pay attention to when you mount the base station:

- Mount the base station high up in the room (above your height). The tracker cannot be detected if the base station is too low or too close to the tracker.

- Make sure that the base station is plugged in and mounted stably, any movement of the base station will disconnect it from the tracker.

- Change the mode of the base station to “b” by using the button on the back.

### Prerequisites

This section is adapted from [SteamVR Tracking without an HMD](http://help.triadsemi.com/en/articles/836917-steamvr-tracking-without-an-hmd). All the detailed instructions could be found on the website.

1. In order for the program to run, you must have the latest version of SteamVR, with opted in for the beta, then change some settings in order to track without an HMD:

	1. Download [Steam](https://store.steampowered.com/about/) to your Windows PC. Create an account and log into Steam.  
	1. Locate SteamVR in the Library tab, and download it.
	1. Right click on SteamVR, select Properties-> Beta, change the first option to "beta - SteamVR Beta Update" in order to opt in for the beta.
	1. After Steam finishes downloading SteamVR beta, right click on SteamVR, select Properties-> Local Files-> Browse Local Files to browse local files.
	1. Locate this file and open it with a text editor:
			steamapps/common/SteamVR/drivers/null/resources/settings/default.settings
	1. Change “enable” to true

	1. Save the file and close it. Then locate this file:
			steamapps/common/SteamVR/resources/settings/default.settings

	1. Set the following keys under “steamvr” to the followings:
			“requireHmd”: false;
			“forcedDriver”: “null”;
			“activateMultipleDrivers”: true;
	1. Save this default.vrsettings file and close
	1. Once successfully set up SteamVR, you can now hit “Play” to start SteamVR. If SteamVR is running, close and restart it. you will see that it is now possible to connect a tracker or controller without the HMD.

2. Use Python 3.10 via a conda env (recommended):

	```
	conda create -y -n vive_tracker python=3.10
	conda activate vive_tracker
	```

3. Install dependencies (openvr, pyquaternion, matplotlib, numpy, **mujoco**) with:

	```
	pip install -e .
	```

4. Now you have installed all the required prerequisites for this program. To have the program running on local environments, simply clone this GitHub repository to a local folder, and you are ready to run the program.

## Running the program

Before running the program, you first need to run SteamVR, then connect the VIVE tracker to your computer(either using the USB cable, or wirelessly using the dongle and dongle cradle). Hold the button on the tracker for one second to turn on the tracker. Make sure that there is nothing between the base station and the tracker, so that the IR light sent by the base station could be detected by the tracker. When everything is connected, you will see that the base station symbol and the tracker symbol of steamVR turn green(without flashing). The light on the tracker should also turn green when it is tracking. Note that the “Not Ready” text is normal.

### Examine the real-time pose information of tracker using `tests/tracker_test.py`

Once both the tracker and the base station are connected, you can run the following command from the repo root to check the real-time pose data of the tracker:

```
python tests/tracker_test.py
```

As the script executes, you will see numbers updating at 250Hz. The first three numbers are the Cartesian coordinates of the tracker in the order of X, Y, Z. The next four numbers are the quaternions of the tracker in the order of w, x, y, z.

The coordinates of the VR world are set up like this: when facing towards the base station, in your front is the +z direction, above you is the +y direction, and to your left is the +x direction.

## Coordinate frames (MuJoCo + Human)

For the MuJoCo tests we use a robotics-style, right-handed human frame:

- Origin: operator pelvis
- +X: right
- +Y: forward
- +Z: up

The MuJoCo scene includes a `human_frame` marker (RGB axis triad + small sphere) in
`asset/inspirehand/handright9253_simplified.xml`. It is a `mocap` body so you can
place it during calibration. By default it sits at the MuJoCo world origin.

### Tracker calibration (human pelvis frame)

Use `tests/test_wrist_tracking.py --calibrate` to compute a rigid transform from
SteamVR space into the robotics human frame (origin at pelvis, +X right, +Y forward, +Z up).

Run:

```
python tests/test_wrist_tracking.py --calibrate
```

Follow the prompts in order (return to pelvis origin each time):

1) Pelvis origin, palm down, fingers forward → press Enter  
2) Return to pelvis, then move +X (right) → press Enter  
3) Return to pelvis, then move +Y (forward) → press Enter  
4) Return to pelvis, then move +Z (up) → press Enter  
5) Return to pelvis, rotate +X (roll, right-hand rule) → press Enter  # roll backwards
6) Return to pelvis, rotate +Z (yaw, right-hand rule) → press Enter # roll anti-clockwise

This saves `tests/tracker_calibration.json`. Subsequent runs apply it automatically.
Steps 5–6 let the script auto-detect rotation sign flips and store them in the calibration.

## Coordinate transforms (basics)

We work with five frames:

1) Human frame (H): pelvis origin, +X right, +Y forward, +Z up.  
2) Hand frame (Hand): `hand_root` in MuJoCo (model-local).  
3) VR base frame (VR): SteamVR tracking space.  
4) VR tracker frame (T): tracker body frame (quaternion is defined in this frame).  
5) MuJoCo world frame (MJ): MuJoCo scene/world frame.

Target relationship: **H and MJ coincide** (same origin + axes). In this project,
the human frame is the absolute reference, and the MuJoCo world is aligned to it.

The tracker gives us `T_VR` (tracker pose in VR). We need two transforms:

- `X_VR→H`: calibration transform from VR space into human frame (computed by calibration).
- `X_T→Hand`: fixed tracker-to-hand mounting offset (depends on how the tracker is attached).

Then the hand pose in the human/MuJoCo world frame is:

```
Hand_pose_H = X_VR→H · T_VR · X_T→Hand
```

If the tracker marker aligns with the human axes but the hand is still “off,”
you likely need to define `X_T→Hand` (tracker mounting offset).

### (legacy) Record a session of pose movement and send it to robot arm

#### Recording data on the windows PC using `utils/record_track.py`

In order to record a session of pose movement, run the following command after both base station and tracker are connected:

```
python utils/record_track.py [duration = 30] [interval = 0.05]
```

## Acknowledgments

* Thanks to Professor Daniel Scharstein from Middlebury College for overseeing this project.

* The program triad_openvr.py and tracker_test.py are downloaded and adapted from [triad_openvr](https://github.com/TriadSemi/triad_openvr).
