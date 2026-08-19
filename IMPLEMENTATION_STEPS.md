# IsaacTeleop + Isaac Lab Local Teleoperation Steps

This note records the local implementation path for running NVIDIA IsaacTeleop with an Isaac Lab simulation.

## 0. Local Status

- OS: Ubuntu 22.04 x86_64
- GLIBC: 2.35
- GPU: NVIDIA GeForce RTX 5070 Ti, 16 GB VRAM
- Driver: 580.173.02
- Conda: available
- Python in base env: 3.13.9
- System `sudo apt` requires an interactive password in this session.
- `git`, `git-lfs`, and `cmake` were installed inside the Conda environment instead.

Isaac Sim 5.x requires Python 3.11, so use a dedicated conda environment.

## 1. Install System Tools

```bash
sudo apt update
sudo apt install -y git git-lfs cmake build-essential curl wget
git lfs install
```

## 2. Create Python Environment

```bash
conda create -n env_isaaclab python=3.11 -y
conda activate env_isaaclab
python -m pip install --upgrade pip
```

For this local setup, the environment was created with user-space tools:

```bash
conda create -n env_isaaclab python=3.11 git git-lfs cmake -c conda-forge -y
conda activate env_isaaclab
git lfs install
```

## 3. Install Isaac Sim

```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
isaacsim
```

Accept the NVIDIA Omniverse EULA on first launch.

## 4. Clone and Install Isaac Lab

```bash
mkdir -p ~/PhdResearch/DualArmTeleopration/external
cd ~/PhdResearch/DualArmTeleopration/external
git clone https://github.com/isaac-sim/IsaacLab.git --branch main
cd IsaacLab
./isaaclab.sh --install
```

Verify:

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

## 5. Clone and Install IsaacTeleop

```bash
cd ~/PhdResearch/DualArmTeleopration/external
git clone https://github.com/NVIDIA/IsaacTeleop.git
cd IsaacTeleop
git checkout v1.3.131
pip install "isaacteleop[cloudxr,retargeters]==1.3.131" --extra-index-url https://pypi.nvidia.com
```

The repository checkout should match the installed package version. The `main` branch examples may use newer APIs than the PyPI release.

For this local environment, these variables are installed into the Conda activate hook:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export NV_CXR_RUNTIME_DIR=/home/yan/.cloudxr/run
export XR_RUNTIME_JSON=/home/yan/.cloudxr/openxr_cloudxr.json
```

Start CloudXR runtime and hosted client:

```bash
cd ~/PhdResearch/DualArmTeleopration/external/IsaacTeleop
python -m isaacteleop.cloudxr --accept-eula --host-client
```

The verified local client URL during setup was:

```text
https://10.16.221.139:48322/client/
```

Quick pipeline check without Isaac Lab:

```bash
python examples/teleop/python/gripper_retargeting_example_simple.py
```

This check requires the CloudXR runtime process to be running and an XR system/client available. Without a connected headset/client, OpenXR may fail at `xrGetSystem` with runtime/system unavailable.

Open the client in a desktop browser or XR headset:

```text
https://nvidia.github.io/IsaacTeleop/client
```

Enter the workstation IP, accept the self-signed certificate, then connect.

## 6. Firewall Ports for Quest/PICO Web Client

```bash
sudo ufw allow 47998/udp
sudo ufw allow 49100,48322/tcp
sudo ufw allow 8080,8443/tcp
```

## 7. First Isaac Lab Teleoperation Run

Keyboard, no XR:

```bash
cd ~/PhdResearch/DualArmTeleopration/external/IsaacLab
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device keyboard
```

SpaceMouse:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --teleop_device spacemouse
```

XR hand tracking / CloudXR:

```bash
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --teleop_device handtracking \
  --device cpu
```

In Isaac Sim UI, open the AR panel, select OpenXR/System OpenXR Runtime, and click Start AR.

## 8. Record Demonstrations

```bash
mkdir -p datasets
./isaaclab.sh -p scripts/tools/record_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --device cpu \
  --teleop_device handtracking \
  --dataset_file ./datasets/dataset.hdf5 \
  --num_demos 10
```

Replay:

```bash
./isaaclab.sh -p scripts/tools/replay_demos.py \
  --task Isaac-Stack-Cube-Franka-IK-Abs-v0 \
  --device cpu \
  --dataset_file ./datasets/dataset.hdf5
```

## 9. Development Hooks for Custom Dual-Arm Work

For a custom dual-arm teleoperation task, implement these layers in order:

1. Isaac Lab task: robot USD/URDF asset, scene, object assets, controllers, reset logic.
2. Teleop device: keyboard/SpaceMouse first, CloudXR handtracking after the task is stable.
3. Retargeting: map left/right hand or controller poses to left/right robot end-effector targets.
4. IK/control: differential IK, operational-space control, or a Pinocchio/Pink based solver for full-body or bimanual robots.
5. Dataset: record HDF5 demos, replay them, then feed Isaac Lab Mimic or robomimic.

## 9.1 ESROBO Fixed-Base Bimanual Teleoperation

Current ESROBO assets and configs:

```text
USD:
  assets/esrobo/esrobo_waist_with_head.usd

Robot cfg:
  src/dual_arm_teleop/isaaclab_ext/assets/esrobo.py

Pink IK cfg:
  src/dual_arm_teleop/isaaclab_ext/configs/esrobo_pink_controller_cfg.py

Project Pink IK action/controller:
  src/dual_arm_teleop/isaaclab_ext/actions/pink_actions.py
  src/dual_arm_teleop/isaaclab_ext/controllers/pink_ik.py

Task cfg:
  src/dual_arm_teleop/isaaclab_ext/tasks/esrobo/robot_only_env_cfg.py
  src/dual_arm_teleop/isaaclab_ext/tasks/esrobo/pick_place_env_cfg.py

UDP body-tracking device:
  src/dual_arm_teleop/isaaclab_ext/devices/udp_bimanual_body_device.py

OpenXR retargeter:
  src/dual_arm_teleop/isaaclab_ext/retargeters/esrobo_upper_body_retargeter.py

Teleoperation entry script:
  scripts/esrobo_teleop_se3_agent.py
```

The project registers these task ids outside the official IsaacLab task package:

```text
DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0
DualArmTeleop-ESROBO-PickPlace-BimanualIK-Abs-v0
```

Action layout:

```text
leftHand_link pose: 7 values, xyz + qw qx qy qz
rightHand_link pose: 7 values, xyz + qw qx qy qz
left active hand joints: 10 values
right active hand joints: 10 values

Total: 34 values
```

Pink IK settings:

```text
base_link_name = "waist_link3"
controlled_frames = ["leftHand_link", "rightHand_link"]
pink_controlled_joint_names = 14 arm joints
hand_joint_names = 20 active hand joints
num_hand_joints = 20
```

Smoke test:

```bash
cd ~/PhdResearch/DualArmTeleopration/external/IsaacLab
PYTHONPATH=../../src:${PYTHONPATH:-} \
./isaaclab.sh -p -c 'import pinocchio; from isaaclab.app import AppLauncher; app_launcher = AppLauncher(headless=True); simulation_app = app_launcher.app; import gymnasium as gym; import isaaclab_tasks; import dual_arm_teleop.isaaclab_ext.tasks; from isaaclab_tasks.utils import parse_env_cfg; task="DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0"; env_cfg=parse_env_cfg(task, device="cuda:0", num_envs=1); env=gym.make(task, cfg=env_cfg).unwrapped; env.reset(); print("ENV_RESET_OK action_dim=" + str(env.action_manager.total_action_dim), flush=True); env.close(); simulation_app.close()'
```

Run with OpenXR hand tracking:

```bash
cd ~/PhdResearch/DualArmTeleopration
./scripts/run_esrobo_xr_teleop_lowres.sh \
  --task DualArmTeleop-ESROBO-RobotOnly-BimanualIK-Abs-v0 \
  --teleop_device handtracking \
  --xr_start_active
```

Low-resolution CloudXR/PICO test launcher:

```bash
cd ~/PhdResearch/DualArmTeleopration
./scripts/run_esrobo_xr_teleop_lowres.sh
```

The script prints a headset URL similar to:

```text
https://<host-ip>:48322/client/?perEyeWidth=1280&perEyeHeight=1280&deviceFrameRate=72&maxStreamingBitrateMbps=60&codec=h265&immersiveMode=vr#/sim/isaaclab
```

Open that URL in the PICO browser, press `Connect`, hold both hands at the desired teleoperation zero pose, then press `Play`. Pressing `Play` resets the ESROBO retargeter calibration; the next valid left/right wrist poses become the zero point, and subsequent hand motion is sent as relative motion from that point. In XR mode the Isaac Lab script deliberately starts with robot control paused; hand markers can move while the robot stays still until the client sends `start teleop`.

Run with PICO full-body arm tracking for ESROBO dual-arm teleoperation without IsaacLab XR image streaming. IsaacLab receives UDP body poses, uses shoulder/elbow/wrist vectors for retargeting, and displays the robot motion in the Isaac Sim window.

Terminal 1, start the CloudXR/OpenXR runtime and connect the PICO client:

```bash
cd ~/PhdResearch/DualArmTeleopration
conda activate env_isaaclab
python -m isaacteleop.cloudxr --accept-eula --host-client
```

Terminal 2, start the ESROBO IsaacLab simulation locally. The launcher activates `env_isaaclab` automatically:

```bash
cd ~/PhdResearch/DualArmTeleopration

ESROBO_BODY_UDP_HOST=0.0.0.0 \
ESROBO_BODY_UDP_PORT=15050 \
ESROBO_BODY_RETARGETING_MODE=arm_vector \
ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS=1 \
ESROBO_BODY_POSITION_DELTA_SIGNS="-1 -1 1" \
ESROBO_BODY_CALIBRATION_DELAY_S=10.0 \
ESROBO_BODY_CALIBRATION_SAMPLE_START_S=6.0 \
./scripts/run_esrobo_bodytracking_teleop.sh
```

Terminal 3, stream PICO full-body data to IsaacLab. The launcher activates `env_isaaclab` automatically:

```bash
cd ~/PhdResearch/DualArmTeleopration

ESROBO_BODY_UDP_HOST=127.0.0.1 \
ESROBO_BODY_UDP_PORT=15050 \
./scripts/run_pico_full_body_to_esrobo_teleop_bridge.sh \
  --rate-hz 60 \
  --print-interval 1 \
  --include-full-body-arrays
```

Calibration flow:

1. Step 1/2: hold both arms naturally down for 10 seconds.
2. Step 2/2: raise and stretch both arms horizontally in front of the chest for 10 seconds.
3. For each pose, the first 0-6 seconds are preparation time and are not recorded.
4. Only the valid body-tracking frames from 6-10 seconds are used for calibration.
5. Start normal teleoperation only after IsaacLab prints `Calibration locked`.

The current ESROBO body-tracking launcher defaults to `ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS=1`, so the robot target mapping is `robot_left<=human_right` and `robot_right<=human_left`. Set `ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS=0` only if the physical robot side mapping is already correct.

The launcher defaults to `ESROBO_BODY_RETARGETING_MODE=arm_vector`. In this mode, each arm target is generated from full-body `shoulder -> elbow -> wrist` vectors: position follows the calibrated shoulder-to-wrist reach delta, and orientation follows the arm-vector frame delta. If shoulder/elbow/wrist frames are not available, the device temporarily falls back to `wrist_delta`.

The launcher also defaults to `ESROBO_BODY_POSITION_DELTA_SIGNS="-1 -1 1"`, which only affects post-calibration end-effector position deltas: robot-frame forward/backward and left/right motion are inverted, while vertical motion and tracker orientation use the normal source-to-robot transform.

The current response profile is tuned for lower latency than the first bring-up: `ESROBO_IK_MAX_JOINT_POSITION_DELTA=0.24`, `ESROBO_IK_FRAME_TASK_GAIN=1.0`, `ESROBO_IK_FRAME_TASK_LM_DAMPING=1.5`, `ESROBO_ARM_EFFORT_LIMIT_SIM=900`, `ESROBO_ARM_VELOCITY_LIMIT_SIM=24`, and `ESROBO_ARM_STIFFNESS/ESROBO_ARM_DAMPING=1800/90`. If the robot is still lagging, test `ESROBO_IK_MAX_JOINT_POSITION_DELTA=0.35 ESROBO_ARM_VELOCITY_LIMIT_SIM=36`; if it jitters or overshoots, reduce those two first.

If the bridge and IsaacLab run on different machines, bind IsaacLab on all interfaces and send packets to the IsaacLab workstation IP:

```bash
ESROBO_BODY_UDP_HOST=0.0.0.0 ESROBO_BODY_UDP_PORT=15050 ./scripts/run_esrobo_bodytracking_teleop.sh
ESROBO_BODY_UDP_HOST=<isaaclab-workstation-ip> ESROBO_BODY_UDP_PORT=15050 ./scripts/run_pico_full_body_to_esrobo_teleop_bridge.sh --include-full-body-arrays
```

Fallback wrist-tracker mode for debugging:

```bash
ESROBO_BODY_RETARGETING_MODE=wrist_delta ./scripts/run_esrobo_bodytracking_teleop.sh
ESROBO_BODY_UDP_HOST=127.0.0.1 ESROBO_BODY_UDP_PORT=15050 ./scripts/run_xrobotoolkit_tracker_to_esrobo_teleop_bridge.sh --left-serial PC2310MLKC090749G --right-serial PC2310MLKC090321G --print-poses
```

To tune without editing files:

```bash
PER_EYE_WIDTH=1280 PER_EYE_HEIGHT=1280 DEVICE_FPS=72 BITRATE_MBPS=60 ./scripts/run_esrobo_xr_teleop_lowres.sh
PER_EYE_WIDTH=1536 PER_EYE_HEIGHT=1344 DEVICE_FPS=90 BITRATE_MBPS=100 ./scripts/run_esrobo_xr_teleop_lowres.sh
./scripts/run_esrobo_xr_teleop_lowres.sh --xr_start_active
```

Use `--xr_start_active` only for debugging, because it bypasses the headset `Play` safety gate and can command a sudden first motion if the XR hands are not aligned.

Runtime notes for the current ESROBO bring-up:

- The ESROBO scene keeps the pick-place table and object outside the initial arm workspace. With the original table placement, the robot arms start in contact/intersection and PhysX can move joints even when the IK target is the current hand pose.
- The ESROBO actuator parameters are intentionally conservative: gravity is disabled, arm stiffness is low, armature is increased, and Pink IK clamps each step. Raise stiffness/speed only after the XR loop is stable.
- The ESROBO OpenXR handtracking retargeter draws `/Visuals/esrobo_ar_human_pose`: left hand joints/bones are blue, right hand joints/bones are orange, head is green, and the shoulder/elbow helper skeleton is pale blue. The handtracking path estimates shoulder/elbow helper links.
- The PICO full-body UDP path draws `/Visuals/esrobo_body_capture_pose`: shoulders, elbows, wrists, neck, and head are shown beside the robot after the two-pose calibration is locked. Robot control uses full-body arm vectors as the bimanual end-effector targets.
- For `handtracking`, if hand markers move but the robot does not, check whether the headset client has sent `Play`. The teleop script prints a waiting message while it is paused.
- For `handtracking`, if the trajectory is offset after moving around, stop teleoperation, return both hands to the desired zero pose, and press `Play` again to recalibrate.
- For `bodytracking_udp`, if the trajectory is offset after moving around, press `R` in the IsaacLab window or restart the teleop process, then repeat the two-pose calibration. Each pose lasts 10 seconds, and only the 6-10 second valid window is recorded.
- If `Play` is pressed but the robot still holds position, inspect `[teleop debug] action=...`; the first 14 values should change when the left/right wrist pose changes.
- Isaac Sim warned that the CPU governor is `powersave`; switching the workstation to a performance governor can improve XR frame pacing.

## 10. Local Verification Results

Completed:

```text
Python: 3.11.15
Torch: 2.7.0+cu128
CUDA visible: yes
GPU: NVIDIA GeForce RTX 5070 Ti
isaacsim import: ok
isaacteleop import: ok
Isaac Lab smoke: reset and 10 sim steps completed
CloudXR runtime: started successfully during verification
```

Notes:

- `isaaclab.sh --install` could not run directly because it calls `sudo apt-get`; Isaac Lab source extensions were installed manually with pip editable installs.
- Full `source/isaaclab_rl[all]` and `source/isaaclab_mimic[all]` extras were not completed because dependency resolution attempted to upgrade Torch to a newer CUDA 13 stack. This is not needed for the teleoperation path.
- Isaac Lab smoke test completed reset and 10 sim steps. Isaac Sim shutdown was slow and the validation command was terminated by `timeout` after the success message.
- IsaacTeleop gripper smoke reached OpenXR instance creation. Without an active headset/client, it stopped at OpenXR system discovery.

## 11. Common Failures

- `git: command not found`: install `git`.
- `ModuleNotFoundError: isaacsim`: wrong environment or Isaac Sim not installed in the active Python.
- Python mismatch: Isaac Sim 5.x needs Python 3.11.
- CloudXR client cannot connect: check IP reachability and ports 47998/udp, 49100/tcp, 48322/tcp.
- SpaceMouse not detected: inspect `/dev/hidraw*` and grant permissions to the matching device.
- XR task unstable: first verify keyboard task, then use handtracking with the absolute task variant.
