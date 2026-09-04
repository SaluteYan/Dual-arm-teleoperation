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

## 9.2 SenseGlove 手指 + IMU 实时遥操作

本节对应当前的 SenseGlove Nova 2 实机链路。该模式只控制 ESROBO 的双手手指和腕部朝向：

- 不需要 PICO/XRoboToolkit。
- 双臂关节 1-4 保持初始状态。
- 腕部关节 5-7 由 SenseGlove IMU 控制。
- 每只灵巧手的 10 个主动关节由 SenseGlove 手部骨架控制。
- IsaacLab 同时显示左右手的实时骨架模型。
- 下列命令不录制数据；不要添加 `--record-path`。

### 9.2.1 实现文件与数据流

主要代码：

```text
SenseGlove ROS2 -> UDP bridge:
  scripts/senseglove_ros_to_esrobo_hand_bridge.py

ROS2/Conda 隔离启动器:
  scripts/run_senseglove_to_esrobo_hand_bridge.sh

IsaacLab 手部模式启动器:
  scripts/run_esrobo_senseglove_hands_teleop.sh

IsaacLab UDP 接收与 IMU/腕部组合:
  src/dual_arm_teleop/isaaclab_ext/devices/udp_bimanual_body_device.py

SenseGlove 标定结果:
  config/senseglove_esrobo_calibration.json
```

实时数据链：

```text
SenseGloveState.hand_position
  -> 20 点手部骨架
  -> dex_vector 手腕到五指指尖向量目标
  -> ESROBO URDF/Mimic FK 优化
  -> 每只手 10 个主动关节

SenseGloveState.imu_orientation
  -> Nova 2 异常 w 分量修复
  -> Unity 到 ROS 坐标修正
  -> 固定 Nova 2 手套轴转换 + 每次中立位标定
  -> 机器人左右腕局部旋转
  -> 腕部关节 5-7
```

### 9.2.2 固定 IMU 轴与两阶段强制标定

Nova 2 IMU 相对手套本体的安装方向是设备级固定关系，不应由每次操作者动作重新估计。当前实现使用固定的手套局部轴矩阵，每次启动只重新记录佩戴后的 IMU 中立位以及手指张合范围。实时 bridge 会覆盖旧标定文件，不存在跳过或复用旧零位的选项。`--calibration-file` 只指定本次结果的保存位置，旧的 `--recalibrate` 参数已经删除。

标定顺序：

| 阶段 | 姿态/动作 | 操作要求 |
|---|---|---|
| 1/2 | 中立位张手 | 双腕静止、掌心朝向身体中间、手指完全伸直；同时记录 IMU 零位、张手特征和张手骨架 |
| 2/2 | 握拳 | 双腕保持相同中立姿态，双手完全握拳 |

每个阶段持续 10 秒。按下 `Enter` 后，0-6 秒用于准备姿态，6-10 秒为有效采集区间；当前 `--calibration-method last` 使用有效区间最后一个完整样本。两个阶段都应保持相同的腕部中立姿态，仅改变手指张合状态。第一阶段同时完成原来的 IMU 中立位和张手参考采集，因此不再重复保持同一个张手姿势。

IMU 零位与固定轴转换在手套中立局部坐标系中计算：

```text
q_local_delta = inverse(q_neutral) * q_current

R_fixed = fixed_nova2_source_to_semantic
rotvec_semantic = R_fixed * rotvec(q_local_delta)
rotvec_robot_local = diag(robot_axis_signs) * rotvec_semantic
```

三个目标分量顺序固定为：

```text
X = 左右摆动
Y = 上下摆动
Z = 掌心翻转
```

根据当前实机验证，动作语义到机器人腕部局部轴的最终符号为：

```text
left  = [+1, -1, +1]
right = [-1, -1, +1]
```

也就是：左手侧偏保持正向、上下摆动反向、掌心翻转保持正向；右手反转侧偏和上下摆动、掌心翻转保持正向。右手设备轴保持单位映射；左手实测设备轴为 `Y=左右摆动`、`Z=上下摆动`、`X=掌心翻转`，先转换到统一的语义 X/Y/Z。左右手都对完整的中立位相对旋转执行一次固定轴映射，左手不再单独拆分和重组 swing/twist；历史真实数据表明旧拆分路径在约 `82 deg` 输入下可额外产生最高约 `26.2 deg` 的姿态差。固定矩阵不会因为操作者每次动作幅度、耦合运动或保持姿态不同而改变。任一手套数据或 IMU 数据不完整时，标定会拒绝继续并退出。

实时 UDP 包设置 `hand_orientation_axis_calibrated=true`。IsaacLab 收到后直接使用固定设备轴与本次中立位共同得到的机器人局部旋转，跳过旧版接收端轴顺序和符号。旧录制或旧 UDP 包没有这个标志时，才使用 `ESROBO_HAND_IMU_LEFT_AXIS_ORDER`、`ESROBO_HAND_IMU_*_LOCAL_AXIS_SIGNS` 和 `ESROBO_HAND_IMU_RIGHT_SWAP_XY` 兼容路径。

### 9.2.3 硬件准备

1. 连接 SenseGlove 通信设备，打开左右 Nova 2。
2. 正确佩戴手套，标定后不要让手套相对手腕滑动。
3. 当前配置为左手序列号 `00885`、右手序列号 `00892`。
4. SenseCom 启动后必须保持运行；可以最小化，但不能关闭。
5. 如有旧进程，在原终端按 `Ctrl+C` 停止，然后检查：

```bash
pgrep -af 'senseglove.launch.py|SenseCom.x86_64|ros2_control_node'
```

### 9.2.4 终端 1：启动 SenseCom 和 ROS2 驱动

SenseGlove ROS2 节点必须使用系统 Python，不要使用 IsaacLab conda 环境：

```bash
cd ~/PhdResearch/DualArmTeleopration

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source external/senseglove_ros_ws/install/setup.bash

ros2 launch senseglove_bringup senseglove.launch.py \
  run_rviz:=false \
  run_sensecom:=true \
  run_finger_distance:=false
```

等待 SenseCom 显示左右手套均已连接。如果启动终端要求确认，按下 `Enter`。后续遥操作期间保持这个终端和 SenseCom 运行。

### 9.2.5 终端 2：标定前检查真实数据

```bash
cd ~/PhdResearch/DualArmTeleopration

conda deactivate 2>/dev/null || true
unset PYTHONPATH
unset LD_LIBRARY_PATH

source /opt/ros/humble/setup.bash
source external/senseglove_ros_ws/install/setup.bash

ros2 topic list | grep senseglove_states
```

必须看到：

```text
/senseglove/glove00885/lh/senseglove_states
/senseglove/glove00892/rh/senseglove_states
```

分别检查左右数据频率：

```bash
timeout 8s ros2 topic hz /senseglove/glove00885/lh/senseglove_states
timeout 8s ros2 topic hz /senseglove/glove00892/rh/senseglove_states
```

再确认每个话题只有一个发布者：

```bash
ros2 topic info -v /senseglove/glove00885/lh/senseglove_states
ros2 topic info -v /senseglove/glove00892/rh/senseglove_states
```

两侧都必须显示 `Publisher count: 1`。如果显示 `2` 或更多，说明重复启动了
SenseGlove ROS2 驱动；应在旧驱动终端按 `Ctrl+C`，只保留一套
`senseglove.launch.py`、左右 `ros2_control_node` 和 SenseCom。重复发布会造成测得频率翻倍、
新旧帧交错以及实时遥操作行为不稳定。

频率存在并不代表数据在更新。依次运行以下命令，同时运动对应手指或手腕，确认数值持续变化；每项检查后按 `Ctrl+C`：

```bash
ros2 topic echo /senseglove/glove00885/lh/senseglove_states --field position
ros2 topic echo /senseglove/glove00892/rh/senseglove_states --field position
ros2 topic echo /senseglove/glove00885/lh/senseglove_states --field imu_orientation
ros2 topic echo /senseglove/glove00892/rh/senseglove_states --field imu_orientation
```

如果出现 `The message type 'senseglove_msgs/msg/SenseGloveState' is invalid`，说明当前终端没有正确 source 工作区，重新执行本节开头的环境清理与两个 `source` 命令。

### 9.2.6 终端 3：启动 IsaacLab 手指 + 腕部仿真

启动器会自动激活 `env_isaaclab`，不需要提前手动激活：

```bash
cd ~/PhdResearch/DualArmTeleopration
./scripts/run_esrobo_senseglove_hands_teleop.sh
```

该模式锁定双臂关节 1-4，只允许腕部姿态与 20 个主动手关节运动。等待机器人和左右手骨架加载完成，然后可检查 UDP 监听：

```bash
ss -lunp | grep ':15050'
```

修改过 IMU 接收、轴标定或手腕组合代码后，必须完全关闭并重新启动本终端中的 IsaacLab，运行中的旧进程不会自动加载新代码。

### 9.2.7 终端 4：启动实时 bridge 并完成标定

```bash
cd ~/PhdResearch/DualArmTeleopration

ESROBO_HAND_SWAP_LEFT_RIGHT_TARGETS=0 \
./scripts/run_senseglove_to_esrobo_hand_bridge.sh \
  --left-serial 00885 \
  --right-serial 00892 \
  --retargeting-mode dex_vector \
  --dex-vector-set fingertip \
  --dex-curl-weight 0.015 \
  --dex-iterations 1 \
  --dex-damping 0.025 \
  --dex-max-joint-step 0.16 \
  --dex-normal-delta 0.003 \
  --dex-direction-weight 0.35 \
  --imu-w-repair auto \
  --calibration-duration 10 \
  --calibration-sample-start 6 \
  --calibration-method last \
  --rate-hz 60 \
  --smoothing-alpha 1.0 \
  --print-interval 1
```

不要添加已经删除的 `--recalibrate`。每次出现 `Press ENTER` 时，按照终端当前显示的两阶段姿态准备动作，再按 `Enter` 开始计时。第一阶段的中立位张手会同时完成 IMU 零位和张手参考采集，不需要做三个方向的测试动作。

当前硬件侧别固定为左手 `00885/lh`、右手 `00892/rh`，默认映射必须是
`robot_left<=left_glove`、`robot_right<=right_glove`。不要在正常实时遥操作命令中加入
`--swap-left-right-targets`。只有录制文件或设备标签被确认写反时，才临时设置
`ESROBO_HAND_SWAP_LEFT_RIGHT_TARGETS=1`；该开关会同时交换手指目标、手部骨架、IMU
中立位和 IMU 姿态，不能用来修正单个 IMU 轴方向。

### 9.2.8 成功标准与停止顺序

标定成功时，bridge 会打印类似：

```text
[senseglove_bridge] Robot hand mapping: left<=left_glove right<=right_glove.
[senseglove_bridge] Fixed Nova 2 source-to-semantic IMU axes: left=[[0, 1, 0], [0, 0, 1], [1, 0, 0]] right=[[1, 0, 0], [0, 1, 0], [0, 0, 1]].
[senseglove_bridge] Fixed IMU target-axis signs: left=[1, -1, 1] right=[-1, -1, 1] for [lateral, vertical, palm_roll].
[senseglove_bridge] Saved calibration to config/senseglove_esrobo_calibration.json.
[senseglove_bridge] sent=... hz=... imu=on skeleton=sgcore_5x4 ...
```

验收时分别只做一个动作：

1. 只向拇指侧偏转，机器人应只表现为左右摆动。
2. 只向手背侧抬起，机器人应只表现为上下摆动。
3. 只将掌心向前翻转，机器人应只表现为掌心翻转。
4. 左右手分别测试，确认没有轴交换或方向反转。
5. 张手、握拳和逐指弯曲时，机器人手指与骨架模型应同步变化。

停止时先在终端 4 按 `Ctrl+C` 停止 bridge，再关闭终端 3 的 IsaacLab，最后在终端 1 按 `Ctrl+C` 停止 ROS2/SenseCom。

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
- SenseGlove topic type is invalid: leave conda, clear `PYTHONPATH`/`LD_LIBRARY_PATH`, then source `/opt/ros/humble/setup.bash` and `external/senseglove_ros_ws/install/setup.bash` again.
- SenseGlove topics report a frequency but the values do not change: SenseCom or the hardware node may be publishing stale cached state; verify both `position` and `imu_orientation` while physically moving each glove.
- Bridge stays at `Waiting for SenseGlove topics`: check that serials `00885`/`00892` match `gloves.yaml`, both SenseCom devices are connected, and both topic values are fresh.
- IMU 中立位之后每次运行方向不同：确认没有使用旧版多动作 bridge；停止并重新启动 bridge 与 IsaacLab，再完成当前两个标定阶段。
- IMU 单轴方向固定但仍与机器人相反：这是固定设备轴到机器人腕部轴的符号问题，应修改代码中的设备级矩阵或 `IMU_ROBOT_LOCAL_AXIS_SIGNS`，不要通过每次改变标定动作补偿。
