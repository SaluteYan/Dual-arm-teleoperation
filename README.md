# DualArmTeleopration (ESROBO 双臂遥操作)

PICO/XRoboToolkit 全身骨骼追踪 + SenseGlove Nova 2 灵巧手数据，遥操作 **ESROBO 双臂灵巧手机器人** 的代码仓库。

本仓库面向 **实际双臂机器人部署**（实机遥操作），不再依赖 IsaacLab 仿真遥操作。仓库只保留遥操作核心代码与机器人 URDF/网格模型，**不包含**仿真专用 USD 资产、录制数据与第三方外部依赖（依赖按下方说明另行获取/安装）。

## 仓库结构

```text
src/dual_arm_teleop/isaaclab_ext/  遥操作核心库（UDP 设备、手臂重定向、IK 控制器、动作、机器人资产、任务配置）
scripts/                            桥接进程与启动脚本（PICO/XRoboToolkit、SenseGlove ROS2、录制/回放、安装器）
urdf/esrobo_waist_with_head/        ESROBO 机器人 URDF 与网格模型（实机模型，含 base_link.STL 等网格）
assets/esrobo/                      ESROBO 资产配置（config.yaml、主 USD 入口；configuration/*.usd 仿真资产不纳入本仓库）
config/senseglove_esrobo_calibration.json  SenseGlove 手部标定文件
pyproject.toml                      Python 包元数据
```

## 实机部署所需依赖（外部，不在本仓库内）

实机遥操作需要以下外部依赖。请按你的目标机器人平台获取对应版本：

| 依赖 | 用途 | 获取方式 |
|------|------|----------|
| **ROS 2** | SenseGlove 数据发布 | 建议 Humble（或 Jazzy，视 `senseglove_ros` 版本） |
| **senseglove_ros_ws** | SenseGlove Nova 2 官方 ROS2 硬件驱动/驱动节点 | 官方 SenseGlove ROS 工作区（含 `senseglove_ros`、`senseglove_api`、`senseglove_com`），放置于 `external/senseglove_ros_ws/` |
| **XRoboToolkit-PC-Service-Pybind** | PICO 全身骨骼/运动追踪数据读取 | 放置于 `external/XRoboToolkit-PC-Service-Pybind/` |
| **XRoboToolkit-Teleop-Sample-Python** | PICO/XRoboToolkit 遥操作示例/数据链路 | 放置于 `external/XRoboToolkit-Teleop-Sample-Python/` |
| **PICO 头显 + XRoboToolkit 应用** | 全身骨骼/手臂/手部追踪硬件 | PICO 设备 + XRoboToolkit PC Service |

> 说明：`IsaacLab` 与 `IsaacTeleop` 为 **仿真专用** 依赖，实机部署无需安装。

## 环境与 Python 版本

- Python `>=3.10`
- 建议为桥接与驱动分别创建独立 conda 环境（如 `env_xrobotoolkit`、`env_isaaclab`），见 `scripts/*.sh` 启动脚本。
- `pyproject.toml` 定义 `dual-arm-teleop` 包，`src` 目录使用 setuptools 打包。

## 实机遥操作流程（概览）

每个桥接/驱动脚本均为独立的 UDP 或 ROS2 进程，通过 UDP 端口（默认 `15050`）或 ROS2 topic 与 ESROBO 双臂控制端通信。核心入口与脚本：

- `scripts/run_xrobotoolkit_pc_service.sh` —— 启动 XRoboToolkit PC Service，供 PICO 连接。
- `scripts/xrobotoolkit_body_udp_bridge.py` —— 读取 PICO 全身骨骼（肩/肘/腕），打包为 UDP 发送。
- `scripts/senseglove_ros_to_esrobo_hand_bridge.py` —— 读取 SenseGlove ROS2 手指关节与 IMU 手部姿态，默认使用 `dex_vector` 模式按 URDF/Mimic FK 做手指向量重定向，标定后打包为 UDP。Nova 2 IMU 相对手套的轴定义采用固定设备矩阵，每次实机 bridge 启动强制执行“中立位双手张开 -> 双手握拳”两阶段标定；第一阶段同时记录 IMU 中立位、张手特征和张手骨架，避免重复采集相同姿势。人的动作差异不会改变 IMU 坐标轴。不存在跳过标定或复用旧零位的选项。`--calibration-file` 只指定本次标定结果的保存位置，已有文件会被覆盖。
- SenseGlove 专用 bridge 默认保持设备侧别，即 `robot_left <= left_glove`、`robot_right <= right_glove`。手指目标、手部骨架、IMU 中立位和固定设备轴转换会使用同一个源手套映射；如需兼容左右标签相反的数据，可显式设置 `ESROBO_HAND_SWAP_LEFT_RIGHT_TARGETS=1`。该设置不会继承 PICO 双臂的 `ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS`。
- 实时硬件约定为左手 `00885/lh`、右手 `00892/rh`。正常启动不要传入 `--swap-left-right-targets`；bridge 启动后应打印 `left<=left_glove right<=right_glove`。同时用 `ros2 topic info -v` 确认左右状态话题均为 `Publisher count: 1`，避免重复驱动造成频率翻倍和帧交错。
- `src/dual_arm_teleop/isaaclab_ext/devices/udp_bimanual_body_device.py` —— UDP 数据接收、手臂重定向（`arm_vector`）、手部姿态合成与动作生成的核心设备。
- `src/dual_arm_teleop/isaaclab_ext/retargeters/esrobo_upper_body_retargeter.py` —— 人体手臂方向到机器人手臂的重定向。

### 当前重定向改进

- 双臂 IK 除肘部/手腕位姿任务外，同时约束 `shoulder -> elbow` 和 `elbow -> wrist` 两个段向量。该约束直接优化大臂、小臂姿态，并使用 Huber 权重降低偶发骨架点异常的影响。默认参数为 `ESROBO_IK_UPPER_ARM_VECTOR_COST=36`、`ESROBO_IK_FOREARM_VECTOR_COST=48`、`ESROBO_IK_ARM_VECTOR_HUBER_DELTA=0.04`；设 `ESROBO_IK_ARM_VECTOR_TASKS=0` 可退回旧 IK。
- 小角度段方向处理改为连续自适应混合，不再在角度阈值内完全锁住，因此快动作不会因死区累计后跳变。
- SenseGlove UDP 包同时携带当前绝对 IMU 四元数、每次启动标定的中性四元数、固定设备轴转换后的局部旋转和单调采样时间。实时接收端优先使用固定轴转换结果，并将 PICO 前臂位置插值到 IMU 采样时刻后组合手腕姿态；前臂接近伸直时沿用上一帧弯曲平面，避免坐标系翻转。旧手部录制没有轴元数据时，会自动走原有接收端固定映射兼容路径。
- Nova 2 的异常 `w=z` 四元数修复会在两个可能的 `w` 符号中选择与上一帧连续的解。
- `dex_vector` 默认采用官方 dex-retargeting teleop 配置风格，只优化手腕到五个指尖的向量，避免 ESROBO 欠驱动手的 Mimic 关节被逐节指骨目标过度约束；同时用张手/握拳标定计算每指闭合进度，以较低权重约束对应 MCP 主动关节覆盖相同运动比例。仍保留方向残差、Huber 鲁棒权重、上一帧正则与软关节限位，并默认每帧一次 warm-start 迭代。可用 `--dex-vector-set dense` 复现旧的逐节指骨目标，或用 `--dex-curl-weight 0` 关闭闭合进度辅助项。

纯数学回归和录制数据对比：

```bash
cd ~/PhdResearch/DualArmTeleopration
/usr/bin/python3 -m unittest tests/test_retargeting_math.py -v
/usr/bin/python3 scripts/compare_senseglove_retargeting_modes.py \
  recordings/<senseglove-recording>.jsonl \
  --stride 10 --max-frames 80
```
- `src/dual_arm_teleop/isaaclab_ext/controllers/pink_ik.py` —— Pink IK 逆运动学控制器。
- `src/dual_arm_teleop/isaaclab_ext/actions/pink_actions.py` —— 双臂 + 双手关节动作生成。

具体接线、端口、标定步骤（含 SenseGlove 两姿态标定、PICO 骨骼标定）请参考各脚本 `--help` 与代码内注释。

## 手部可视化挂载（骨架手 vs 机器人手）

本仓库默认将第一帧有效的机器人目标前臂姿态锁定为手腕随动零位。此时左右灵巧手保持机器人 USD 初始姿态，掌心朝向身体中间；之后手腕姿态按照目标前臂相对该零位的旋转随动，并在此基础上叠加 SenseGlove IMU 相对前臂旋转。显示用双手骨架直接读取同一画面中人体骨架的肩/肘/腕，锁定首帧前臂坐标系后逐帧应用 `R_current_forearm @ R_start_forearm.T`，因此双手会和显示的双臂严格随动；SenseGlove IMU 只叠加手相对前臂的旋转（`ESROBO_HAND_SKELETON_PALMS_FACE_EACH_OTHER=0`、`ESROBO_HAND_SKELETON_LEFT_LOCAL_ROTATION_DEG=0`）。如需复现旧版动态“掌心相对”显示，可设 `ESROBO_HAND_SKELETON_PALMS_FACE_EACH_OTHER=1`。

SenseGlove 左右手是具有相反手性的点云。显示层会额外反射左手局部掌面法向轴，再挂载到机器人左腕；该修正使左右骨架手互为镜像，同时不改变发送给 `dex_vector` 的手指关节重定向数据。

实时 Nova 2 数据使用固定的手套局部轴定义，不再根据每次手腕动作拟合轴基。每次启动只记录 `q_neutral`，并在手套中立局部坐标中计算 `inverse(q_neutral) * q_current`。右手固定设备矩阵保持单位阵；根据左手实机逐轴验证，左手设备轴为 `Y=左右摆动`、`Z=上下摆动`、`X=掌心翻转`，因此固定矩阵使用 `[[0,1,0],[0,0,1],[1,0,0]]` 转换到统一语义顺序。左手局部轴符号为 `[+1, -1, +1]`，右手为 `[-1, -1, +1]`；左手侧偏符号与右手不同，以匹配镜像腕部的拇指/小指侧运动方向。左右手现在都直接对完整的中立位相对旋转执行一次固定轴映射；左手不再拆分和重组 swing/twist，从而避免大角度组合旋转因乘法顺序产生额外姿态偏移。`ESROBO_HAND_IMU_LEFT_AXIS_ORDER`、`ESROBO_HAND_IMU_*_LOCAL_AXIS_SIGNS` 和 `ESROBO_HAND_IMU_RIGHT_SWAP_XY` 仅供缺少固定轴元数据的旧录制或旧 UDP 包兼容使用。

联合回放默认使用 `--hand-imu-mode relative`，适用于双臂和手套分别录制的文件：手套 IMU 被解释为腕部相对动作，不会反向抵消当前双臂的前臂随动。只有身体与手套在同一次动作中同步采集时才使用 `--hand-imu-mode synchronized`；使用 `--hand-imu-mode disabled` 可仅检查前臂带动双手的效果。

## 大文件说明

`urdf/esrobo_waist_with_head/meshes/base_link.STL`（约 227 MB）超出 GitHub 100 MB 单文件限制，通过 **Git LFS** 管理。克隆后需执行：

```bash
git lfs install
git lfs pull
```

## 许可证

各源文件沿用其原始 SPDX 头（多数为 BSD-3-Clause，来自 Isaac Lab 工程）。第三方依赖遵循其各自许可证。
