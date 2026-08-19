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
- `scripts/senseglove_ros_to_esrobo_hand_bridge.py` —— 读取 SenseGlove ROS2 手指关节与 IMU 手部姿态，标定后打包为 UDP。
- `src/dual_arm_teleop/isaaclab_ext/devices/udp_bimanual_body_device.py` —— UDP 数据接收、手臂重定向（`arm_vector`）、手部姿态合成与动作生成的核心设备。
- `src/dual_arm_teleop/isaaclab_ext/retargeters/esrobo_upper_body_retargeter.py` —— 人体手臂方向到机器人手臂的重定向。
- `src/dual_arm_teleop/isaaclab_ext/controllers/pink_ik.py` —— Pink IK 逆运动学控制器。
- `src/dual_arm_teleop/isaaclab_ext/actions/pink_actions.py` —— 双臂 + 双手关节动作生成。

具体接线、端口、标定步骤（含 SenseGlove 两姿态标定、PICO 骨骼标定）请参考各脚本 `--help` 与代码内注释。

## 手部可视化挂载（骨架手 vs 机器人手）

本仓库默认将骨架模型双手以 **与机器人灵巧手相同的手臂相对姿态** 挂载（`ESROBO_HAND_SKELETON_PALMS_FACE_EACH_OTHER=0`、`ESROBO_HAND_SKELETON_LEFT_LOCAL_ROTATION_DEG=0`），使骨架手与机器人手在初始姿态及双臂运动过程中相对手臂的姿态保持一致。如需复现旧版镜像显示，可将 `ESROBO_HAND_SKELETON_LEFT_LOCAL_ROTATION_DEG` 设为 `180`。

## 大文件说明

`urdf/esrobo_waist_with_head/meshes/base_link.STL`（约 227 MB）超出 GitHub 100 MB 单文件限制，通过 **Git LFS** 管理。克隆后需执行：

```bash
git lfs install
git lfs pull
```

## 许可证

各源文件沿用其原始 SPDX 头（多数为 BSD-3-Clause，来自 Isaac Lab 工程）。第三方依赖遵循其各自许可证。
