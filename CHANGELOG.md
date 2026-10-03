# 更改记录

## 2026-10-02：双臂 + 双手遥操作的手腕随动与静止稳定性优化

- 记录整理日期：2026-10-03。
- 对应提交：[`ea73b12853869c3b7ebefb33c3041b91fce9e674`](https://github.com/SaluteYan/Dual-arm-teleoperation/commit/ea73b12853869c3b7ebefb33c3041b91fce9e674)，已于 2026-10-03 推送至 `origin/main`。
- 提交标题：`Fix forearm-relative wrist tracking and per-arm static hold`。
- 修改前基线：`df1b41b`（本次优化开始时的 HEAD）。
- 修改范围：13 个代码、测试与文档文件，新增 854 行、删除 135 行。

### 1. 问题与目标

使用录制数据回放双臂和双手遥操作时，观察到以下问题：

1. 灵巧手整体姿态与人体手部整体姿态变化不一致，手臂运动与手腕独立运动的组合关系不正确。
2. 显示的人体手部骨架有时出现异常姿态变化，显示结果与机器人手腕目标缺少统一计算路径。
3. 人体动作停止后，连接灵巧手的机械臂末端关节仍继续运动。

本次目标是保留人类手臂与手的两层运动关系：手的空间位置和世界姿态随手臂运动；同时保留手相对于前臂的独立弯腕、侧摆和翻掌。静止时抑制微小输入噪声和 IK 解漂移，真实的手臂或腕部运动仍能解除对应侧保持。

### 2. 手腕姿态合成与中立挂载

新增 [wrist_pose.py](src/dual_arm_teleop/wrist_pose.py)，提供不依赖 IsaacLab 的 NumPy 姿态数学函数，供设备代码复用。

统一手腕目标的旋转关系：

```text
R_hand_world = R_forearm_current @ R_forearm_rest.T @ R_hand_rest @ R_wrist_local
```

| 符号 | 含义 |
|---|---|
| `R_forearm_current` | 当前目标前臂坐标系在世界中的旋转 |
| `R_forearm_rest` | 机器人中立前臂坐标系在世界中的旋转 |
| `R_hand_rest` | 机器人 USD 初始手掌在世界中的旋转 |
| `R_wrist_local` | 独立腕部动作在手掌挂载局部坐标系中的旋转 |

其中 `R_forearm_rest.T @ R_hand_rest` 表示中立位下前臂到手掌的挂载关系。前臂随动先改变手掌的世界朝向，独立腕部旋转再在局部坐标系中叠加，避免错误的世界轴旋转顺序。

[udp_bimanual_body_device.py](src/dual_arm_teleop/isaaclab_ext/devices/udp_bimanual_body_device.py) 的主要调整：

- `_wrist_orientation_matrix()` 使用机器人中立前臂建立挂载，不再将任意第一帧人体前臂直接视为机器人中立位。因此第一帧手臂已倾斜时，灵巧手也会随该前臂方向倾斜。
- 机器人中立前臂接近伸直、无法可靠确定滚转方向时，将首次有效前臂法向投影到中立前臂，避免任意全局法向造成掌心翻转。
- `_relative_hand_rotation_local()` 将腕部变化表达在初始手掌挂载轴中，并缓存每侧最后接受的局部旋转。
- 同步采集的绝对 IMU 数据通过同步前臂变化扣除父坐标系运动，避免前臂运动被重复计入。
- 分别录制的身体与手套数据在 `relative` 模式下解释为独立腕部动作，不扣除另一份录制中的前臂运动，也不额外调用同步前臂估计覆盖当前前臂。

手腕位置继续由现有手臂重定向生成；本次重点修正姿态组合和后续 IK 跟随，而非引入另一套手腕位置算法。

### 3. 接近伸直时的前臂坐标系稳定

原先通过大臂和前臂方向的叉积直接确定弯曲平面。手臂接近伸直时，叉积很小，微小骨架点噪声可能被归一化为较大的掌心旋转。

新的 `forearm_rotation()`：

- 弯肘小于约 8° 时，将上一有效平面法向投影到当前前臂的垂直平面，保持滚转连续。
- 在约 8° 至 20° 之间平滑恢复观测到的弯曲平面，实际权重由叉积幅值的正弦区间计算，并采用平滑插值。
- 首帧接近伸直且没有历史法向时，使用确定性的备用轴，不将微小测量叉积当作可靠滚转参考。
- 将弯曲轴定义为 `cross(normal, forearm)`，构成右手正交坐标系，修正旧定义可能产生 `det=-1` 的问题。
- 原设备中的 `_forearm_points_to_rotation()` 委托该共享函数，统一目标和可视化使用的基础数学。

### 4. 腕部小噪声与手套断流

- 局部腕部姿态相对最后接受值的变化小于 **0.35°** 时，保持最后接受的旋转。
- 比较锚点不会在每个噪声帧更新，缓慢但持续的真实腕部运动可以累计超过阈值，避免将慢动作永久锁住。
- 相对前臂模式下，IMU 超时或暂时断流时保留最后的局部腕部旋转，前臂继续随动；不突然回到无腕部动作的中立姿态。
- 设备重置时清理该缓存，使新一轮启动重新建立参考。

### 5. 显示骨架手与机器人目标统一

默认 `robot_hand_local` 显示路径中的 `_attached_hand_skeleton_rotation()` 直接读取同一帧机器人手腕目标，再应用显示挂载关系；显示层不再重新解释 IMU 或锁定另一套前臂参考。

显示手骨架的位置仍挂载到显示的人体腕点。默认显示设置保持 `ESROBO_HAND_SKELETON_PALMS_FACE_EACH_OTHER=0` 和 `ESROBO_HAND_SKELETON_LEFT_LOCAL_ROTATION_DEG=0`。

这里的一致性是**显示手骨架与机器人手腕目标的一致性**。机器人在 PhysX 中的实际姿态仍受 IK 和动力学跟随影响，不能由显示一致性推导出实测误差为零。

### 6. 双臂独立静止判定与关节保持

新增 [pose_stability.py](src/dual_arm_teleop/pose_stability.py) 中的 `PoseTargetStability`，并修改 [pink_actions.py](src/dual_arm_teleop/isaaclab_ext/actions/pink_actions.py)：

1. 左右臂分别维护固定锚点、稳定帧计数和保持状态，单侧运动只解除该侧保持。
2. 位置与角度分别判断。位置变化使用距离，姿态变化使用四元数对应的旋转角，并对四元数 `q` / `-q` 的等价表示保持不变性。
3. 默认目标稳定容差为 **3 mm / 1°**，连续 **12 帧**。变化始终相对固定锚点累计，避免缓慢运动因单帧变化很小而被持续误判为静止。
4. 姿态权重为零的肘部任务不参与姿态稳定或到位判定，避免未受约束的肘部朝向阻止手腕保持。
5. 目标稳定后，位姿已到位或 IK 关节解连续收敛，都可触发保持。收敛判定使用每侧关节解最大变化，默认阈值为 `0.001 rad`，连续 12 帧。
6. 保持已求解的关节目标，不再反复用实测关节覆盖目标，使 PhysX 能继续跟踪同一个固定目标。
7. 清零处于保持状态的关节速度前馈，其他侧仍可正常求解和运动。

[pink_ik.py](src/dual_arm_teleop/isaaclab_ext/controllers/pink_ik.py) 新增 `hold_command_joints()`，将已保持关节的内部 warm-start 状态与固定目标同步，避免另一侧继续求解时带动已保持侧漂移。

手指关节继续独立接收手套目标，不纳入手臂静止判定。保持的目标是消除静止时继续迭代产生的漂移，而不是保证所有不可达姿势都达到零误差。

### 7. 默认参数变化

调整 [esrobo_pink_controller_cfg.py](src/dual_arm_teleop/isaaclab_ext/configs/esrobo_pink_controller_cfg.py) 和 [run_esrobo_bodytracking_teleop.sh](scripts/run_esrobo_bodytracking_teleop.sh)，保持启动脚本与控制器的腕部任务权重一致。

| 参数 | 修改前 | 修改后 | 用途 |
|---|---:|---:|---|
| `ESROBO_IK_POSITION_COST` | 32 | 64 | 腕部位置任务权重 |
| `ESROBO_IK_ORIENTATION_COST` | 0.05 | 8 | 腕部姿态任务权重 |
| `ESROBO_IK_STATIC_TARGET_TOLERANCE` | `1e-6`，旧逻辑混合比较位姿分量 | `0.003 m` | 固定锚点位置变化容差 |
| `ESROBO_IK_STATIC_TARGET_ANGULAR_TOLERANCE_RAD` | 无独立参数 | `0.0174532925 rad`，约 1° | 固定锚点角度变化容差 |
| `ESROBO_IK_STATIC_TARGET_HOLD_FRAMES` | 4 | 12 | 稳定帧数 |
| `ESROBO_IK_STATIC_JOINT_TOLERANCE_RAD` | 无 | `0.001 rad` | IK 解收敛变化阈值，动作层读取环境变量 |

既有到位阈值继续为位置 `0.01 m`、姿态 `0.05 rad`。启动脚本新增角度变化容差的导出；上述环境变量可覆盖默认值。0.35° IMU 死区和前臂约 8° / 20° 区间目前是代码内常量。

### 8. 历史手套录制的固定轴兼容

修改 [replay_esrobo_arm_hand_recordings.py](scripts/replay_esrobo_arm_hand_recordings.py)：

- 加载旧手部录制时，使用 `senseglove_raw` 内的修正 IMU 四元数和录制头的 `calibration.imu_neutral`，按当前桥接代码的固定轴映射重建手腕局部旋转。
- 保留录制的手指关节目标，并遵循包内左右手来源映射。
- 已带固定轴标记、缺少原始 IMU 或缺少有效中立参考的包沿用原包，保持兼容。
- 新增 `--preserve-recorded-hand-imu`，可保留旧录制映射进行比较。
- 对已有固定轴元数据的包保留原始绝对 IMU，不用局部旋转反推传感器世界姿态；发送时刷新采样单调时间。

模式选择：`relative` 适用于本次身体与手套分别录制的数据；`synchronized` 适用于同一次动作同步采集的数据；`disabled` 可关闭腕部 IMU，单独检查前臂随动。

### 9. 新增评估与回归测试

新增 [evaluate_esrobo_arm_hand_tracking.py](scripts/evaluate_esrobo_arm_hand_tracking.py)：

- 通过本机 UDP、实际设备解析、Pink IK 和 PhysX 运行无界面仿真，不仅检查数学函数。
- 按原始录制时间间隔使用仿真时钟调度，默认回放 30 秒，末尾追加 4 秒固定输入。
- `--tail-mode hand_dropout` 在末尾只继续发送身体包、停止手套包，用于检查 IMU 超时行为。
- `--source-root` 可选择另一份源码，比较修改前后的实现。
- 输出逐帧 `.json` 和对应 `.summary.json`，记录手腕目标、实测位姿、末端关节、关节命令、显示姿态差和保持状态。

测试文件：

| 文件 | 覆盖内容 |
|---|---|
| [test_wrist_pose.py](tests/test_wrist_pose.py)，新增 | 伸直时抗噪声、首帧法向、中立挂载、正交右手坐标系、局部旋转顺序、同步父运动只计一次、分别录制时不抵消前臂随动 |
| [test_pose_stability.py](tests/test_pose_stability.py)，新增 | 四元数符号等价、零权重肘部朝向、慢动作累计解除保持、静止噪声保持、独立腕部旋转释放及重置 |
| [test_hand_recording_replay.py](tests/test_hand_recording_replay.py)，修改 | 旧 IMU 固定轴重建、手指目标保留、固定轴包绝对 IMU 保留、已标定与缺少原始数据的兼容 |

[README.md](README.md) 同步更新了姿态关系、静止保持、历史录制兼容和无界面评估说明。以上文件连同前述设备、控制器、动作、配置和启动脚本，构成本次 13 个文件的提交。

### 10. 仿真验证结果

数据文件：`recordings/pico_body_20260811_214430.jsonl` 和 `recordings/senseglove_real_20260818_222013.jsonl`。两份数据分别录制，采用相对腕部解释。

修改前使用基线源码副本，修改后使用本次代码；采用相同的录制事件时间调度，回放约 30 秒后追加 4 秒固定输入。平均值与 P95 按评估脚本的整个采样区间统计，包含末尾固定输入阶段。以下误差为**机器人实测手腕与手腕目标之间的误差**，不等同于机器人与真实人手之间的直接测量误差。

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| 左手平均姿态误差 | 35.10° | 4.38° |
| 右手平均姿态误差 | 38.47° | 8.28° |
| 左手平均位置误差 | 17.57 mm | 15.13 mm |
| 右手平均位置误差 | 46.92 mm | 25.07 mm |
| 左手姿态误差 P95 | 69.34° | 26.20° |
| 右手姿态误差 P95 | 65.71° | 24.92° |
| 固定输入最后 3 秒，两侧末端共 6 个关节的最大变化 | 10.476° | 0.000° |
| 固定输入最后 3 秒，双臂 14 个关节命令的最大变化 | 0.302945 rad | 0.000000 rad |
| 左侧显示手骨架与手腕目标姿态差 P95 | 4.10° | 0.00° |
| 右侧显示手骨架与手腕目标姿态差 P95 | 4.45° | 0.00° |

修改后末帧左右臂保持状态为 `[[true, true]]`；末帧左右实测姿态误差仍为 **0.20° / 15.18°**。因此静止保持生效不代表残余跟随误差已消除。

手套断流测试的末尾 4 秒只继续发送身体数据：

- 超时前后左右手腕目标最大姿态变化约 **0.00718° / 0.00878°**，未出现突然回中。
- 最后 3 秒末端实测关节变化及双臂关节命令变化均为 **0**。
- 末帧左右臂保持状态为 `[[true, true]]`。

详细本地证据：[验证报告](recordings/wrist_tracking_eval_20261002/report.md)、`before.json`、`after.json`、`hand_dropout.json`、对应 `.summary.json` 和日志。`recordings/` 被 Git 忽略，这些链接依赖本机证据文件，不随代码提交上传。

### 11. 软件检查与验证边界

本次优化完成时已通过项目回归测试；2026-10-03 提交前再次运行，结果为 **28 passed**。同时通过修改文件的 Python 编译、启动脚本 `bash -n`、定向 Ruff 和暂存区 `git diff --check`。定向 Ruff 忽略现有启动导入相关的 `E402/F403/F405`，不代表整个仓库通过默认全部规则。

```bash
cd ~/PhdResearch/DualArmTeleopration
env -u PYTHONPATH -u LD_LIBRARY_PATH \
  ~/anaconda3/envs/env_isaaclab/bin/python -m pytest -q tests
```

验证覆盖代码数学、UDP 解析、显示姿态关系、实际 IK、PhysX 跟随、固定输入保持和手套断流。没有连接真实机器人、实时 PICO 或 SenseGlove 做本次验收；仿真时钟调度不测真实端到端延迟。仍存在动态跟随和不可完全满足约束的残余误差，本次没有证明所有姿势完全对齐，也没有证明原始传感器异常尖峰全部消除。

个人 `config/senseglove_esrobo_calibration.json` 改动继续保留在本地，未包含在上述提交中；录制、评估数据和日志也未上传。

### 12. 运行与复现

#### 有界面双臂 + 双手回放

先关闭上一轮仿真与回放进程，重新启动以加载修改后的代码。终端 1：

```bash
cd ~/PhdResearch/DualArmTeleopration
unset LD_LIBRARY_PATH PYTHONPATH
ESROBO_IK_POSITION_COST=64 \
ESROBO_IK_ORIENTATION_COST=8 \
ESROBO_USE_HAND_IMU_ORIENTATION=1 \
ESROBO_HAND_IMU_RELATIVE_TO_ARM_FRAME=1 \
ESROBO_BODY_ENABLE_VISUALIZATION=1 \
ESROBO_HAND_SKELETON_VISUALIZATION=1 \
./scripts/run_esrobo_bodytracking_teleop.sh
```

等待出现 `Teleoperation started` 后，在终端 2 发送录制数据：

```bash
cd ~/PhdResearch/DualArmTeleopration
unset LD_LIBRARY_PATH PYTHONPATH
/usr/bin/python3 -u scripts/replay_esrobo_arm_hand_recordings.py \
  --host 127.0.0.1 --port 15050 \
  --body-record-path recordings/pico_body_20260811_214430.jsonl \
  --hand-record-path recordings/senseglove_real_20260818_222013.jsonl \
  --hand-imu-mode relative --speed 1.0
```

启动脚本默认未开启手腕 IMU，因此上面显式设置 `ESROBO_USE_HAND_IMU_ORIENTATION=1`。在此回放中，身体运动应带动手的空间位置和世界姿态，手套数据仍可产生相对于前臂的腕部旋转；显示骨架手应与手腕目标共享朝向。

#### 无界面跟随与静止评估

本机需已有 `env_isaaclab`、IsaacLab 外部依赖、机器人资产及上述本地录制文件。

```bash
cd ~/PhdResearch/DualArmTeleopration
unset LD_LIBRARY_PATH PYTHONPATH
source ~/anaconda3/etc/profile.d/conda.sh
conda activate env_isaaclab
export TERM=xterm-256color
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib"
./external/IsaacLab/isaaclab.sh -p scripts/evaluate_esrobo_arm_hand_tracking.py \
  --headless --device cuda:0 --enable_pinocchio \
  --output recordings/wrist_tracking_eval/current.json
```

检查生成的 `current.summary.json`：重点比较姿态和位置误差、`tail_wrist_range_rad`、`tail_command_range_rad`、`visual_p95_deg` 和 `final_hold`。再次评估手套断流时，将输出路径改为另一文件并追加 `--tail-mode hand_dropout`。重新运行受配置和运行环境影响，结果应以新生成的证据为准。
