# ESROBO Teleoperation Reproducibility Snapshot

本目录记录 2026-09-14 对本机工作副本和 GitHub 远程执行 `git fetch --prune` 后得到的可复现基线。外部仓库均保持在实际安装使用的 commit，不因上游 `main` 更新而自动升级。

## 1. 版本锁定

| 项目 | 仓库、版本或证据 | 精确状态 |
|---|---|---|
| Dual-arm-teleoperation | <https://github.com/SaluteYan/Dual-arm-teleoperation> | 本复现包的源代码基线为 `ed5bba42a7a03e855a548dbe09fe2a1ca25f8878`；最终发布 commit 以包含本文件的 Git commit 为准 |
| XRoboToolkit-PC-Service-Pybind | <https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind> | `main`, `c64ccf6acd577a333e03b66fafe8efeeceb511b1`, package `1.0.2`; fetch 后与 `origin/main` 一致 |
| XRoboToolkit-Teleop-Sample-Python | <https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python> | `main`, `79e5cb8a56e3455515ce1b476e993c764ec58739`, package `1.0.3`; fetch 后与 `origin/main` 一致 |
| Isaac Sim | NVIDIA pip distribution | `isaacsim==5.1.0.0`, `isaacsim-core==5.1.0.0` |
| Isaac Lab | <https://github.com/isaac-sim/IsaacLab> | installed/local `main` commit `858234d06e6845d75420d2558e236526282e5da6`; Python package `isaaclab==0.54.4`; upstream `origin/main` was one commit ahead at audit time |
| IsaacTeleop | <https://github.com/NVIDIA/IsaacTeleop> | detached tag `v1.3.131`, commit `7002ed63d69454ae4f15c0ee19f803fd2846592b`; package `isaacteleop==1.3.131`; intentionally pinned instead of current `main` |
| SenseGlove ROS | <https://github.com/Adjuvo/senseglove_ros> | `humble-dev`, `a14a4683cf60b7296c863ec69c6e3f8e5c998e99`; fetch 后与 `origin/humble-dev` 一致 |
| dex-retargeting | <https://github.com/dexsuite/dex-retargeting> | tag `v0.5.0`, commit `3f56141bc8bd2760d5e452e382937269554ebb21` |
| PICO XRoboToolkit app | local APK `XRoboToolkit-PICO-1.1.1.apk` | version `1.1.1` from the vendor artifact name; APK SHA256 is recorded below. No headset was attached during this audit, so installed-device `versionName` was not re-queried |
| XRoboToolkit PC Service | `/opt/apps/roboticsservice` | executable embeds version `1.3.1`; executable and gRPC library SHA256 are recorded below |
| SenseCom | vendor installation in SenseGlove ROS workspace | `1.9.2` (`versionInfo.txt` and `SenseCom_Data/app.info`) |

The SenseGlove checkout contains required local source/config adaptations on top of the clean upstream commit. Reapply [`patches/senseglove-ros-local-adaptation.patch`](patches/senseglove-ros-local-adaptation.patch) after cloning. The patch sets Nova 2 serials `00885/00892`, handles the 20th joint and malformed frames, and fixes RViz topics. SenseCom binaries are not included in this repository; install vendor SenseCom 1.9.2 separately.

## 2. Environment Files

- `conda/env_isaaclab.yml`: full `conda env export` for simulation.
- `conda/env_xrobotoolkit.yml`: full `conda env export` for the PICO bridge.
- `conda/*-explicit-linux-64.txt`: exact platform lock URLs for byte-level conda package reproduction.
- `pip/env_isaaclab-pip-freeze.txt`: Isaac Sim/Lab Python environment.
- `pip/env_xrobotoolkit-pip-freeze.txt`: XRoboToolkit bridge environment.
- `pip/system-ros-humble-pip-freeze.txt`: system Python used by ROS 2 Humble/SenseGlove.

The generated YAML files omit the machine-specific `prefix:` line. One pip entry for `xrobotoolkit_sdk` retains its original local editable path as evidence; reproduce it by cloning the pinned Pybind commit under `external/` and running the project installer.

Audited host: Ubuntu 22.04.5 LTS, kernel `6.8.0-138-generic`, NVIDIA driver `580.173.02`, RTX 5070 Ti 16 GB, conda `25.11.0`.

## 3. Assets And Hashes

The complete `assets/esrobo/` directory is versioned. The 487 MiB base USD is stored through Git LFS; run `git lfs install && git lfs pull` after cloning.

- `manifests/assets-esrobo-usd-sha256.txt`: every USD file.
- `manifests/xrobotoolkit-pico-apk-sha256.txt`: vendor PICO APK identity.
- `manifests/xrobotoolkit-pc-service-sha256.txt`: installed service binary identity.
- `manifests/sample-recordings-sha256.txt`: curated recording identity.

Verify assets from the repository root:

```bash
sha256sum -c reproducibility/manifests/assets-esrobo-usd-sha256.txt
git lfs ls-files
```

## 4. Successful Recordings

`samples/pico-body-success-20260811.jsonl` is a continuous five-second excerpt (`t_rel_s=3.0..8.0`, 449 packets plus header) from the successful 30.43-second PICO recording `pico_body_20260811_214430.jsonl`. It contains valid full-body joint positions/orientations, including both shoulders, elbows and wrists.

`samples/senseglove-hands-success-20260818.jsonl` is a continuous five-second excerpt (`t_rel_s=0.0..5.0`, 301 packets plus header) from the successful 30-second, nominal 60 Hz Nova 2 recording `senseglove_real_20260818_222013.jsonl`. It contains both hands, finger data, IMU orientation and the two-pose calibration metadata.

Only bounded excerpts are committed. The 221 MiB local raw recording collection remains ignored to avoid turning Git into a dataset store.

## 5. Verified Replay

The archived successful run used the current UDP action path and received 20 hand targets, both calibrated IMU deltas and full-body frames. Evidence is in:

- `logs/arm-hand-replay-sender-success.log`: 5,362 events sent over 29.923 seconds.
- `logs/isaaclab-arm-hand-replay-success.log`: Isaac Sim startup, robot asset load, UDP receiver, hand mimic setup, teleoperation start and packet receipt.

Run the checked-in samples in two terminals. Terminal 1:

```bash
cd ~/PhdResearch/DualArmTeleopration
unset LD_LIBRARY_PATH PYTHONPATH
ESROBO_BODY_UDP_HOST=0.0.0.0 \
ESROBO_BODY_UDP_PORT=15050 \
ESROBO_BODY_RETARGETING_MODE=arm_vector \
ESROBO_BODY_ARM_VECTOR_POSITION_MODE=segment_direction_absolute \
ESROBO_BODY_REQUIRE_CALIBRATION=0 \
ESROBO_BODY_SWAP_LEFT_RIGHT_TARGETS=0 \
ESROBO_BODY_POSITION_DELTA_SIGNS="1 1 1" \
ESROBO_BODY_ENABLE_VISUALIZATION=1 \
ESROBO_USE_HAND_IMU_ORIENTATION=1 \
ESROBO_HAND_IMU_RELATIVE_TO_ARM_FRAME=1 \
ESROBO_HAND_SKELETON_VISUALIZATION=1 \
ESROBO_DEBUG_XR_DATA=0 \
ESROBO_DEBUG_PERF=0 \
./scripts/run_esrobo_bodytracking_teleop.sh
```

Wait until IsaacLab prints `Listening for body-tracking UDP packets`, then run Terminal 2:

```bash
cd ~/PhdResearch/DualArmTeleopration
unset LD_LIBRARY_PATH PYTHONPATH
ESROBO_BODY_UDP_HOST=127.0.0.1 \
ESROBO_BODY_UDP_PORT=15050 \
./scripts/run_esrobo_arm_hand_recording_replay.sh \
  --body-record-path reproducibility/samples/pico-body-success-20260811.jsonl \
  --hand-record-path reproducibility/samples/senseglove-hands-success-20260818.jsonl \
  --hand-imu-mode relative \
  --print-interval 0
```

Success criteria are: robot and combined arm/hand skeletons move; the IsaacLab terminal reports 20 hand targets, left/right IMU deltas, and full-body skeleton activation; no second process writes a conflicting stream to UDP port `15050`.

## 6. Live Hardware Boundary

The sample recordings and archived log prove prior successful acquisition/replay. This 2026-09-14 audit did not reconnect the PICO or Nova 2 gloves, so it does not claim fresh live-hardware or headset-installed-version validation. Before real-robot deployment, repeat the topic-rate, single-publisher, calibration and emergency-stop checks in `IMPLEMENTATION_STEPS.md`.
