"""Project-local OpenXR device wrapper for ESROBO teleoperation."""

from __future__ import annotations

from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg
from isaaclab.utils import configclass


class ESROBOOpenXRDevice(OpenXRDevice):
    """OpenXR device with project-local retargeter reset hooks."""

    def reset(self) -> None:
        super().reset()
        self.reset_retargeters()

    def reset_retargeters(self) -> None:
        for retargeter in getattr(self, "_retargeters", []):
            if hasattr(retargeter, "reset"):
                retargeter.reset()

    def set_debug_print(self, enabled: bool = True, interval: int = 30) -> None:
        """Compatibility hook for the project teleop script."""
        if enabled:
            print(
                "[ESROBO OpenXR] Raw OpenXR debug printing uses the official device implementation; "
                "project retargeter/action debug output remains available."
            )


@configclass
class ESROBOOpenXRDeviceCfg(OpenXRDeviceCfg):
    """Configuration for the project-local OpenXR device wrapper."""

    class_type: type[DeviceBase] = ESROBOOpenXRDevice
