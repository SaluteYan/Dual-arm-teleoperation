"""Per-arm pose stability using accumulated motion, independent of quaternion sign."""

from __future__ import annotations

import torch


class PoseTargetStability:
    """Compare against a fixed anchor so slow deliberate motion releases a hold."""

    def __init__(
        self,
        frame_indices: list[list[int]],
        orientation_frames: list[int],
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        hold_frames: int,
    ):
        self.frame_indices = frame_indices
        self.orientation_frames = set(orientation_frames)
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.hold_frames = hold_frames
        self.anchor: torch.Tensor | None = None
        self.count: torch.Tensor | None = None

    def reset(self) -> None:
        self.anchor = None
        self.count = None

    def update(self, targets: torch.Tensor) -> torch.Tensor:
        if self.anchor is None:
            self.anchor = targets.detach().clone()
            self.count = torch.zeros(
                (targets.shape[0], len(self.frame_indices)),
                dtype=torch.long,
                device=targets.device,
            )
        for side_index, indices in enumerate(self.frame_indices):
            if not indices:
                continue
            position_change = torch.linalg.vector_norm(
                targets[:, indices, :3] - self.anchor[:, indices, :3], dim=-1
            ).amax(dim=1)
            stable = position_change <= self.position_tolerance_m
            orientation_indices = [i for i in indices if i in self.orientation_frames]
            if orientation_indices:
                current = torch.nn.functional.normalize(
                    targets[:, orientation_indices, 3:], dim=-1
                )
                reference = torch.nn.functional.normalize(
                    self.anchor[:, orientation_indices, 3:], dim=-1
                )
                dots = torch.sum(current * reference, dim=-1).abs().clamp(0.0, 1.0)
                angle = 2.0 * torch.acos(dots)
                stable &= angle.amax(dim=1) <= self.orientation_tolerance_rad
            self.count[:, side_index] = torch.where(
                stable,
                self.count[:, side_index] + 1,
                0,
            )
            # Re-anchor only after leaving the tolerance, never every noisy frame.
            changed_envs = torch.nonzero(~stable, as_tuple=False).flatten()
            for frame_index in indices:
                self.anchor[changed_envs, frame_index] = targets[
                    changed_envs, frame_index
                ].detach()
        return self.count >= self.hold_frames
