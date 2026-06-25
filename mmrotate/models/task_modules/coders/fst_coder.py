# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch
from torch import Tensor

from mmrotate.registry import TASK_UTILS


@TASK_UTILS.register_module()
class FSTCoder:
    """Fourier Series Transformation coder for periodic angles.

    Args:
        fourier_size (int): Number of harmonic frequency pairs.
        omega (float): Base angular frequency. For le90, ``omega=2`` maps
            the pi-periodic box angle to a 2*pi phase.
        with_dc (bool): Whether ``encode`` prepends a zero DC component.
    """

    def __init__(self,
                 fourier_size: int = 2,
                 omega: float = 2.0,
                 with_dc: bool = True) -> None:
        self.fourier_size = fourier_size
        self.omega = omega
        self.with_dc = with_dc
        self.encode_size = 2 * fourier_size + int(with_dc)

    def encode(self, angle_targets: Tensor) -> Tensor:
        """Encode angles into Fourier components.

        Args:
            angle_targets (Tensor): Angle tensor with shape ``(..., 1)``.

        Returns:
            Tensor: Encoded components with shape
            ``(..., 2 * fourier_size + int(with_dc))``.
        """
        if angle_targets.size(-1) != 1:
            raise ValueError('FSTCoder expects the last dimension to be 1.')

        components = []
        if self.with_dc:
            components.append(torch.zeros_like(angle_targets))

        phase = self.omega * angle_targets
        for order in range(1, self.fourier_size + 1):
            components.append(torch.cos(order * phase))
            components.append(torch.sin(order * phase))
        return torch.cat(components, dim=-1)

    def decode(self, angle_preds: Tensor, keepdim: bool = False) -> Tensor:
        """Decode Fourier components back to angles."""
        if angle_preds.size(-1) == self.encode_size and self.with_dc:
            angle_preds = angle_preds[..., 1:]

        if angle_preds.size(-1) % 2 != 0:
            raise ValueError('Fourier angle components must be cos/sin pairs.')
        if angle_preds.size(-1) < 2:
            shape = angle_preds.shape[:-1] + ((1, ) if keepdim else ())
            return angle_preds.new_zeros(shape)

        cos_1 = angle_preds[..., 0]
        sin_1 = angle_preds[..., 1]
        phase_1 = torch.atan2(sin_1, cos_1)
        final_phase = phase_1

        if angle_preds.size(-1) >= 4:
            cos_2 = angle_preds[..., 2]
            sin_2 = angle_preds[..., 3]
            phase_2 = torch.atan2(sin_2, cos_2) / 2.0
            phase_2_alt = phase_2 + torch.where(
                phase_2 <= 0,
                angle_preds.new_tensor(math.pi),
                angle_preds.new_tensor(-math.pi))
            score_1 = torch.cos(phase_2 - phase_1)
            score_2 = torch.cos(phase_2_alt - phase_1)
            final_phase = torch.where(score_1 > score_2, phase_2,
                                      phase_2_alt)

        angle = final_phase / self.omega
        return angle.unsqueeze(-1) if keepdim else angle
