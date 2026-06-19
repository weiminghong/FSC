# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from torch import Tensor

from mmrotate.models.dense_heads.angle_branch_retina_head import \
    AngleBranchRetinaHead
from mmrotate.registry import MODELS


@MODELS.register_module()
class FourierSeriesRetinaHead(AngleBranchRetinaHead):
    """RetinaNet head with Fourier Series Coder angle supervision."""

    def __init__(self,
                 *args,
                 phase_mod: bool = True,
                 phase_mod_weight: float = 1.0,
                 loss_fourier: dict = dict(
                     type='mmdet.L1Loss', loss_weight=0.2),
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=[
                         dict(
                             type='Normal',
                             name='retina_cls',
                             std=0.01,
                             bias_prob=0.01),
                         dict(
                             type='Normal',
                             name='retina_angle_cls',
                             std=0.01,
                             bias_prob=0.01),
                     ]),
                 **kwargs) -> None:
        self.phase_mod = phase_mod
        self.phase_mod_weight = phase_mod_weight
        super().__init__(
            *args,
            loss_angle=loss_fourier,
            init_cfg=init_cfg,
            **kwargs)

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                            angle_pred: Tensor, anchors: Tensor,
                            labels: Tensor, label_weights: Tensor,
                            bbox_targets: Tensor, bbox_weights: Tensor,
                            angle_targets: Tensor, angle_weights: Tensor,
                            avg_factor: int) -> Tuple[Tensor, Tensor, Tensor]:
        loss_cls, loss_bbox, loss_fourier = super().loss_by_feat_single(
            cls_score, bbox_pred, angle_pred, anchors, labels, label_weights,
            bbox_targets, bbox_weights, angle_targets, angle_weights,
            avg_factor)

        if not self.phase_mod:
            return loss_cls, loss_bbox, loss_fourier

        angle_pred = angle_pred.float().permute(0, 2, 3, 1).reshape(
            -1, self.encode_size)
        angle_weights = angle_weights.reshape(-1, 1)
        trig_pred = angle_pred[:, 1:] if self.encode_size % 2 == 1 else angle_pred
        if trig_pred.size(1) == 0 or trig_pred.size(1) % 2 != 0:
            return loss_cls, loss_bbox, loss_fourier

        num_freqs = trig_pred.size(1) // 2
        trig_pairs = trig_pred.reshape(-1, 2)
        squared_norm = trig_pairs.pow(2).sum(dim=-1, keepdim=True)
        target_norm = torch.ones_like(squared_norm)
        norm_weights = angle_weights.repeat_interleave(num_freqs, dim=0)
        loss_norm = self.loss_angle(
            squared_norm,
            target_norm,
            weight=norm_weights,
            avg_factor=max(avg_factor * num_freqs, 1.0))
        loss_fourier = loss_fourier + self.phase_mod_weight * loss_norm
        return loss_cls, loss_bbox, loss_fourier

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            angle_preds: List[Tensor],
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        losses = super().loss_by_feat(
            cls_scores,
            bbox_preds,
            angle_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        losses['loss_fourier'] = losses.pop('loss_angle')
        return losses
