from typing import Tuple, Union, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.registry import MODELS
from mmengine.model import BaseModel
from mmdet.utils import ConfigType, MultiConfig, InstanceList, reduce_mean
from math import sqrt
from mmdet.models.losses import weighted_loss, QualityFocalLoss
from mmdet.models.dense_heads import DINOHead, YOLOV3Head
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
import warnings
from mmdet.models.utils import multi_apply
from mmdet.structures import SampleList


class CRAU(nn.Module):
    def __init__(self, in_channels, ksize=(3, 3), stride=2, padding=1):
        super().__init__()
        self.w, self.h = ksize
        self.padding = padding
        self.stride = stride
        # Project the channel dimension, so we can ues 1x1 conv
        self.qhead = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Unfold(ksize, stride=stride, padding=padding),
        )
        self.vhead = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.Unfold((1, 1))
        )
        self.khead = nn.Unfold((1, 1))

    def forward(self, feat: Tensor, src: Tensor) -> Tensor:
        b, c, *size = src.shape
        v = self.vhead(feat)
        k = self.khead(feat)
        q = self.qhead(src).view(b, c, self.w * self.h, -1)
        A = (
            torch.einsum("bckd,bcd->bck", q, k).div(sqrt(q.shape[-1])).softmax(-1)
        )  # k = ksize * ksize, d = h * w
        weights = torch.einsum("bck,bcd->bckd", A, v).flatten(1, 2)
        return F.fold(
            weights, size, (self.w, self.h), stride=self.stride, padding=self.padding
        ).mul(src)


class CRAP(nn.Module):
    def __init__(self, in_channels, ksize=(3, 3), stride=2, padding=1):
        super().__init__()
        self.w, self.h = ksize
        self.padding = padding
        self.qhead = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.Unfold((1, 1))
        )
        self.khead = nn.Sequential(
            nn.Unfold(ksize, stride=stride, padding=1),
        )
        self.vhead = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.Unfold(ksize, stride=stride, padding=1),
        )

    def forward(self, feat: Tensor, src: Tensor) -> Tensor:
        b, c, *size = feat.shape
        q = self.qhead(feat)
        k = self.khead(src).view(b, c, self.w * self.h, -1)
        v = self.vhead(src).view(b, c, self.w * self.h, -1)
        A = (
            torch.einsum("bcd,bckd->bck", q, k).div(sqrt(q.shape[-1])).softmax(-1)
        )  # k = ksize * ksize, d = h * w
        weights = torch.einsum("bck,bckd->bckd", A, v).flatten(1, 2)
        # stride 取1 而不取self.stride的原因是，注意力权重的形状已经与feat一致了，只需要将其每个像素
        # 按1x1 卷积的方式整合为原始形状即可
        return F.fold(
            weights, size, (self.w, self.h), stride=1, padding=self.padding
        ).mul(feat)


@MODELS.register_module()
class CraterDETRFPN(BaseModel):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        ksize: int = 3,
        stride: int = 2,
        start_level: int = 0,
        end_level: int = -1,
        downsample_cfg: ConfigType = dict(mode="nearest"),
        init_cfg: MultiConfig = dict(
            type="Xavier", layer="Conv2d", distribution="uniform"
        ),
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.fp16_enabled = False
        self.downsample_cfg = downsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
        self.start_level = start_level
        self.end_level = end_level

        self.craus = nn.ModuleList()
        self.craps = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            crau = CRAU(in_channels[i], (ksize, ksize), stride)
            crap = CRAP(out_channels, (ksize, ksize), stride)

            self.craus.append(crau)
            self.craps.append(crap)

    def forward(
        self, inputs: Tuple[Tensor], memory: Tensor, spatial_shapes: Tensor
    ) -> tuple:
        """Forward function.

        Args:
            inputs (tuple[Tensor]): Features from the upstream network, each
                is a 4D-tensor.

        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [*inputs[self.start_level : self.backbone_end_level - 1], memory]
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            laterals[i - 1] = self.craus[i](laterals[i], laterals[i - 1])

        # build outputs
        memory_list = [laterals[0]] + [
            self.craps[i](
                F.interpolate(
                    laterals[i],
                    size=(*spatial_shapes[self.start_level + i + 1],),
                    **self.downsample_cfg,
                ),
                laterals[i],
            )
            for i in range(used_backbone_levels - 1)
        ]
        return memory_list


@MODELS.register_module()
class SOSIoULoss(nn.Module):
    r"""Crater-DETR: A Novel Transformer Network for  Crater
      Detection Based on Dense Supervision and  Multiscale Fusion
      https://ieeexplore.ieee.org/document/10466735/`_.

    Code is rewrite by BugBubbles holy221aba@gmail.com .

    Args:
        eps (float): Epsilon to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
    """

    def __init__(
        self, eps: float = 1e-6, reduction: str = "mean", loss_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        weight: Optional[Tensor] = None,
        avg_factor: Optional[int] = None,
        reduction_override: Optional[str] = None,
        cls_scores: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
                shape (n, 4).
            target (Tensor): The learning target of the prediction,
                shape (n, 4).
            weight (Optional[Tensor], optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (Optional[int], optional): Average factor that is used
                to average the loss. Defaults to None.
            reduction_override (Optional[str], optional): The reduction method
                used to override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".

        Returns:
            Tensor: Loss tensor.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, "none", "mean", "sum")
        reduction = reduction_override if reduction_override else self.reduction
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * sosiou_loss(
            pred,
            target,
            weight,
            reduction=reduction,
            avg_factor=avg_factor,
            cls_scores=cls_scores,
            labels=labels,
            **kwargs,
        )
        return loss


@weighted_loss
def sosiou_loss(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-7,
    cls_scores: Tensor = None,
    labels: Tensor = None,
) -> Tensor:
    r"""Implementation of paper `Extended-IoU Loss: A Systematic
    IoU-Related Method: Beyond Simplified Regression for Better
    Localization <https://ieeexplore.ieee.org/abstract/document/9429909>`_

    Code is modified from https://github.com//ShiqiYu/libfacedetection.train.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        smooth_point (float): hyperparameter, default is 0.1.
        eps (float): Epsilon to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    if cls_scores is not None or labels is not None:
        assert (
            cls_scores is not None and labels is not None
        ), "cls_scores and labels should be provided together"
    px1, py1, px2, py2 = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    tx1, ty1, tx2, ty2 = target[:, 0], target[:, 1], target[:, 2], target[:, 3]

    # compute the area of intersection
    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * (
        torch.min(py2, ty2) - torch.max(py1, ty1)
    ).clamp(0)
    # compute euclidean distance
    t_w = tx2 - tx1
    t_h = ty2 - ty1
    t_c = (pred[:, :2] + pred[:, 2:]) / 2
    p_w = px2 - px1
    p_h = py2 - py1
    p_c = (pred[:, :2] + pred[:, 2:]) / 2
    enclose_w_pow = (torch.max(px2, tx2) - torch.min(px1, tx1)).pow(2)
    enclose_h_pow = (torch.max(py2, ty2) - torch.min(py1, ty1)).pow(2)
    # compute the area of union
    union = p_w * p_h + t_w * t_h - inter
    # compute the IoU
    ious = inter / (union + eps)

    # choose the class score cooresponding to the target class
    loss = (
        1
        - ious.mul(1 - cls_scores.sigmoid().take(labels.sub(1).unsqueeze(1)).squeeze(1))
        if cls_scores is not None
        else ious
        + (t_c - p_c).pow(2).sum(1) / (enclose_w_pow + enclose_h_pow + eps)
        + (t_w - p_w).pow(2) / (enclose_w_pow + eps)
        + (t_h - p_h).pow(2) / (enclose_h_pow + eps)
    )
    return loss


@MODELS.register_module()
class SOSDINOHead(DINOHead):
    def loss_by_feat_single(
        self,
        cls_scores: Tensor,
        bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
    ) -> Tuple[Tensor]:
        """Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images, has shape (bs, num_queries, cls_out_channels).
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape (bs, num_queries, 4).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tuple[Tensor]: A tuple including `loss_cls`, `loss_box` and
            `loss_iou`.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, batch_gt_instances, batch_img_metas
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        if isinstance(self.loss_cls, QualityFocalLoss):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero().squeeze(1)
            scores = label_weights.new_zeros(labels.shape)
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_decode_bbox_targets = bbox_cxcywh_to_xyxy(pos_bbox_targets)
            pos_bbox_pred = bbox_preds.reshape(-1, 4)[pos_inds]
            pos_decode_bbox_pred = bbox_cxcywh_to_xyxy(pos_bbox_pred)
            scores[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            )
            loss_cls = self.loss_cls(
                cls_scores, (labels, scores), label_weights, avg_factor=cls_avg_factor
            )
        else:
            loss_cls = self.loss_cls(
                cls_scores, labels, label_weights, avg_factor=cls_avg_factor
            )

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # construct factors used for rescale bboxes
        factors = []
        for img_meta, bbox_pred in zip(batch_img_metas, bbox_preds):
            (
                img_h,
                img_w,
            ) = img_meta["img_shape"]
            factor = (
                bbox_pred.new_tensor([img_w, img_h, img_w, img_h])
                .unsqueeze(0)
                .repeat(bbox_pred.size(0), 1)
            )
            factors.append(factor)
        factors = torch.cat(factors, 0)

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        bbox_preds = bbox_preds.reshape(-1, 4)
        bboxes = bbox_cxcywh_to_xyxy(bbox_preds) * factors
        bboxes_gt = bbox_cxcywh_to_xyxy(bbox_targets) * factors

        # regression IoU loss, defaultly GIoU loss
        if isinstance(self.loss_iou, SOSIoULoss):
            loss_iou = self.loss_iou(
                bboxes,
                bboxes_gt,
                bbox_weights,
                avg_factor=num_total_pos,
                cls_scores=cls_scores,
                labels=labels,
            )
        else:
            loss_iou = self.loss_iou(
                bboxes, bboxes_gt, bbox_weights, avg_factor=num_total_pos
            )

        # regression L1 loss
        loss_bbox = self.loss_bbox(
            bbox_preds, bbox_targets, bbox_weights, avg_factor=num_total_pos
        )
        return loss_cls, loss_bbox, loss_iou


@MODELS.register_module()
class AACLAYOLOV3Head(YOLOV3Head):

    def forward_aux(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        rescale: bool = False,
        cfg: Optional[Dict] = None,
    ) -> InstanceList:
        """Perform forward propagation of the detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[obj:`InstanceData`]: Detection results of each image
            after the post process.
        """
        batch_img_metas = [data_samples.metainfo for data_samples in batch_data_samples]

        outs = self(x)

        predictions = self.predict_by_feat(
            *outs, cfg=cfg, batch_img_metas=batch_img_metas, rescale=rescale
        )
        return predictions
