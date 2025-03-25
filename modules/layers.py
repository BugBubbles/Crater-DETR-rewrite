from typing import Tuple, Union, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mmdet.registry import MODELS
from mmengine.model import BaseModel
from mmdet.utils import ConfigType, MultiConfig, reduce_mean
from math import sqrt
from mmdet.models.losses import weighted_loss, QualityFocalLoss
from mmdet.models.dense_heads import DINOHead, YOLOV3Head, DeformableDETRHead
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_overlaps
from mmdet.structures import SampleList
from mmdet.utils import InstanceList, OptInstanceList


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
    def loss(
        self,
        hidden_states: Tensor,
        references: List[Tensor],
        enc_outputs_class: Tensor,
        enc_outputs_coord: Tensor,
        batch_data_samples: SampleList,
        dn_meta: Dict[str, int],
        aux_head_output: InstanceList,
    ) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (
            enc_outputs_class,
            enc_outputs_coord,
            batch_gt_instances,
            batch_img_metas,
            aux_head_output,
            dn_meta,
        )
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        enc_cls_scores: Tensor,
        enc_bbox_preds: Tensor,
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        aux_head_output: InstanceList,
        dn_meta: Dict[str, int],
        batch_gt_instances_ignore: OptInstanceList = None,
    ) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels), where
                `num_queries_total` is the sum of `num_denoising_queries`
                and `num_matching_queries`.
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            enc_cls_scores (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_bbox_preds (Tensor): The proposal generate from the encode
                feature map, has shape (bs, num_feat_points, 4) with the last
                dimension arranged as (cx, cy, w, h).
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of
                gt_instance. It usually includes ``bboxes`` and ``labels``
                attributes.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            dn_meta (Dict[str, int]): The dictionary saves information about
                group collation, including 'num_denoising_queries' and
                'num_denoising_groups'. It will be used for split outputs of
                denoising and matching parts and loss calculation.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # extract denoising and matching part of outputs
        (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_cdn_cls_scores,
            all_layers_cdn_bbox_preds,
            all_layers_adn_cls_scores,
            all_layers_adn_bbox_preds,
        ) = self.split_outputs(all_layers_cls_scores, all_layers_bbox_preds, **dn_meta)

        loss_dict = super(DeformableDETRHead, self).loss_by_feat(
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
        )
        # NOTE DETRHead.loss_by_feat but not DeformableDETRHead.loss_by_feat
        # is called, because the encoder loss calculations are different
        # between DINO and DeformableDETR.

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            # NOTE The enc_loss calculation of the DINO is
            # different from that of Deformable DETR.
            enc_loss_cls, enc_losses_bbox, enc_losses_iou = self.loss_by_feat_single(
                enc_cls_scores,
                enc_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox
            loss_dict["enc_loss_iou"] = enc_losses_iou

        if all_layers_cdn_cls_scores is not None:
            # calculate denoising loss from all decoder layers
            dn_losses_cls, dn_losses_bbox, dn_losses_iou = self.loss_dn(
                all_layers_cdn_cls_scores,
                all_layers_cdn_bbox_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dict(
                    num_denoising_groups=dn_meta["num_denoising_groups"],
                    num_denoising_queries=dn_meta["num_cdn_queries"],
                ),
            )
            # collate denoising loss
            loss_dict["cdn_loss_cls"] = dn_losses_cls[-1]
            loss_dict["cdn_loss_bbox"] = dn_losses_bbox[-1]
            loss_dict["cdn_loss_iou"] = dn_losses_iou[-1]
            for num_dec_layer, (loss_cls_i, loss_bbox_i, loss_iou_i) in enumerate(
                zip(dn_losses_cls[:-1], dn_losses_bbox[:-1], dn_losses_iou[:-1])
            ):
                loss_dict[f"d{num_dec_layer}.cdn_loss_cls"] = loss_cls_i
                loss_dict[f"d{num_dec_layer}.cdn_loss_bbox"] = loss_bbox_i
                loss_dict[f"d{num_dec_layer}.cdn_loss_iou"] = loss_iou_i
        if all_layers_adn_cls_scores is not None:
            if all_layers_adn_cls_scores.numel() > 0:
                # calculate matching loss from all decoder layers
                adn_loss_dict = super(DeformableDETRHead, self).loss_by_feat(
                    all_layers_adn_cls_scores,
                    all_layers_adn_bbox_preds,
                    aux_head_output,
                    batch_img_metas,
                    batch_gt_instances_ignore,
                )
                # collate matching loss
                loss_dict["adn_loss_cls"] = adn_loss_dict["loss_cls"]
                loss_dict["adn_loss_bbox"] = adn_loss_dict["loss_bbox"]
                loss_dict["adn_loss_iou"] = adn_loss_dict["loss_iou"]
            else:
                loss_dict["adn_loss_cls"] = loss_dict["loss_cls"].new_zeros(1)
                loss_dict["adn_loss_bbox"] = loss_dict["loss_cls"].new_zeros(1)
                loss_dict["adn_loss_iou"] = loss_dict["loss_cls"].new_zeros(1)
        return loss_dict

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

    @staticmethod
    def split_outputs(
        all_layers_cls_scores: Tensor,
        all_layers_bbox_preds: Tensor,
        num_cdn_queries: int = None,
        num_adn_queries: int = None,
        **kwargs,
    ) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if num_cdn_queries is not None and num_adn_queries is not None:
            num_denoising_queries = num_cdn_queries + num_adn_queries
            all_layers_cdn_cls_scores = all_layers_cls_scores[
                :, :, num_adn_queries:num_denoising_queries, :
            ]
            all_layers_cdn_bbox_preds = all_layers_bbox_preds[
                :, :, num_adn_queries:num_denoising_queries, :
            ]
            all_layers_adn_cls_scores = all_layers_cls_scores[:, :, :num_adn_queries, :]
            all_layers_adn_bbox_preds = all_layers_bbox_preds[:, :, :num_adn_queries, :]
            all_layers_matching_cls_scores = all_layers_cls_scores[
                :, :, num_denoising_queries:, :
            ]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[
                :, :, num_denoising_queries:, :
            ]
        elif num_cdn_queries is not None:
            all_layers_cdn_cls_scores = all_layers_cls_scores[:, :, num_cdn_queries:, :]
            all_layers_cdn_bbox_preds = all_layers_bbox_preds[:, :, num_cdn_queries:, :]
            all_layers_adn_cls_scores = None
            all_layers_adn_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores[
                :, :, num_cdn_queries:, :
            ]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[
                :, :, num_cdn_queries:, :
            ]
        elif num_adn_queries is not None:
            all_layers_cdn_cls_scores = None
            all_layers_cdn_bbox_preds = None
            all_layers_adn_cls_scores = all_layers_cls_scores[:, :, :num_adn_queries, :]
            all_layers_adn_bbox_preds = all_layers_bbox_preds[:, :, :num_adn_queries, :]
            all_layers_matching_cls_scores = all_layers_cls_scores[
                :, :, num_adn_queries:, :
            ]
            all_layers_matching_bbox_preds = all_layers_bbox_preds[
                :, :, num_adn_queries:, :
            ]
        else:
            all_layers_cdn_cls_scores = None
            all_layers_cdn_bbox_preds = None
            all_layers_adn_cls_scores = None
            all_layers_adn_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (
            all_layers_matching_cls_scores,
            all_layers_matching_bbox_preds,
            all_layers_cdn_cls_scores,
            all_layers_cdn_bbox_preds,
            all_layers_adn_cls_scores,
            all_layers_adn_bbox_preds,
        )


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
