import torch
from typing import Union, Tuple, Optional, List
from torch import Tensor
from torch.nn.modules.utils import _pair
from mmdet.registry import TASK_UTILS
from mmdet.utils import ConfigType
from mmdet.models.task_modules import GridAssigner, AssignResult, AnchorGenerator
from mmengine.structures import InstanceData


@TASK_UTILS.register_module()
class AACLAssigner(GridAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, `0`, or a positive integer
    indicating the ground truth index.

    - -1: don't care
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple[float, float]): IoU threshold for negative
        bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            Defaults to 0.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        iou_calculator (:obj:`ConfigDict` or dict): Config of overlaps
            Calculator.
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: Union[float, Tuple[float, float]],
        min_pos_iou: float = 0.0,
        gt_max_assign_all: bool = True,
        iou_calculator: ConfigType = dict(type="BboxOverlaps2D"),
    ) -> None:
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(
        self,
        pred_instances: InstanceData,
        gt_instances: InstanceData,
        gt_instances_ignore: Optional[InstanceData] = None,
        **kwargs
    ) -> AssignResult:
        """Assign gt to bboxes. The process is very much like the max iou
        assigner, except that positive samples are constrained within the cell
        that the gt boxes fell in.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, 0, or a positive number. -1 means don't care,
        0 means negative sample, positive number is the index (1-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to -1
        2. assign proposals whose iou with all gts <= neg_iou_thr to 0
        3. for each bbox within a cell, if the iou with its nearest gt >
            pos_iou_thr and the center of that gt falls inside the cell,
            assign it to that bbox
        4. for each gt bbox, assign its nearest proposals within the cell the
            gt bbox falls in to itself.

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels

        priors = pred_instances.priors
        responsible_flags = pred_instances.responsible_flags

        num_gts, num_priors = gt_bboxes.size(0), priors.size(0)

        # compute iou between all gt and priors
        overlaps = self.iou_calculator(gt_bboxes, priors)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_priors,), -1, dtype=torch.long)

        if num_gts == 0 or num_priors == 0:
            # No ground truth or priors, return empty assignment
            max_overlaps = overlaps.new_zeros((num_priors,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            assigned_labels = overlaps.new_full((num_priors,), -1, dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels
            )

        # 2. assign negative: below
        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        # shape of max_overlaps == argmax_overlaps == num_priors
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        if isinstance(self.neg_iou_thr, float):
            assigned_gt_inds[
                (max_overlaps >= 0) & (max_overlaps <= self.neg_iou_thr)
            ] = 0
        elif isinstance(self.neg_iou_thr, (tuple, list)):
            assert len(self.neg_iou_thr) == 2
            assigned_gt_inds[
                (max_overlaps > self.neg_iou_thr[0])
                & (max_overlaps <= self.neg_iou_thr[1])
            ] = 0

        # 3. assign positive: falls into responsible cell and above
        # positive IOU threshold, the order matters.
        # the prior condition of comparison is to filter out all
        # unrelated anchors, i.e. not responsible_flags
        overlaps[:, ~responsible_flags.type(torch.bool)] = -1.0

        # calculate max_overlaps again, but this time we only consider IOUs
        # for anchors responsible for prediction
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # for each gt, which anchor best overlaps with it
        # for each gt, the max iou of all proposals
        # shape of gt_max_overlaps == gt_argmax_overlaps == num_gts
        gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=1)

        pos_inds = (max_overlaps > self.pos_iou_thr) & responsible_flags.type(
            torch.bool
        )
        assigned_gt_inds[pos_inds] = argmax_overlaps[pos_inds] + 1

        # 4. assign positive to max overlapped anchors within responsible cell
        for i in range(num_gts):
            if gt_max_overlaps[i] > self.min_pos_iou:
                if self.gt_max_assign_all:
                    max_iou_inds = (
                        overlaps[i, :] == gt_max_overlaps[i]
                    ) & responsible_flags.type(torch.bool)
                    assigned_gt_inds[max_iou_inds] = i + 1
                elif responsible_flags[gt_argmax_overlaps[i]]:
                    assigned_gt_inds[gt_argmax_overlaps[i]] = i + 1

        # assign labels of positive anchors
        assigned_labels = assigned_gt_inds.new_full((num_priors,), -1)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze()
        if pos_inds.numel() > 0:
            assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels
        )


@TASK_UTILS.register_module()
class AACLAnchorGenerator(AnchorGenerator):
    """Adaptive Anchor generator for HRFPNet.

    More details can be found in the `paper
    <https://ieeexplore.ieee.org/document/9521676>`_ .

    Code is rewrited by BugBubbles: holy221aba@gmail.com

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels.
        base_sizes (list[list[tuple[int, int]]]): The basic sizes
            of anchors in multiple levels.
    """

    def __init__(
        self,
        strides: Union[List[int], List[Tuple[int, int]]],
        base_sizes: List[List[Tuple[int, int]]],
        use_box_type: bool = False,
    ) -> None:
        self.strides = [_pair(stride) for stride in strides]
        self.centers = [(stride[0] / 2.0, stride[1] / 2.0) for stride in self.strides]
        self.base_sizes = []
        num_anchor_per_level = len(base_sizes[0])
        for base_sizes_per_level in base_sizes:
            assert num_anchor_per_level == len(base_sizes_per_level)
            self.base_sizes.append(
                [_pair(base_size) for base_size in base_sizes_per_level]
            )
        self.base_anchors = self.gen_base_anchors()
        self.use_box_type = use_box_type

    @property
    def num_levels(self) -> int:
        """int: number of feature levels that the generator will be applied"""
        return len(self.base_sizes)

    def gen_base_anchors(self) -> List[Tensor]:
        """Generate base anchors.

        Returns:
            list(torch.Tensor): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        for i, base_sizes_per_level in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(base_sizes_per_level, center)
            )
        return multi_level_base_anchors

    def gen_single_level_base_anchors(
        self,
        base_sizes_per_level: List[Tuple[int]],
        center: Optional[Tuple[float]] = None,
    ) -> Tensor:
        """Generate base anchors of a single level.

        Args:
            base_sizes_per_level (list[tuple[int]]): Basic sizes of
                anchors.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        x_center, y_center = center
        base_anchors = []
        for base_size in base_sizes_per_level:
            w, h = base_size

            # use float anchor and the anchor's center is aligned with the
            # pixel center
            base_anchor = torch.Tensor(
                [
                    x_center - 0.5 * w,
                    y_center - 0.5 * h,
                    x_center + 0.5 * w,
                    y_center + 0.5 * h,
                ]
            )
            base_anchors.append(base_anchor)
        base_anchors = torch.stack(base_anchors, dim=0)

        return base_anchors
