from typing import Union
import torch
from torch import Tensor
from mmdet.registry import TASK_UTILS
from mmdet.structures import SampleList
from mmdet.models.layers import CdnQueryGenerator
from mmengine.structures import InstanceData


@TASK_UTILS.register_module()
class CdnQueryGenerator(CdnQueryGenerator):
    pass


@TASK_UTILS.register_module()
class AdnQueryGenerator(CdnQueryGenerator):
    """Crater-DETR: a novel transformer network for crater detection
    based on dense supervision and multiscale fusion <https://ieeexplore.ieee.org/document/10466735/>`_

    Code is rewrited by BugBubbles: holy221aba@gmail.com`_.
    """

    def __call__(
        self,
        batch_data_samples: SampleList,
        aux_head_output: InstanceData,
    ) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth.

        Descriptions of the Number Values in code and comments:
            - num_target_total: the total target number of the input batch
              samples.
            - max_num_target: the max target number of the input batch samples.
            - num_noisy_targets: the total targets number after adding noise,
              i.e., num_target_total * num_groups * 2.
            - num_denoising_queries: the length of the output batched queries,
              i.e., max_num_target * num_groups * 2.

        NOTE The format of input bboxes in batch_data_samples is unnormalized
        (x, y, x, y), and the output bbox queries are embedded by normalized
        (cx, cy, w, h) format bboxes going through inverse_sigmoid.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): List of the batch
                data samples, each includes `gt_instance` which has attributes
                `bboxes` and `labels`. The `bboxes` has unnormalized coordinate
                format (x, y, x, y).

        Returns:
            tuple: The outputs of the dn query generator.

            - dn_label_query (Tensor): The output content queries for denoising
              part, has shape (bs, num_denoising_queries, dim), where
              `num_denoising_queries = max_num_target * num_groups * 2`.
            - dn_bbox_query (Tensor): The output reference bboxes as positions
              of queries for denoising part, which are embedded by normalized
              (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
              shape (bs, num_denoising_queries, 4) with the last dimension
              arranged as (cx, cy, w, h).
            - attn_mask (Tensor): The attention mask to prevent information
              leakage from different denoising groups and matching parts,
              will be used as `self_attn_mask` of the `decoder`, has shape
              (num_queries_total, num_queries_total), where `num_queries_total`
              is the sum of `num_denoising_queries` and `num_matching_queries`.
            - dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
        """
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        aux_labels_list = []
        aux_bboxes_list = []
        for sample, aux_pred in zip(batch_data_samples, aux_head_output):
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            aux_bboxes = aux_pred.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
            aux_bboxes_list.append(aux_bboxes)
            aux_labels_list.append(aux_pred.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)
        length = min(map(len, aux_labels_list))
        # if the auxiliary head outputs the same number of bboxes and labels
        # for each sample, we can stack them directly so we should keep the
        # same length for each sample
        aux_bboxes = torch.stack([bbox[:length] for bbox in aux_bboxes_list])
        aux_scores = torch.stack([label[:length] for label in aux_labels_list])

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)

        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        batch_idx = torch.cat(
            [torch.full_like(t.long(), i) for i, t in enumerate(gt_labels_list)]
        )
        dn_label_query, dn_bbox_query = self.collate_dn_queries(
            dn_label_query,
            dn_bbox_query,
            batch_idx,
            len(batch_data_samples),
            num_groups,
        )
        # adn part
        # the auxiliary head outputs DEEM to be noised labels and bboxes
        # noted that auxiliary outputs should detach from gradient to
        # prevent the transformer decoder gradient backpropagating to
        # the auxiliary head
        adn_label_query = self.label_embedding(aux_scores)

        dn_label_query = torch.cat([dn_label_query, adn_label_query], dim=1)
        dn_bbox_query = torch.cat(
            [dn_bbox_query, aux_bboxes.detach().requires_grad_()], dim=1
        )

        # attention mask is the combination of cdn and adn attention mask
        attn_mask = self.generate_dn_mask(
            max_num_target,
            num_groups,
            aux_scores.shape[1],
            device=dn_label_query.device,
        )

        dn_meta = dict(
            num_denoising_queries=dn_label_query.shape[1],
            num_denoising_groups=num_groups,
        )

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def generate_dn_mask(
        self,
        max_num_target: int,
        num_groups: int,
        num_adn_queries: int,
        device: Union[torch.device, str],
    ) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 0 0 0 0 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
          1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________|
          |_____________|       |
        num_adn_queries + num_cdn_queries = num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.

        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_cdn_queries = int(max_num_target * 2 * num_groups)
        num_denoising_queries = num_cdn_queries + num_adn_queries
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total, num_queries_total, device=device, dtype=torch.bool
        )
        # Make the matching part cannot see the denoising groups
        # including cdn and adn queries
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other

        ## for adn part
        attn_mask[num_adn_queries:num_denoising_queries, :num_adn_queries] = True

        ## for cdn part
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(
                num_adn_queries + max_num_target * 2 * i,
                num_adn_queries + max_num_target * 2 * (i + 1),
            )
            left_scope = slice(
                num_adn_queries, num_adn_queries + max_num_target * 2 * i
            )
            right_scope = slice(
                num_adn_queries + max_num_target * 2 * (i + 1),
                num_adn_queries + num_denoising_queries,
            )
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True

        return attn_mask
