import torch
from torch import nn, Tensor
from typing import Dict, List, Tuple, Union, Optional
from mmdet.structures import OptSampleList, SampleList
from mmengine.structures import InstanceData
from mmdet.utils import OptConfigType, ConfigType
from mmdet.models.detectors import DeformableDETR
from mmdet.registry import MODELS, TASK_UTILS
from torch.nn import functional as F
from mmdet.models.layers import (
    SinePositionalEncoding,
    DeformableDetrTransformerEncoder,
    DinoTransformerDecoder,
    SinePositionalEncoding,
)
from mmcv.ops import MultiScaleDeformableAttention


@MODELS.register_module()
class CraterDETR(DeformableDETR):
    r"""Crater-DETR: a novel transformer network for crater detection
      based on dense supervision and multiscale fusion <https://ieeexplore.ieee.org/document/10466735/>`_

    Args:
        decoder (:obj:`ConfigDict` or dict, optional): Config of the
            Transformer decoder. Defaults to None.
        bbox_head (:obj:`ConfigDict` or dict, optional): Config for the
            bounding box head module. Defaults to None.
        with_box_refine (bool, optional): Whether to refine the references
            in the decoder. Defaults to `False`.
        as_two_stage (bool, optional): Whether to generate the proposal
            from the outputs of encoder. Defaults to `False`.
        num_feature_levels (int, optional): Number of feature levels.
            Defaults to 4.
    """

    def __init__(
        self,
        *args,
        aux_start_level: int = 0,
        aux_end_level: int = 3,
        aux_bbox_head: ConfigType,
        fpn: ConfigType,
        dn_cfg: OptConfigType = None,
        train_cfg: OptConfigType,
        test_cfg: OptConfigType,
        aux_pred_cfg: OptConfigType = None,
        **kwargs,
    ):
        """
        dn_cfg : Automatic denoising query generator config
        """
        if train_cfg:
            assert (
                "bbox_head" in train_cfg and "aux_bbox_head" in train_cfg
            ), "bbox_head and aux_bbox_head config must be explicitly set in train_cfg"
        super().__init__(
            *args, train_cfg=train_cfg["bbox_head"], test_cfg=test_cfg, **kwargs
        )
        aux_bbox_head.update(train_cfg=train_cfg["aux_bbox_head"], test_cfg=test_cfg)
        self.aux_bbox_head = MODELS.build(aux_bbox_head)
        self.aux_start_level = aux_start_level
        self.aux_end_level = aux_end_level
        self.fpn = MODELS.build(fpn)
        self.dn_query_generator = None
        if dn_cfg is not None:
            assert (
                "num_classes" not in dn_cfg
                and "num_queries" not in dn_cfg
                and "hidden_dim" not in dn_cfg
            ), (
                "The three keyword args `num_classes`, `embed_dims`, and "
                "`num_matching_queries` are set in `detector.__init__()`, "
                "users should NOT set them in `dn_cfg` config explicitly."
            )
            dn_cfg["num_classes"] = self.bbox_head.num_classes
            dn_cfg["embed_dims"] = self.embed_dims
            dn_cfg["num_matching_queries"] = self.num_queries
            self.dn_query_generator = TASK_UTILS.build(dn_cfg)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aux_pred_cfg = aux_pred_cfg

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(**self.positional_encoding)
        self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        self.decoder = DinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, (
            f"embed_dims should be exactly 2 times of num_feats. "
            f"Found {self.embed_dims} and {num_feats}."
        )

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims)
        )
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super(DeformableDETR, self).init_weights()
        for coder in self.encoder, self.decoder:
            for p in coder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        nn.init.xavier_uniform_(self.memory_trans_fc.weight)
        nn.init.xavier_uniform_(self.query_embedding.weight)
        nn.init.normal_(self.level_embed)

    def _forward(
        self, batch_inputs: Tensor, batch_data_samples: OptSampleList = None
    ) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[Tensor]: A tuple of features from ``bbox_head`` forward.
        """
        img_feats = self.extract_feat(batch_inputs)
        _, head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Forward process of Transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.
        The difference is that the ground truth in `batch_data_samples` is
        required for the `pre_decoder` to prepare the query of DINO.
        Additionally, DINO inherits the `pre_transformer` method and the
        `forward_encoder` method of DeformableDETR. More details about the
        two methods can be found in `mmdet/detector/deformable_detr.py`.

        Args:
            img_feats (tuple[Tensor]): Tuple of feature maps from neck. Each
                feature map has shape (bs, dim, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            dict: The dictionary of bbox_head function inputs, which always
            includes the `hidden_states` of the decoder output and may contain
            `references` including the initial and intermediate references.
        """
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(
            img_feats, batch_data_samples
        )

        pred_maps, aux_head_output = self.forward_encoder(
            **encoder_inputs_dict,
            img_feats=img_feats,
            batch_data_samples=batch_data_samples,
        )

        memory_flat_dict = self.memory_flatten(
            pred_maps, batch_data_samples=batch_data_samples
        )

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **memory_flat_dict, aux_head_output=aux_head_output
        )
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return pred_maps, head_inputs_dict

    def forward_encoder(
        self,
        feat,
        feat_mask,
        feat_pos,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        img_feats,
        batch_data_samples,
    ):
        """Forward with Transformer encoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            feat (Tensor): Sequential features, has shape (bs, num_feat_points,
                dim).
            feat_mask (Tensor): ByteTensor, the padding mask of the features,
                has shape (bs, num_feat_points).
            feat_pos (Tensor): The positional embeddings of the features, has
                shape (bs, num_feat_points, dim).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).

        Returns:
            dict: The dictionary of encoder outputs, which includes the
            `memory` of the encoder output.
        """
        memory = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
        )
        bs, _, c = memory.shape
        # get the last feature level from memory
        w, h = spatial_shapes[-1]
        # CRAU and CRAP pyramid features
        memory = memory[:, -w * h :].permute(0, 2, 1).view(bs, c, w, h)
        memory_list = self.fpn(img_feats, memory)

        # pred_maps = self.aux_bbox_head(memory_list)[0]
        aux_head_output = (
            self.aux_bbox_head.forward_aux(
                memory_list[self.aux_start_level : self.aux_end_level],
                batch_data_samples,
                cfg=self.aux_pred_cfg,
            )
            if self.training
            else None
        )
        return memory_list, aux_head_output

    def memory_flatten(
        self, mlvl_feats: Tuple[Tensor], batch_data_samples: OptSampleList = None
    ) -> Tuple[Dict]:
        """Process image features before feeding them to the transformer.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            mlvl_feats (tuple[Tensor]): Multi-level features that may have
                different resolutions, output from neck. Each feature has
                shape (bs, dim, h_lvl, w_lvl), where 'lvl' means 'layer'.
            batch_data_samples (list[:obj:`DetDataSample`], optional): The
                batch data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The first dict contains the inputs of encoder and the
            second dict contains the inputs of decoder.

            - encoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_encoder()`, which includes 'feat', 'feat_mask',
              and 'feat_pos'.
            - decoder_inputs_dict (dict): The keyword args dictionary of
              `self.forward_decoder()`, which includes 'memory_mask'.
        """
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all(
            [s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list]
        )
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = [None] * len(mlvl_feats)
        else:
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.
            mlvl_masks = [
                F.interpolate(masks[None], size=feat.shape[-2:])
                .to(torch.bool)
                .squeeze(0)
                for feat in mlvl_feats
            ]
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for feat, mask in zip(mlvl_feats, mlvl_masks):
            batch_size, c, *_ = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            mask_flatten.append(mask.flatten(1) if mask is not None else None)

            feat_flatten.append(feat)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        return dict(
            memory=feat_flatten,
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            batch_data_samples=batch_data_samples,
        )

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        batch_data_samples: OptSampleList = None,
        aux_head_output: List[InstanceData] = None,
    ) -> Tuple[Dict, Dict]:
        """Prepare intermediate variables before entering Transformer decoder,
        such as `query`, `query_pos`, and `reference_points`.

        Args:
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points). Will only be used when
                `as_two_stage` is `True`.
            spatial_shapes (Tensor): Spatial shapes of features in all levels.
                With shape (num_levels, 2), last dimension represents (h, w).
                Will only be used when `as_two_stage` is `True`.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
                Defaults to None.

        Returns:
            tuple[dict]: The decoder_inputs_dict and head_inputs_dict.

            - decoder_inputs_dict (dict): The keyword dictionary args of
              `self.forward_decoder()`, which includes 'query', 'memory',
              `reference_points`, and `dn_mask`. The reference points of
              decoder input here are 4D boxes, although it has `points`
              in its name.
            - head_inputs_dict (dict): The keyword dictionary args of the
              bbox_head functions, which includes `topk_score`, `topk_coords`,
              and `dn_meta` when `self.training` is `True`, else is empty.
        """
        bs, _, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers
        ].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes
        )
        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](
            output_memory
        )
        enc_outputs_coord_unact = (
            self.bbox_head.reg_branches[self.decoder.num_layers](output_memory)
            + output_proposals
        )

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1
        )[1]
        topk_score = torch.gather(
            enc_outputs_class,
            1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features),
        )
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4)
        )
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        query = self.query_embedding.weight.repeat(bs, 1, 1)
        if self.training and self.dn_query_generator is not None:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(
                batch_data_samples, aux_head_output
            )
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = (
            dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta,
            )
            if self.training
            else dict()
        )
        return decoder_inputs_dict, head_inputs_dict

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Dict:
        """Forward with Transformer decoder.

        The forward procedure of the transformer is defined as:
        'pre_transformer' -> 'encoder' -> 'pre_decoder' -> 'decoder'
        More details can be found at `TransformerDetector.forward_transformer`
        in `mmdet/detector/base_detr.py`.

        Args:
            query (Tensor): The queries of decoder inputs, has shape
                (bs, num_queries_total, dim), where `num_queries_total` is the
                sum of `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            memory (Tensor): The output embeddings of the Transformer encoder,
                has shape (bs, num_feat_points, dim).
            memory_mask (Tensor): ByteTensor, the padding mask of the memory,
                has shape (bs, num_feat_points).
            reference_points (Tensor): The initial reference, has shape
                (bs, num_queries_total, 4) with the last dimension arranged as
                (cx, cy, w, h).
            spatial_shapes (Tensor): Spatial shapes of features in all levels,
                has shape (num_levels, 2), last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels, ) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
            valid_ratios (Tensor): The ratios of the valid width and the valid
                height relative to the width and the height of features in all
                levels, has shape (bs, num_levels, 2).
            dn_mask (Tensor, optional): The attention mask to prevent
                information leakage from different denoising groups and
                matching parts, will be used as `self_attn_mask` of the
                `self.decoder`, has shape (num_queries_total,
                num_queries_total).
                It is `None` when `self.training` is `False`.

        Returns:
            dict: The dictionary of decoder outputs, which includes the
            `hidden_states` of the decoder output and `references` including
            the initial and intermediate reference_points.
        """
        inter_states, references = self.decoder(
            query=query,
            value=memory,
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs,
        )

        if len(query) == self.num_queries and self.dn_query_generator is not None:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += (
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0
            )

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references)
        )
        return decoder_outputs_dict

    def loss(
        self, batch_inputs: Tensor, batch_data_samples: SampleList
    ) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (bs, dim, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        img_feats = self.extract_feat(batch_inputs)
        pred_maps, head_inputs_dict = self.forward_transformer(
            img_feats, batch_data_samples
        )

        bbox_losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples
        )

        aux_bbox_losses = self.aux_bbox_head.loss(
            pred_maps[self.aux_start_level : self.aux_end_level],
            batch_data_samples=batch_data_samples,
        )

        return dict(
            **bbox_losses,
            **{"aux." + k: sum(v) for k, v in aux_bbox_losses.items()},
        )

    def predict(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        rescale: bool = True,
        **kwargs,
    ) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        img_feats = self.extract_feat(batch_inputs)
        _, head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict, rescale=rescale, batch_data_samples=batch_data_samples
        )
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list
        )
        return batch_data_samples
