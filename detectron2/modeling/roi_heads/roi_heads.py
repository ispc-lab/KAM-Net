# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry

from ..backbone.resnet import BottleneckBlock, make_stage
from ..matcher import Matcher
from ..poolers import ROIPooler
from ..proposal_generator.proposal_utils import add_ground_truth_to_proposals
from ..sampling import subsample_labels
from .box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers
from .keypoint_head import build_keypoint_head
from .mask_head import build_mask_head


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .corner_target import corner_target
from mmcv.cnn import normal_init
# from mmdet.ops import DeformConv
from mmdet.models.losses import smooth_l1_loss





ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)


# def build_roi_heads(cfg, input_shape):
#     """
#     Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
#     """
#     name = cfg.MODEL.ROI_HEADS.NAME
#     return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)



def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)


def select_foreground_proposals(
    proposals: List[Instances], bg_label: int
) -> Tuple[List[Instances], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Instances]) -> List[Instances]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        if len(proposals_per_image) == 0:
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = torch.nonzero(selection, as_tuple=True)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg))
    return ret


class ROIHeads(torch.nn.Module):
    """
    ROIHeads perform all per-region computation in an R-CNN.

    It contains logic of cropping the regions, extract per-region features,
    and make per-region predictions.

    It can have many variants, implemented as subclasses of this class.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(ROIHeads, self).__init__()
        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

    def _sample_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances]
    ) -> List[Instances]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_sample_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        gt_boxes = [x.gt_boxes for x in targets]
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        Args:
            images (ImageList):
            features (dict[str,Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            proposals (list[Instances]): length `N` list of `Instances`. The i-th
                `Instances` contains object proposals for the i-th input image,
                with fields "proposal_boxes" and "objectness_logits".
            targets (list[Instances], optional): length `N` list of `Instances`. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.  Specify `targets` during training only.
                It may have the following fields:

                - gt_boxes: the bounding box of each instance.
                - gt_classes: the label for each instance with a category ranging in [0, #class].
                - gt_masks: PolygonMasks or BitMasks, the ground-truth masks of each instance.
                - gt_keypoints: NxKx3, the groud-truth keypoints for each instance.

        Returns:
            list[Instances]: length `N` list of `Instances` containing the
            detected instances. Returned during inference only; may be [] during training.

            dict[str->Tensor]:
            mapping from a named loss to a tensor storing the loss. Used during training only.
        """
        raise NotImplementedError()


@ROI_HEADS_REGISTRY.register()
class Res5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)

        assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            first_stride=2,
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def forward(self, images, features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        predictions = self.box_predictor(box_features.mean(dim=[2, 3]))

        if self.training:
            del features
            losses = self.box_predictor.losses(predictions, proposals)
            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                # Since the ROI feature transform is shared between boxes and masks,
                # we don't need to recompute features. The mask loss is only defined
                # on foreground proposals, so we need to select out the foreground
                # features.
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))
            return [], losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances


@ROI_HEADS_REGISTRY.register()
class StandardROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    The cropped rois go to separate branches (boxes and masks) directly.
    This way, it is easier to make separate abstractions for different branches.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    def __init__(self, cfg, input_shape):
        super(StandardROIHeads, self).__init__(cfg, input_shape)
        self._init_box_head(cfg, input_shape)
        self._init_mask_head(cfg, input_shape)
        self._init_keypoint_head(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(cfg, self.box_head.output_shape)

    def _init_mask_head(self, cfg, input_shape):
        # fmt: off
        self.mask_on           = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def _init_keypoint_head(self, cfg, input_shape):
        # fmt: off
        self.keypoint_on  = cfg.MODEL.KEYPOINT_ON
        if not self.keypoint_on:
            return
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in self.in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features][0]

        self.keypoint_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.keypoint_head = build_keypoint_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self, features: Dict[str, torch.Tensor], proposals: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return self.box_predictor.losses(predictions, proposals)
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is defined on positive proposals with at >=1 visible keypoints.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposals = select_proposals_with_visible_keypoints(proposals)
            proposal_boxes = [x.proposal_boxes for x in proposals]

            keypoint_features = self.keypoint_pooler(features, proposal_boxes)
            return self.keypoint_head(keypoint_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            keypoint_features = self.keypoint_pooler(features, pred_boxes)
            return self.keypoint_head(keypoint_features, instances)



class pool(nn.Module):
    def __init__(self, dim, pool1, pool2):  # pool1, pool2 should be Class name
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1 = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        # pool 1
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)

        # pool 2
        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2

class pool_new(nn.Module):
    def __init__(self, dim, pool1, pool2):
        super(pool, self).__init__()
        self.p1_conv1 = convolution(3, dim, 128)
        self.p2_conv1 = convolution(3, dim, 128)

        self.p_conv1 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.p_bn1   = nn.BatchNorm2d(dim)

        self.conv1 = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = convolution(3, dim, dim)

        self.pool1 = pool1()
        self.pool2 = pool2()

        self.look_conv1 = convolution(3, dim, 128)
        self.look_conv2 = convolution(3, dim, 128)
        self.P1_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)
        self.P2_look_conv = nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=False)

    def forward(self, x):
        # pool 1
        look_conv1   = self.look_conv1(x)
        p1_conv1     = self.p1_conv1(x)
        look_right   = self.pool2(look_conv1)
        P1_look_conv = self.P1_look_conv(p1_conv1+look_right)
        pool1        = self.pool1(P1_look_conv)

        # pool 2
        look_conv2   = self.look_conv2(x)
        p2_conv1 = self.p2_conv1(x)
        look_down   = self.pool1(look_conv2)
        P2_look_conv = self.P2_look_conv(p2_conv1+look_down)
        pool2    = self.pool2(P2_look_conv)

        # pool 1 + pool 2
        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1   = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1   = self.bn1(conv1)
        relu1 = self.relu1(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


class TopLeftPool(pool):
    def __init__(self, dim):
        super(TopLeftPool, self).__init__(dim, TopPool, LeftPool)


class BottomRightPool(pool):
    def __init__(self, dim):
        super(BottomRightPool, self).__init__(dim, BottomPool, RightPool)


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


def top_pool(x):  # from right to left
    """
    :param x:feature map x, a Tensor
    :return: feature map with the same size as x
    """
    x_p = torch.zeros_like(x)
    x_p[:, :, :, -1] = x[:, :, :, -1]
    _, _, h, w = x.size()
    for col in range(w - 1, -1, -1):
        x_p[:, :, :, col] = x[:, :, :, col:].max(-1)[0]

    return x_p


def left_pool(x):  # from bottom to top
    x_p = torch.zeros_like(x)
    x_p[:, :, -1, :] = x[:, :, -1, :]
    _, _, h, w = x.size()
    for row in range(h - 1, -1, -1):
        x_p[:, :, row, :] = x[:, :, row:, :].max(-2)[0]

    return x_p


def bottom_pool(x):  # from left to right
    x_p = torch.zeros_like(x)
    x_p[:, :, :, 0] = x[:, :, :, 0]
    _, _, h, w = x.size()
    for col in range(1, w):
        x_p[:, :, :, col] = x[:, :, :, 0:col + 1].max(-1)[0]

    return x_p


def right_pool(x):  # from up to bottom
    x_p = torch.zeros_like(x)
    x_p[:, :, 0, :] = x[:, :, 0, :]
    _, _, h, w = x.size()
    for row in range(1, h):
        x_p[:, :, row, :] = x[:, :, 0:row + 1, :].max(-2)[0]

    return x_p


def det_loss_(preds, gt, Epsilon=1e-12):
    # TODO: add Gaussian to gt_heatmap
    # _, t_num = gt.view([gt.size(0), -1]).size()
    pos_weights = (gt == 1.0).type_as(gt)
    neg_weights = torch.pow(1 - gt, 4).type_as(gt)
    pos_loss = -torch.log(preds + Epsilon) * torch.pow(1 - preds, 2) * pos_weights
    neg_loss = -torch.log(1 - preds + Epsilon) * torch.pow(preds, 2) * neg_weights
    # obj_num = pos_weights.sum(-1).sum(-1).sum(-1)
    obj_num = pos_weights.sum()
    # loss = pos_loss.sum(-1).sum(-1).sum(-1)/obj_num + neg_loss.sum(-1).sum(-1).sum(-1)/(t_num-obj_num)
    if obj_num < 1:
        loss = neg_loss.sum()
    else:
        loss = (pos_loss + neg_loss).sum() / obj_num

    return loss


def _neg_loss(preds, gt, Epsilon=1e-12):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    #
    neg_weights = torch.pow(1 - gt[neg_inds], 4)
    #
    loss = 0
    for pred in preds:
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]
        #
        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights
        #
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        #
        # avoid the error when num_pos is zero
        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def off_loss_(preds, target, mask):
    """
    :param preds: pred_offsets
    :param gt:  gt_offsets
    :param mask: denotes where is those corners
    :return: smooth l1 loss of offsets
    """
    mask = (mask.sum(1) > 0).unsqueeze(1).type_as(preds)
    preds *= mask
    target *= mask

    return smooth_l1_loss(preds, target, reduction='none')


def ae_loss_(tl_preds, br_preds, match):
    """
    :param tl_preds: predicted tensor of top-left embedding
    :param br_preds: predicted tensor of bottom-right embedding
    :param match:
    :return: pull loss and push loss
    """
    b = tl_preds.size(0)

    loss = 0
    pull = 0
    push = 0
    for i in range(b):
        # loss += ae_loss_per_image(tl_preds[i], br_preds[i], match[i])
        loss = ae_loss_per_image(tl_preds[i], br_preds[i], match[i])
        pull += loss[0]
        push += loss[1]
    # return loss
    return pull, push


def ae_loss_per_image(tl_preds, br_preds, match, pull_weight=0.25, push_weight=0.25):
    tl_list = torch.Tensor([]).type_as(tl_preds)
    br_list = torch.Tensor([]).type_as(tl_preds)
    me_list = torch.Tensor([]).type_as(tl_preds)
    for m in match:
        tl_y = m[0][0]
        tl_x = m[0][1]
        br_y = m[1][0]
        br_x = m[1][1]
        tl_e = tl_preds[:, tl_y, tl_x]
        br_e = br_preds[:, br_y, br_x]
        tl_list = torch.cat([tl_list, tl_e])
        br_list = torch.cat([br_list, br_e])
        me_list = torch.cat([me_list, ((tl_e + br_e) / 2.0)])

    assert tl_list.size() == br_list.size()

    N = tl_list.size(0)

    if N > 0:
        pull_loss = (torch.pow(tl_list - me_list, 2) + torch.pow(br_list - me_list, 2)).sum() / N
    else:
        pull_loss = 0

    margin = 1
    push_loss = 0
    for i in range(N):
        mask = torch.ones(N, device=tl_preds.device)
        mask[i] = 0
        push_loss += (mask * F.relu(margin - abs(me_list[i] - me_list))).sum()

    if N > 1:
        push_loss /= (N * (N - 1))
    else:
        pass
    '''if N>0:
        N2 = N*(N-1)
        x0 = me_list.unsqueeze(0)
        x1 = me_list.unsqueeze(1)
        push_loss = (F.relu(1 - torch.abs(x0-x1))-1/(N+1e-4))/(N2+1e-4)
        #push_loss -= 1/(N+1e-4)
        #push_loss /= (N2+1e-4)
        push_loss = push_loss.sum()
    else:
        push_loss = 0'''

    return pull_weight * pull_loss, push_weight * push_loss


def make_kp_layer(out_dim, cnv_dim=256, curr_dim=256):
    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


def _sigmoid(x):
    x = torch.clamp(torch.sigmoid(x), min=1e-4, max=1 - 1e-4)
    return x


# class Hourhead(nn.Module):
#     def __init__(self,
#                  num_classes=2,
#                  in_channels=256, with_mask=False):
#         super(Centripetal_mask, self).__init__()
#         self.num_classes = num_classes - 1
#         self.in_channels = in_channels
#
#         self.tl_out_channels = self.num_classes + 2 + 2  # 2 is the dim for offset map, as there are 2 coordinates, x,y
#         self.br_out_channels = self.num_classes + 2 + 2
#
#         self.convs = nn.ModuleList()
#         self.mid_convs = nn.ModuleList()
#
#         self.with_mask = with_mask
#
#         self._init_layers()
#
#     def _init_layers(self):
#
#         self.tl_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
#         self.br_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
#         self.mid_tl_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
#         self.mid_br_fadp = DeformConv(self.in_channels, self.in_channels, 3, 1, 1)
#
#         self.tl_offset = nn.Conv2d(2, 18, 1, bias=False)
#         self.br_offset = nn.Conv2d(2, 18, 1, bias=False)
#         self.mid_tl_offset = nn.Conv2d(2, 18, 1, bias=False)
#         self.mid_br_offset = nn.Conv2d(2, 18, 1, bias=False)
#
#         #         self.tl_pool = TopLeftPool(self.in_channels)
#         #         self.br_pool = BottomRightPool(self.in_channels)
#         #         self.mid_tl_pool = TopLeftPool(self.in_channels)
#         #         self.mid_br_pool = BottomRightPool(self.in_channels)
#
#         self.tl_heat = make_kp_layer(out_dim=self.num_classes)
#         self.br_heat = make_kp_layer(out_dim=self.num_classes)
#
#         self.tl_off_c = make_kp_layer(out_dim=2)
#         self.br_off_c = make_kp_layer(out_dim=2)
#
#         self.tl_off_c_2 = make_kp_layer(out_dim=2)
#         self.br_off_c_2 = make_kp_layer(out_dim=2)
#
#         self.tl_off = make_kp_layer(out_dim=2)
#         self.br_off = make_kp_layer(out_dim=2)
#
#         # middle supervision
#
#         self.mid_tl_heat = make_kp_layer(out_dim=self.num_classes)
#         self.mid_br_heat = make_kp_layer(out_dim=self.num_classes)
#
#         self.mid_tl_off_c = make_kp_layer(out_dim=2)
#         self.mid_br_off_c = make_kp_layer(out_dim=2)
#
#         self.mid_tl_off_c_2 = make_kp_layer(out_dim=2)
#         self.mid_br_off_c_2 = make_kp_layer(out_dim=2)
#
#         self.mid_tl_off = make_kp_layer(out_dim=2)
#         self.mid_br_off = make_kp_layer(out_dim=2)
#
#         if self.with_mask:
#             for i in range(4):
#                 self.convs.append(
#                     ConvModule(self.in_channels, self.in_channels, 3, padding=1)
#                 )
#                 self.mid_convs.append(
#                     ConvModule(self.in_channels, self.in_channels, 3, padding=1)
#                 )
#
#             self.conv_logits = nn.Conv2d(self.in_channels, 81, 1)
#             self.mid_conv_logits = nn.Conv2d(self.in_channels, 81, 1)
#
#     def init_weights(self):
#         """
#         TODO: weight init method
#         """
#         self.tl_heat[-1].bias.data.fill_(-2.19)
#         self.br_heat[-1].bias.data.fill_(-2.19)
#         self.mid_tl_heat[-1].bias.data.fill_(-2.19)
#         self.mid_br_heat[-1].bias.data.fill_(-2.19)
#         normal_init(self.tl_offset, std=0.1)
#         normal_init(self.tl_fadp, std=0.01)
#         normal_init(self.br_offset, std=0.1)
#         normal_init(self.br_fadp, std=0.01)
#         normal_init(self.mid_tl_offset, std=0.1)
#         normal_init(self.mid_tl_fadp, std=0.01)
#         normal_init(self.mid_br_offset, std=0.1)
#         normal_init(self.mid_br_fadp, std=0.01)
#
#     def forward_single(self, feats):
#         '''tl_result = self.tl_branch(x)
#         br_result = self.br_branch(x)'''
#         x = feats[-1]
#         mask = None
#         mask_mid = None
#         if self.with_mask:
#             mask = x
#             for conv in self.convs:
#                 mask = conv(mask)
#             mask = self.conv_logits(mask)
#
#         tl_pool = x
#         tl_heat = self.tl_heat(x)
#         tl_off_c = self.tl_off_c(tl_pool)  # 可变形卷积offset
#         tl_off = self.tl_off(tl_pool)
#         tl_offmap = self.tl_offset(tl_off_c.detach())  # 到可变形卷积的各个点
#         x_tl_fadp = self.tl_fadp(tl_pool, tl_offmap)
#         tl_off_c_2 = self.tl_off_c_2(x_tl_fadp)  # 矢量
#
#         br_pool = x
#         br_heat = self.br_heat(x)
#         br_off_c = self.br_off_c(br_pool)
#         br_off = self.br_off(br_pool)
#         br_offmap = self.br_offset(br_off_c.detach())
#         x_br_fadp = self.br_fadp(br_pool, br_offmap)
#         br_off_c_2 = self.br_off_c_2(x_br_fadp)
#
#         tl_result = torch.cat([tl_heat, tl_off_c, tl_off_c_2, tl_off], 1)
#         br_result = torch.cat([br_heat, br_off_c, br_off_c_2, br_off], 1)
#
#         x = feats[0]
#
#         if self.with_mask:
#             mask_mid = x
#             for conv in self.mid_convs:
#                 mask_mid = conv(mask_mid)
#             mask_mid = self.mid_conv_logits(mask_mid)
#
#         tl_pool_mid = x
#         tl_heat_mid = self.mid_tl_heat(x)
#         tl_off_c_mid = self.mid_tl_off_c(tl_pool_mid)
#         tl_off_mid = self.mid_tl_off(tl_pool_mid)
#         tl_offmap_mid = self.mid_tl_offset(tl_off_c_mid.detach())
#         x_tl_fadp_mid = self.mid_tl_fadp(tl_pool_mid, tl_offmap_mid)
#         tl_off_c_2_mid = self.mid_tl_off_c_2(x_tl_fadp_mid)
#
#         br_pool_mid = x
#         br_heat_mid = self.mid_br_heat(x)
#         br_off_c_mid = self.mid_br_off_c(br_pool_mid)
#         br_off_mid = self.mid_br_off(br_pool_mid)
#         br_offmap_mid = self.mid_br_offset(br_off_c_mid.detach())
#         x_br_fadp_mid = self.mid_br_fadp(br_pool_mid, br_offmap_mid)
#         br_off_c_2_mid = self.mid_br_off_c_2(x_br_fadp_mid)
#
#         tl_result_mid = torch.cat([tl_heat_mid, tl_off_c_mid, tl_off_c_2_mid, tl_off_mid], 1)
#         br_result_mid = torch.cat([br_heat_mid, br_off_c_mid, br_off_c_2_mid, br_off_mid], 1)
#
#         if self.with_mask:
#             return tl_result, br_result, mask, tl_result_mid, br_result_mid, mask_mid
#         else:
#             return tl_result, br_result, None, tl_result_mid, br_result_mid, None
#
#     def forward(self, feats):
#         """
#         :param feats: different layer's feature
#         :return: the raw results
#         """
#         feat = feats  # [-1]# we only use the feature of the last layer
#         return self.forward_single(feat)
#
#     def loss(self, tl_result, br_result, mask, mid_tl_result, mid_br_result, mid_mask, gt_bboxes, gt_labels, gt_masks,
#              img_metas, cfg, imgscale):
#         gt_tl_heatmap, gt_br_heatmap, gt_tl_offsets, gt_br_offsets, gt_tl_off_c, gt_br_off_c, \
#         gt_tl_off_c2, gt_br_off_c2 = corner_target(gt_bboxes=gt_bboxes, gt_labels=gt_labels, feats=tl_result,
#                                                    imgscale=imgscale, direct=True, scale=1.0, dcn=True)
#         # pred_tl_heatmap = _sigmoid(tl_result[:, :self.num_classes, :, :])
#         pred_tl_heatmap = tl_result[:, :self.num_classes, :, :].sigmoid()
#         pred_tl_off_c = tl_result[:, self.num_classes:self.num_classes + 2, :, :]
#         pred_tl_off_c2 = tl_result[:, self.num_classes + 2:self.num_classes + 4, :, :]
#         pred_tl_offsets = tl_result[:, -2:, :, :]
#         # pred_br_heatmap = _sigmoid(br_result[:, :self.num_classes, :, :])
#         pred_br_heatmap = br_result[:, :self.num_classes, :, :].sigmoid()
#         pred_br_off_c = br_result[:, self.num_classes:self.num_classes + 2, :, :]
#         pred_br_off_c2 = br_result[:, self.num_classes + 2:self.num_classes + 4, :, :]
#         pred_br_offsets = br_result[:, -2:, :, :]
#
#         # mid_pred_tl_heatmap = _sigmoid(mid_tl_result[:, :self.num_classes, :, :])
#         mid_pred_tl_heatmap = mid_tl_result[:, :self.num_classes, :, :].sigmoid()
#         mid_pred_tl_off_c = mid_tl_result[:, self.num_classes:self.num_classes + 2, :, :]
#         mid_pred_tl_off_c2 = mid_tl_result[:, self.num_classes + 2:self.num_classes + 4, :, :]
#         mid_pred_tl_offsets = mid_tl_result[:, -2:, :, :]
#         # mid_pred_br_heatmap = _sigmoid(mid_br_result[:, :self.num_classes, :, :])
#         mid_pred_br_heatmap = mid_br_result[:, :self.num_classes, :, :].sigmoid()
#         mid_pred_br_off_c = mid_br_result[:, self.num_classes:self.num_classes + 2, :, :]
#         mid_pred_br_off_c2 = mid_br_result[:, self.num_classes + 2:self.num_classes + 4, :, :]
#         mid_pred_br_offsets = mid_br_result[:, -2:, :, :]
#
#         tl_det_loss = det_loss_(pred_tl_heatmap, gt_tl_heatmap) + det_loss_(mid_pred_tl_heatmap, gt_tl_heatmap)
#         br_det_loss = det_loss_(pred_br_heatmap, gt_br_heatmap) + det_loss_(mid_pred_br_heatmap, gt_br_heatmap)
#         # tl_det_loss = _neg_loss([pred_tl_heatmap, mid_pred_tl_heatmap], gt_tl_heatmap)
#         # br_det_loss = _neg_loss([pred_br_heatmap, mid_pred_br_heatmap], gt_br_heatmap)
#
#         det_loss = (tl_det_loss + br_det_loss) / 2.0
#
#         tl_off_mask = gt_tl_heatmap.eq(1).type_as(gt_tl_heatmap)
#         br_off_mask = gt_br_heatmap.eq(1).type_as(gt_br_heatmap)
#
#         tl_off_c_loss = off_loss_(pred_tl_off_c, gt_tl_off_c, mask=tl_off_mask) + off_loss_(mid_pred_tl_off_c,
#                                                                                             gt_tl_off_c,
#                                                                                             mask=tl_off_mask)
#         br_off_c_loss = off_loss_(pred_br_off_c, gt_br_off_c, mask=br_off_mask) + off_loss_(mid_pred_br_off_c,
#                                                                                             gt_br_off_c,
#                                                                                             mask=br_off_mask)
#         off_c_loss = tl_off_c_loss.sum() / tl_off_mask.sum() + br_off_c_loss.sum() / br_off_mask.sum()
#         off_c_loss /= 2.0
#         off_c_loss *= 0.05
#
#         tl_off_c2_loss = off_loss_(pred_tl_off_c2, gt_tl_off_c2, mask=tl_off_mask) + off_loss_(mid_pred_tl_off_c2,
#                                                                                                gt_tl_off_c2,
#                                                                                                mask=tl_off_mask)
#         br_off_c2_loss = off_loss_(pred_br_off_c2, gt_br_off_c2, mask=br_off_mask) + off_loss_(mid_pred_br_off_c2,
#                                                                                                gt_br_off_c2,
#                                                                                                mask=br_off_mask)
#         off_c2_loss = tl_off_c2_loss.sum() / tl_off_mask.sum() + br_off_c2_loss.sum() / br_off_mask.sum()
#         off_c2_loss /= 2.0
#
#         tl_off_loss = off_loss_(pred_tl_offsets, gt_tl_offsets, mask=tl_off_mask) + off_loss_(mid_pred_tl_offsets,
#                                                                                               gt_tl_offsets,
#                                                                                               mask=tl_off_mask)
#         br_off_loss = off_loss_(pred_br_offsets, gt_br_offsets, mask=br_off_mask) + off_loss_(mid_pred_br_offsets,
#                                                                                               gt_br_offsets,
#                                                                                               mask=br_off_mask)
#         off_loss = tl_off_loss.sum() / tl_off_mask.sum() + br_off_loss.sum() / br_off_mask.sum()
#         off_loss /= 2.0
#
#         mask_loss = 0
#         if self.with_mask:
#             for b_id in range(len(gt_labels)):
#                 for mask_id in range(len(gt_labels[b_id])):
#                     mask_label = gt_labels[b_id][mask_id]
#                     m_pred = mask[b_id][mask_label]
#                     mid_m_pred = mid_mask[b_id][mask_label]
#                     m_gt = torch.from_numpy(gt_masks[b_id][mask_id]).float().cuda()
#                     mask_loss += F.binary_cross_entropy_with_logits(m_pred, m_gt)
#                     mask_loss += F.binary_cross_entropy_with_logits(mid_m_pred, m_gt)
#             mask_loss /= mask.size(0)
#             mask_loss /= 2.0
#
#         # return dict(det_loss=det_loss, ae_loss=ae_loss, off_loss=off_loss)
#         if self.with_mask:
#             return dict(det_loss=det_loss, off_c_loss=off_c_loss, off_c2_loss=off_c2_loss, off_loss=off_loss,
#                         mask_loss=mask_loss)
#         else:
#             return dict(det_loss=det_loss, off_c_loss=off_c_loss, off_c2_loss=off_c2_loss, off_loss=off_loss)
