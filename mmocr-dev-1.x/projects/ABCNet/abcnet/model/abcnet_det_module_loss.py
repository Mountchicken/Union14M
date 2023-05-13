# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.utils import multi_apply
from mmdet.utils import reduce_mean
from torch import Tensor

from mmocr.models.textdet.module_losses.base import BaseTextDetModuleLoss
from mmocr.registry import MODELS, TASK_UTILS
from mmocr.structures import TextDetDataSample
from mmocr.utils import ConfigType, DetSampleList, RangeType
from ..utils import poly2bezier

INF = 1e8


@MODELS.register_module()
class ABCNetDetModuleLoss(BaseTextDetModuleLoss):
    # TODO add docs

    def __init__(
        self,
        num_classes: int = 1,
        bbox_coder: ConfigType = dict(type='mmdet.DistancePointBBoxCoder'),
        regress_ranges: RangeType = ((-1, 64), (64, 128), (128, 256),
                                     (256, 512), (512, INF)),
        strides: List[int] = (8, 16, 32, 64, 128),
        center_sampling: bool = True,
        center_sample_radius: float = 1.5,
        norm_on_bbox: bool = True,
        loss_cls: ConfigType = dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox: ConfigType = dict(type='mmdet.GIoULoss', loss_weight=1.0),
        loss_centerness: ConfigType = dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bezier: ConfigType = dict(
            type='mmdet.SmoothL1Loss', reduction='mean', loss_weight=1.0)
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.prior_generator = MlvlPointGenerator(strides)
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.loss_centerness = MODELS.build(loss_centerness)
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_bezier = MODELS.build(loss_bezier)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if use_sigmoid_cls:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

    def forward(self, inputs: Tuple[Tensor],
                data_samples: DetSampleList) -> Dict:
        """Compute ABCNet loss.

        Args:
            inputs (tuple(tensor)): Raw predictions from model, containing
                ``cls_scores``, ``bbox_preds``, ``beizer_preds`` and
                ``centernesses``.
                Each is a tensor of shape :math:`(N, H, W)`.
            data_samples (list[TextDetDataSample]): The data samples.

        Returns:
            dict: The dict for abcnet-det losses with loss_cls, loss_bbox,
            loss_centerness and loss_bezier.
        """
        cls_scores, bbox_preds, centernesses, beizer_preds = inputs
        assert len(cls_scores) == len(bbox_preds) == len(centernesses) == len(
            beizer_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        labels, bbox_targets, bezier_targets = self.get_targets(
            all_level_points, data_samples)

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_bezier_preds = [
            bezier_pred.permute(0, 2, 3, 1).reshape(-1, 16)
            for bezier_pred in beizer_preds
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_bezier_preds = torch.cat(flatten_bezier_preds)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_bezier_targets = torch.cat(bezier_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bezier_preds = flatten_bezier_preds[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)
        pos_bezier_targets = flatten_bezier_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_target_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=num_pos)
            loss_bezier = self.loss_bezier(
                pos_bezier_preds,
                pos_bezier_targets,
                weight=pos_centerness_targets[:, None],
                avg_factor=centerness_denorm)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_bezier = pos_bezier_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_bezier=loss_bezier)

    def get_targets(self, points: List[Tensor], data_samples: DetSampleList
                    ) -> Tuple[List[Tensor], List[Tensor]]:
        """Compute regression, classification and centerness targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            data_samples: Batch of data samples. Each data sample contains
                a gt_instance, which usually includes bboxes and labels
                attributes.

        Returns:
            tuple: Targets of each level.

            - concat_lvl_labels (list[Tensor]): Labels of each level.
            - concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
            level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, bezier_targets_list = multi_apply(
            self._get_targets_single,
            data_samples,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        bezier_targets_list = [
            bezier_targets.split(num_points, 0)
            for bezier_targets in bezier_targets_list
        ]
        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_bezier_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            bezier_targets = torch.cat(
                [bezier_targets[i] for bezier_targets in bezier_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
                bezier_targets = bezier_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_bezier_targets.append(bezier_targets)
        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_bezier_targets)

    def _get_targets_single(self, data_sample: TextDetDataSample,
                            points: Tensor, regress_ranges: Tensor,
                            num_points_per_lvl: List[int]
                            ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        gt_instances = data_sample.gt_instances
        gt_instances = gt_instances[~gt_instances.ignored]
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        data_sample.gt_instances = gt_instances
        polygons = gt_instances.polygons
        beziers = gt_bboxes.new([poly2bezier(poly) for poly in polygons])
        gt_instances.beziers = beziers
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4)), \
                   gt_bboxes.new_zeros((num_points, 16))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        beziers = beziers.reshape(-1, 8,
                                  2)[None].expand(num_points, num_gts, 8, 2)
        beziers_left = beziers[..., 0] - xs[..., None]
        beziers_right = beziers[..., 1] - ys[..., None]
        bezier_targets = torch.stack((beziers_left, beziers_right), dim=-1)
        bezier_targets = bezier_targets.view(num_points, num_gts, 16)
        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0])
            & (max_regress_distance <= regress_ranges[..., 1]))

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        bezier_targets = bezier_targets[range(num_points), min_area_inds]

        return labels, bbox_targets, bezier_targets

    def centerness_target(self, pos_bbox_targets: Tensor) -> Tensor:
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        if len(left_right) == 0:
            centerness_targets = left_right[..., 0]
        else:
            centerness_targets = (
                left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                    top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
