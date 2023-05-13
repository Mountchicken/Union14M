# Copyright (c) OpenMMLab. All rights reserved.
from typing import Tuple

from mmengine.structures import LabelData
from torch import Tensor

from mmocr.registry import MODELS, TASK_UTILS
from mmocr.structures import TextRecogDataSample  # noqa F401
from mmocr.utils import DetSampleList, OptMultiConfig, RecSampleList
from .base_roi_head import BaseRoIHead


@MODELS.register_module()
class RecRoIHead(BaseRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""

    def __init__(self,
                 neck=None,
                 sampler: OptMultiConfig = None,
                 roi_extractor: OptMultiConfig = None,
                 rec_head: OptMultiConfig = None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if sampler is not None:
            self.sampler = TASK_UTILS.build(sampler)
        if neck is not None:
            self.neck = MODELS.build(neck)
        self.roi_extractor = MODELS.build(roi_extractor)
        self.rec_head = MODELS.build(rec_head)

    def loss(self, inputs: Tuple[Tensor], data_samples: DetSampleList) -> dict:
        """Perform forward propagation and loss calculation of the detection
        roi on the features of the upstream network.

        Args:
            x (tuple[Tensor]): List of multi-level img features.
            rpn_results_list (list[:obj:`InstanceData`]): List of region
                proposals.
            DetSampleList (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components
        """
        proposals = [
            ds.gt_instances[~ds.gt_instances.ignored] for ds in data_samples
        ]

        proposals = [p for p in proposals if len(p) > 0]
        bbox_feats = self.roi_extractor(inputs, proposals)
        rec_data_samples = [
            TextRecogDataSample(gt_text=LabelData(item=text))
            for proposal in proposals for text in proposal.texts
        ]
        return self.rec_head.loss(bbox_feats, rec_data_samples)

    def predict(self, inputs: Tuple[Tensor],
                data_samples: DetSampleList) -> RecSampleList:
        if hasattr(self, 'neck') and self.neck is not None:
            inputs = self.neck(inputs)
        pred_instances = [ds.pred_instances for ds in data_samples]
        bbox_feats = self.roi_extractor(inputs, pred_instances)
        if bbox_feats.size(0) == 0:
            return []
        len_instance = sum(
            [len(instance_data) for instance_data in pred_instances])
        rec_data_samples = [TextRecogDataSample() for _ in range(len_instance)]
        rec_data_samples = self.rec_head.predict(bbox_feats, rec_data_samples)
        return rec_data_samples
