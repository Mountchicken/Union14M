# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import Sequential

from mmocr.models.builder import DECODERS
from .base_decoder import BaseDecoder


@DECODERS.register_module()
class CRNNDecoder(BaseDecoder):
    """Decoder for CRNN.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        rnn_flag (bool): Use RNN or CNN as the decoder.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 in_channels=None,
                 num_classes=None,
                 rnn_flag=True,
                 init_cfg=dict(type='Xavier', layer='Conv2d'),
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.rnn_flag = rnn_flag

        self.rnn = nn.LSTM(
            512, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(512, num_classes)

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        assert feat.size(2) == 1, 'feature height must be 1'
        x = feat.squeeze(2)  # [N, C, W]
        x = x.transpose(2, 1).contiguous()  # [N, W, C]
        x, _ = self.rnn(x)  # [N, W, C]
        x = self.fc(x)  # [N, W, C]
        return x

    def forward_test(self, feat, out_enc, img_metas):
        """
        Args:
            feat (Tensor): A Tensor of shape :math:`(N, H, 1, W)`.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, W, C)` where
            :math:`C` is ``num_classes``.
        """
        return self.forward_train(feat, out_enc, None, img_metas)
