# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule, Sequential

from mmocr.models.builder import BACKBONES


@BACKBONES.register_module()
class VeryDeepVgg(BaseModule):
    """Implement VGG-VeryDeep backbone for text recognition, modified from
    `VGG-VeryDeep <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        leaky_relu (bool): Use leakyRelu or not.
        input_channels (int): Number of channels of input image tensor.
    """

    def __init__(self,
                 input_channels=3,
                 output_channel=512,
                 init_cfg=[
                     dict(type='Xavier', layer='Conv2d'),
                     dict(type='Uniform', layer='BatchNorm2d')
                 ]):
        super().__init__(init_cfg=init_cfg)

        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2), output_channel
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channels, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 64x16x50
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  # 128x8x25
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),  # 256x8x25
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 256x4x25
            nn.Conv2d(
                self.output_channel[2],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),  # 512x4x25
            nn.Conv2d(
                self.output_channel[3],
                self.output_channel[3],
                3,
                1,
                1,
                bias=False),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),  # 512x2x25
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True))  # 512x1x24

    def forward(self, input):
        return self.ConvNet(input)

    def out_channels(self):
        return 512
