# Copyright (c) OpenMMLab. All rights reserved.
import unittest

import torch

from mmocr.models.textrecog.encoders import SATRNEncoder
from mmocr.structures import TextRecogDataSample


class TestSATRNEncoder(unittest.TestCase):

    def setUp(self):
        self.feat = torch.randn(1, 512, 8, 25)
        data_info = TextRecogDataSample()
        data_info.set_metainfo(dict(valid_ratio=1.0))
        self.data_info = [data_info]

    def test_encoder(self):
        satrn_encoder = SATRNEncoder()
        satrn_encoder.init_weights()
        satrn_encoder.train()
        out_enc = satrn_encoder(self.feat)
        self.assertEqual(out_enc.shape, torch.Size([1, 200, 512]))
        out_enc = satrn_encoder(self.feat, self.data_info)
        self.assertEqual(out_enc.shape, torch.Size([1, 200, 512]))
