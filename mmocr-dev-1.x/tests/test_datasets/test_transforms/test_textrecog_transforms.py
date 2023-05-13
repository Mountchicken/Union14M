# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest

import numpy as np
from parameterized import parameterized

from mmocr.datasets.transforms import (CropHeight, ImageContentJitter,
                                       PadToWidth, PyramidRescale,
                                       RescaleToHeight, ReversePixels,
                                       TextRecogGeneralAug)


class TestPadToWidth(unittest.TestCase):

    def test_pad_to_width(self):
        data_info = dict(img=np.random.random((16, 25, 3)))
        # test size and size_divisor are both set
        with self.assertRaises(AssertionError):
            PadToWidth(width=10.5)

        transform = PadToWidth(width=100)
        results = transform(copy.deepcopy(data_info))
        self.assertTupleEqual(results['img'].shape[:2], (16, 100))
        self.assertEqual(results['valid_ratio'], 25 / 100)

    def test_repr(self):
        transform = PadToWidth(width=100)
        self.assertEqual(
            repr(transform),
            ("PadToWidth(width=100, pad_cfg={'type': 'Pad'})"))


class TestPyramidRescale(unittest.TestCase):

    def setUp(self):
        self.data_info = dict(img=np.random.random((128, 100, 3)))

    def test_init(self):
        # factor is int
        transform = PyramidRescale(factor=4, randomize_factor=False)
        self.assertEqual(transform.factor, 4)
        # factor is float
        with self.assertRaisesRegex(TypeError,
                                    '`factor` should be an integer'):
            PyramidRescale(factor=4.0)
        # invalid base_shape
        with self.assertRaisesRegex(TypeError,
                                    '`base_shape` should be a list or tuple'):
            PyramidRescale(base_shape=128)
        with self.assertRaisesRegex(
                ValueError, '`base_shape` should contain two integers'):
            PyramidRescale(base_shape=(128, ))
        with self.assertRaisesRegex(
                ValueError, '`base_shape` should contain two integers'):
            PyramidRescale(base_shape=(128.0, 2.0))
        # invalid randomize_factor
        with self.assertRaisesRegex(TypeError,
                                    '`randomize_factor` should be a bool'):
            PyramidRescale(randomize_factor=None)

    def test_transform(self):
        # test if the rescale keeps the original size
        transform = PyramidRescale()
        results = transform(copy.deepcopy(self.data_info))
        self.assertEqual(results['img'].shape, (128, 100, 3))
        # test factor = 0
        transform = PyramidRescale(factor=0, randomize_factor=False)
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(np.all(results['img'] == self.data_info['img']))

    def test_repr(self):
        transform = PyramidRescale(
            factor=4, base_shape=(128, 512), randomize_factor=False)
        self.assertEqual(
            repr(transform),
            ('PyramidRescale(factor = 4, randomize_factor = False, '
             'base_w = 128, base_h = 512)'))


class TestRescaleToHeight(unittest.TestCase):

    def test_rescale_height(self):
        data_info = dict(
            img=np.random.random((16, 25, 3)),
            gt_seg_map=np.random.random((16, 25, 3)),
            gt_bboxes=np.array([[0, 0, 10, 10]]),
            gt_keypoints=np.array([[[10, 10, 1]]]))
        with self.assertRaises(AssertionError):
            RescaleToHeight(height=20.9)
        with self.assertRaises(AssertionError):
            RescaleToHeight(height=20, min_width=20.9)
        with self.assertRaises(AssertionError):
            RescaleToHeight(height=20, max_width=20.9)
        with self.assertRaises(AssertionError):
            RescaleToHeight(height=20, width_divisor=0.5)
        transform = RescaleToHeight(height=32)
        results = transform(copy.deepcopy(data_info))
        self.assertTupleEqual(results['img'].shape[:2], (32, 50))
        self.assertTupleEqual(results['scale'], (50, 32))
        self.assertTupleEqual(results['scale_factor'], (50 / 25, 32 / 16))

        # test min_width
        transform = RescaleToHeight(height=32, min_width=60)
        results = transform(copy.deepcopy(data_info))
        self.assertTupleEqual(results['img'].shape[:2], (32, 60))
        self.assertTupleEqual(results['scale'], (60, 32))
        self.assertTupleEqual(results['scale_factor'], (60 / 25, 32 / 16))

        # test max_width
        transform = RescaleToHeight(height=32, max_width=45)
        results = transform(copy.deepcopy(data_info))
        self.assertTupleEqual(results['img'].shape[:2], (32, 45))
        self.assertTupleEqual(results['scale'], (45, 32))
        self.assertTupleEqual(results['scale_factor'], (45 / 25, 32 / 16))

        # test width_divisor
        transform = RescaleToHeight(height=32, width_divisor=4)
        results = transform(copy.deepcopy(data_info))
        self.assertTupleEqual(results['img'].shape[:2], (32, 48))
        self.assertTupleEqual(results['scale'], (48, 32))
        self.assertTupleEqual(results['scale_factor'], (48 / 25, 32 / 16))

    def test_repr(self):
        transform = RescaleToHeight(height=32)
        self.assertEqual(
            repr(transform), ('RescaleToHeight(height=32, '
                              'min_width=None, max_width=None, '
                              'width_divisor=1, '
                              "resize_cfg={'type': 'Resize', 'scale': 0})"))


class TestTextRecogGeneralAug(unittest.TestCase):

    def setUp(self) -> None:
        self.transform = TextRecogGeneralAug()

    @parameterized.expand([(np.random.random((3, 3, 3)), ),
                           (np.random.random((10, 10, 3)), ),
                           (np.random.random((30, 30, 3)), )])
    def test_transform(self, img):
        data_info = dict(img=img)
        results = self.transform(copy.deepcopy(data_info))
        self.assertEqual(results['img'].shape[:2], results['img_shape'])

    def test_repr(self):
        repr_str = self.transform.__repr__()
        self.assertEqual(repr_str, 'TextRecogGeneralAug()')


class TestCropHeight(unittest.TestCase):

    def setUp(self) -> None:
        self.data_info = dict(img=np.random.random((20, 20, 3)))

    @parameterized.expand([
        (3, 3),
        (5, 10),
    ])
    def test_transform(self, min_pixels, max_pixels):
        self.transform = CropHeight(
            min_pixels=min_pixels, max_pixels=max_pixels)
        results = self.transform(copy.deepcopy(self.data_info))
        self.assertEqual(results['img'].shape[:2], results['img_shape'])
        h_diff = self.data_info['img'].shape[0] - results['img_shape'][0]
        self.assertGreaterEqual(h_diff, min_pixels)
        self.assertLessEqual(h_diff, max_pixels)

    def test_invalid(self):
        with self.assertRaises(AssertionError):
            self.transform = CropHeight(min_pixels=10, max_pixels=9)

    def test_repr(self):
        transform = CropHeight(min_pixels=2, max_pixels=10)
        repr_str = transform.__repr__()
        self.assertEqual(repr_str, 'CropHeight(min_pixels = 2, '
                         'max_pixels = 10)')


class TestImageContentJitter(unittest.TestCase):

    def setUp(self) -> None:
        self.transform = ImageContentJitter()

    @parameterized.expand([(np.random.random((3, 3, 3)), ),
                           (np.random.random((10, 10, 3)), ),
                           (np.random.random((30, 30, 3)), )])
    def test_transform(self, img):
        data_info = dict(img=img)
        self.transform(copy.deepcopy(data_info))

    def test_repr(self):
        repr_str = self.transform.__repr__()
        self.assertEqual(repr_str, 'ImageContentJitter()')


class TestReversePixels(unittest.TestCase):

    def setUp(self) -> None:
        self.transform = ReversePixels()

    @parameterized.expand([(np.random.random((3, 3, 3)), ),
                           (np.random.random((10, 10, 3)), ),
                           (np.random.random((30, 30, 3)), )])
    def test_transform(self, img):
        data_info = dict(img=img)
        results = self.transform(copy.deepcopy(data_info))
        self.assertTrue(np.array_equal(results['img'], 255. - img))

    def test_repr(self):
        repr_str = self.transform.__repr__()
        self.assertEqual(repr_str, 'ReversePixels()')
