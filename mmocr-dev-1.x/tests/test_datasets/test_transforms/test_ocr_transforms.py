# Copyright (c) OpenMMLab. All rights reserved.
import copy
import unittest
import unittest.mock as mock

import numpy as np

from mmocr.datasets.transforms import (FixInvalidPolygon, RandomCrop,
                                       RandomRotate, RemoveIgnored, Resize)
from mmocr.utils import poly2shapely


class TestRandomCrop(unittest.TestCase):

    def setUp(self):
        img = np.zeros((30, 30, 3))
        gt_polygons = [
            np.array([5., 5., 25., 5., 25., 10., 5., 10.]),
            np.array([5., 20., 25., 20., 25., 25., 5., 25.])
        ]
        gt_bboxes = np.array([[5, 5, 25, 10], [5, 20, 25, 25]])
        labels = np.array([0, 1])
        gt_ignored = np.array([True, False], dtype=bool)
        texts = ['text1', 'text2']
        self.data_info = dict(
            img=img,
            gt_polygons=gt_polygons,
            gt_bboxes=gt_bboxes,
            gt_bboxes_labels=labels,
            gt_ignored=gt_ignored,
            gt_texts=texts)

    @mock.patch('mmocr.datasets.transforms.ocr_transforms.np.random.randint')
    def test_sample_crop_box(self, mock_randint):

        def rand_min(low, high):
            return low

        trans = RandomCrop(min_side_ratio=0.3)
        mock_randint.side_effect = rand_min
        crop_box = trans._sample_crop_box((30, 30), self.data_info.copy())
        assert np.allclose(np.array(crop_box), np.array([0, 0, 25, 10]))

        def rand_max(low, high):
            return high - 1

        mock_randint.side_effect = rand_max
        crop_box = trans._sample_crop_box((30, 30), self.data_info.copy())
        assert np.allclose(np.array(crop_box), np.array([4, 19, 30, 30]))

    @mock.patch('mmocr.datasets.transforms.ocr_transforms.np.random.randint')
    def test_transform(self, mock_randint):

        def rand_min(low, high):
            return low

        # mock_randint.side_effect = [0, 0, 0, 0, 30, 0, 0, 0, 15]
        mock_randint.side_effect = rand_min
        trans = RandomCrop(min_side_ratio=0.3)
        polygon_target = np.array([5., 5., 25., 5., 25., 10., 5., 10.])
        bbox_target = np.array([[5., 5., 25., 10.]])
        results = trans(self.data_info)

        self.assertEqual(results['img'].shape, (10, 25, 3))
        self.assertEqual(results['img_shape'], (10, 25))
        self.assertTrue(np.allclose(results['gt_bboxes'], bbox_target))
        self.assertEqual(results['gt_bboxes'].shape, (1, 4))
        self.assertEqual(len(results['gt_polygons']), 1)
        self.assertTrue(np.allclose(results['gt_polygons'][0], polygon_target))
        self.assertEqual(results['gt_bboxes_labels'][0], 0)
        self.assertEqual(results['gt_ignored'][0], True)
        self.assertEqual(results['gt_texts'][0], 'text1')

        def rand_max(low, high):
            return high - 1

        mock_randint.side_effect = rand_max
        trans = RandomCrop(min_side_ratio=0.3)
        polygon_target = np.array([1, 1, 21, 1, 21, 6, 1, 6])
        bbox_target = np.array([[1, 1, 21, 6]])
        results = trans(self.data_info)

        self.assertEqual(results['img'].shape, (6, 21, 3))
        self.assertEqual(results['img_shape'], (6, 21))
        self.assertTrue(np.allclose(results['gt_bboxes'], bbox_target))
        self.assertEqual(results['gt_bboxes'].shape, (1, 4))
        self.assertEqual(len(results['gt_polygons']), 1)
        self.assertTrue(np.allclose(results['gt_polygons'][0], polygon_target))
        self.assertEqual(results['gt_bboxes_labels'][0], 0)
        self.assertTrue(results['gt_ignored'][0])
        self.assertEqual(results['gt_texts'][0], 'text1')

    def test_repr(self):
        transform = RandomCrop(min_side_ratio=0.4)
        self.assertEqual(repr(transform), ('RandomCrop(min_side_ratio = 0.4)'))


class TestRandomRotate(unittest.TestCase):

    def setUp(self):
        img = np.random.random((5, 5))
        self.data_info1 = dict(img=img.copy(), img_shape=img.shape[:2])
        self.data_info2 = dict(
            img=np.random.random((30, 30, 3)),
            gt_bboxes=np.array([[10, 10, 20, 20], [5, 5, 10, 10]]),
            img_shape=(30, 30))
        self.data_info3 = dict(
            img=np.random.random((30, 30, 3)),
            gt_polygons=[np.array([10., 10., 20., 10., 20., 20., 10., 20.])],
            img_shape=(30, 30))

    def test_init(self):
        # max angle is float
        with self.assertRaisesRegex(TypeError,
                                    '`max_angle` should be an integer'):
            RandomRotate(max_angle=16.8)
        # invalid pad value
        with self.assertRaisesRegex(
                ValueError, '`pad_value` should contain three integers'):
            RandomRotate(pad_value=[16.8, 0.1])

    def test_transform(self):
        self._test_recog()
        self._test_bboxes()
        self._test_polygons()

    def _test_recog(self):
        # test random rotate for recognition (image only) input
        transform = RandomRotate(max_angle=10)
        results = transform(copy.deepcopy(self.data_info1))
        self.assertTrue(np.allclose(results['img'], self.data_info1['img']))

    @mock.patch(
        'mmocr.datasets.transforms.ocr_transforms.np.random.random_sample')
    def _test_bboxes(self, mock_sample):
        # test random rotate for bboxes
        # returns 1. for random_sample() in _sample_angle(), i.e., angle = 90
        mock_sample.side_effect = [1.]
        transform = RandomRotate(max_angle=90, use_canvas=True)
        results = transform(copy.deepcopy(self.data_info2))
        self.assertTrue(
            np.allclose(results['gt_bboxes'][0], np.array([10, 10, 20, 20])))
        self.assertTrue(
            np.allclose(results['gt_bboxes'][1], np.array([5, 20, 10, 25])))
        self.assertEqual(results['img'].shape, self.data_info2['img'].shape)

    @mock.patch(
        'mmocr.datasets.transforms.ocr_transforms.np.random.random_sample')
    def _test_polygons(self, mock_sample):
        # test random rotate for polygons
        # returns 1. for random_sample() in _sample_angle(), i.e., angle = 90
        mock_sample.side_effect = [1.]
        transform = RandomRotate(max_angle=90, use_canvas=True)
        results = transform(copy.deepcopy(self.data_info3))
        self.assertTrue(
            np.allclose(results['gt_polygons'][0],
                        np.array([10., 20., 10., 10., 20., 10., 20., 20.])))
        self.assertEqual(results['img'].shape, self.data_info3['img'].shape)

    def test_repr(self):
        transform = RandomRotate(
            max_angle=10,
            pad_with_fixed_color=False,
            pad_value=(0, 0, 0),
            use_canvas=False)
        self.assertEqual(
            repr(transform),
            ('RandomRotate(max_angle = 10, '
             'pad_with_fixed_color = False, pad_value = (0, 0, 0), '
             'use_canvas = False)'))


class TestResize(unittest.TestCase):

    def test_resize_wo_img(self):
        # keep_ratio = True
        dummy_result = dict(img_shape=(10, 20))
        resize = Resize(scale=(40, 30), keep_ratio=True)
        result = resize(dummy_result)
        self.assertEqual(result['img_shape'], (20, 40))
        self.assertEqual(result['scale'], (40, 20))
        self.assertEqual(result['scale_factor'], (2., 2.))
        self.assertEqual(result['keep_ratio'], True)

        # keep_ratio = False
        dummy_result = dict(img_shape=(10, 20))
        resize = Resize(scale=(40, 30), keep_ratio=False)
        result = resize(dummy_result)
        self.assertEqual(result['img_shape'], (30, 40))
        self.assertEqual(result['scale'], (40, 30))
        self.assertEqual(result['scale_factor'], (
            2.,
            3.,
        ))
        self.assertEqual(result['keep_ratio'], False)

    def test_resize_bbox(self):
        # keep_ratio = True
        dummy_result = dict(
            img_shape=(10, 20),
            gt_bboxes=np.array([[0, 0, 1, 1]], dtype=np.float32))
        resize = Resize(scale=(40, 30))
        result = resize(dummy_result)
        self.assertEqual(result['gt_bboxes'].dtype, np.float32)


class TestFixInvalidPolygon(unittest.TestCase):

    def setUp(self):
        # All polygons are invalid w/o gt_bboxes
        self.data_info = dict(
            img=np.random.random((30, 40, 3)),
            gt_polygons=[
                np.array([0., 0., 10., 10., 10., 0., 0., 10.]),
                np.array([0., 0., 10., 0., 0., 10., 5., 10.]),
                np.array([0, 10]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
            ],
            gt_ignored=np.array([False, False, False, False], dtype=bool))
        # All polygons are invalid with gt_bboxes
        # the third one can be recovered from gt_bboxes
        # the fourth one has no valid polygon and bbox
        self.data_info2 = dict(
            img=np.random.random((30, 40, 3)),
            gt_polygons=[
                np.array([0., 0., 10., 10., 10., 0.]),
                np.array([0., 0., 10., 0., 0., 10.]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
            ],
            gt_bboxes=np.array([[0., 0., 10., 10.], [0., 0., 10., 10.],
                                [0, 0, 10, 10], [0, 0, 0, 0]]),
            gt_ignored=np.array([False, False, False, False], dtype=bool))
        # Contains all unfixable polygons
        self.data_info3 = dict(
            img=np.random.random((30, 40, 3)),
            gt_polygons=[
                np.array([0, 10]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
            ],
            gt_ignored=np.array([False, False], dtype=bool))
        # The first one is valid, and the second one is invalid
        self.data_info4 = dict(
            img=np.random.random((30, 40, 3)),
            gt_polygons=[
                np.array([0., 0., 10., 0., 10., 10., 0., 10.]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
            ],
            gt_ignored=np.array([False, False], dtype=bool))
        # no gt_polygons
        self.data_info5 = dict(
            img=np.random.random((30, 40, 3)),
            gt_bboxes=np.array([[0., 0., 10., 10.], [0., 0., 10., 10.],
                                [0, 0, 10, 10], [0, 0, 0, 0]]),
            gt_ignored=np.array([False, False, False, False], dtype=bool))

    def test_transform_fix(self):
        transform = FixInvalidPolygon(mode='fix', min_poly_points=4)
        results = transform(copy.deepcopy(self.data_info))
        # The third one is removed because it doesn't have enough points
        # The fourth one is removed because it is a line
        assert len(
            results['gt_polygons']) == len(self.data_info['gt_polygons']) - 2
        for poly in results['gt_polygons']:
            self.assertTrue(poly2shapely(poly).is_valid)
        results = transform(copy.deepcopy(self.data_info2))
        # The fourth one is removed because it is a line, and its bbox is also
        # invalid
        assert len(
            results['gt_polygons']) == len(self.data_info['gt_polygons']) - 1
        for poly in results['gt_polygons']:
            self.assertTrue(len(poly) >= 8 and len(poly) % 2 == 0)
        # test not fixing the invalid polygons from bboxes
        transform_wo_bbox = FixInvalidPolygon(
            mode='fix', min_poly_points=4, fix_from_bbox=False)
        results = transform_wo_bbox(copy.deepcopy(self.data_info2))
        # The fourth one is removed because it is a line, and its bbox is also
        # invalid
        assert len(
            results['gt_polygons']) == len(self.data_info['gt_polygons']) - 2
        for poly in results['gt_polygons']:
            self.assertTrue(len(poly) >= 8 and len(poly) % 2 == 0)
        # Fixing all invalid polygons would result in an empty result dict,
        # and therefore the transform would return None
        results = transform(copy.deepcopy(self.data_info3))
        self.assertIsNone(results)
        # If no gt_polygon inside
        results = transform(copy.deepcopy(self.data_info5))
        for k, v in results.items():
            self.assertTrue(np.array_equal(v, self.data_info5[k]))

    def test_transform_ignore(self):
        transform = FixInvalidPolygon(mode='ignore')
        results = transform(copy.deepcopy(self.data_info))
        self.assertTrue(
            np.array_equal(results['gt_ignored'],
                           np.array([True, True, True, True], dtype=bool)))
        results = transform(copy.deepcopy(self.data_info4))
        self.assertTrue(
            np.array_equal(results['gt_ignored'],
                           np.array([False, True], dtype=bool)))
        for poly, ignored in zip(results['gt_polygons'],
                                 results['gt_ignored']):
            if not ignored:
                self.assertTrue(poly2shapely(poly).is_valid)
        # If no gt_polygon inside
        results = transform(copy.deepcopy(self.data_info5))
        for k, v in results.items():
            self.assertTrue(np.array_equal(v, self.data_info5[k]))

    def test_repr(self):
        transform = FixInvalidPolygon()
        print(repr(transform))
        self.assertEqual(
            repr(transform),
            'FixInvalidPolygon(mode = "fix", min_poly_points = 4, '
            'fix_from_bbox = True)')


class TestRemoveIgnored(unittest.TestCase):

    def setUp(self):
        self.data_info = dict(
            img=np.random.random((30, 40, 3)),
            gt_polygons=[
                np.array([0., 0., 10., 10., 10., 0.]),
                np.array([0., 0., 10., 0., 0., 10.]),
                np.array([0, 10, 0, 10, 1, 2, 3, 4]),
                np.array([0, 10, 0, 10, 10, 0, 0, 10]),
            ],
            gt_bboxes=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [0, 0, 10, 10],
                                [0, 0, 0, 0]]),
            gt_ignored=np.array([False, True, True, False], dtype=bool),
            gt_texts=['t1', 't2', 't3', 't4'],
            gt_bboxes_labels=np.array([0, 1, 2, 3]))
        self.keys = [
            'gt_polygons', 'gt_bboxes', 'gt_ignored', 'gt_texts',
            'gt_bboxes_labels'
        ]

    def test_transform(self):
        transform = RemoveIgnored()
        results = transform(copy.deepcopy(self.data_info))
        for original_idx, new_idx in enumerate([0, 3]):
            for key in self.keys:
                self.assertTrue(
                    np.array_equal(results[key][original_idx],
                                   self.data_info[key][new_idx]))
