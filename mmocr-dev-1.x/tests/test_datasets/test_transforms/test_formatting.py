# Copyright (c) OpenMMLab. All rights reserved.
import copy
from unittest import TestCase

import numpy as np
import torch

from mmocr.datasets.transforms import (PackKIEInputs, PackTextDetInputs,
                                       PackTextRecogInputs)


class TestPackTextDetInputs(TestCase):

    def test_packdetinput(self):
        datainfo = dict(
            img=np.random.random((10, 10)),
            img_shape=(10, 10),
            ori_shape=(10, 10),
            pad_shape=(10, 10),
            scale_factor=(1, 1),
            img_path='tmp/tmp.jpg',
            flip=True,
            flip_direction='left',
            gt_bboxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]],
                               dtype=np.float32),
            gt_bboxes_labels=np.array([0, 0], np.int64),
            gt_polygons=[
                np.array([0, 0, 0, 10, 10, 10, 10, 0], dtype=np.float32),
                np.array([5, 5, 5, 15, 15, 15, 15, 5], dtype=np.float32)
            ],
            gt_texts=['mmocr', 'mmocr_ignore'],
            gt_ignored=np.bool_([False, True]))
        with self.assertRaises(KeyError):
            transform = PackTextDetInputs(meta_keys=('tmp', ))
            transform(copy.deepcopy(datainfo))
        transform = PackTextDetInputs()
        results = transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertEqual(results['inputs'].shape, torch.Size([1, 10, 10]))
        self.assertTupleEqual(tuple(results['inputs'].shape), (1, 10, 10))
        self.assertIn('data_samples', results)

        # test non-contiugous img
        nc_datainfo = copy.deepcopy(datainfo)
        nc_datainfo['img'] = nc_datainfo['img'].transpose(1, 0)
        results = transform(nc_datainfo)
        self.assertIn('inputs', results)
        self.assertEqual(results['inputs'].shape, torch.Size([1, 10, 10]))

        data_sample = results['data_samples']
        self.assertIn('bboxes', data_sample.gt_instances)
        self.assertIsInstance(data_sample.gt_instances.bboxes, torch.Tensor)
        self.assertEqual(data_sample.gt_instances.bboxes.dtype, torch.float32)
        self.assertIsInstance(data_sample.gt_instances.polygons[0], np.ndarray)
        self.assertEqual(data_sample.gt_instances.polygons[0].dtype,
                         np.float32)
        self.assertEqual(data_sample.gt_instances.ignored.dtype, torch.bool)
        self.assertEqual(data_sample.gt_instances.labels.dtype, torch.int64)
        self.assertIsInstance(data_sample.gt_instances.texts, list)

        self.assertIn('img_path', data_sample)
        self.assertIn('flip', data_sample)

        transform = PackTextDetInputs(meta_keys=('img_path', ))
        results = transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertIn('data_samples', results)

        data_sample = results['data_samples']
        self.assertIn('bboxes', data_sample.gt_instances)
        self.assertIn('img_path', data_sample)
        self.assertNotIn('flip', data_sample)

        datainfo.pop('gt_texts')
        transform = PackTextDetInputs()
        results = transform(copy.deepcopy(datainfo))
        data_sample = results['data_samples']
        self.assertNotIn('texts', data_sample.gt_instances)

        datainfo = dict(img_shape=(10, 10))
        transform = PackTextDetInputs(meta_keys=('img_shape', ))
        results = transform(copy.deepcopy(datainfo))
        self.assertNotIn('inputs', results)
        data_sample = results['data_samples']
        self.assertNotIn('texts', data_sample.gt_instances)

    def test_repr(self):
        transform = PackTextDetInputs()
        self.assertEqual(
            repr(transform),
            ("PackTextDetInputs(meta_keys=('img_path', 'ori_shape', "
             "'img_shape', 'scale_factor', 'flip', 'flip_direction'))"))


class TestPackTextRecogInputs(TestCase):

    def test_packrecogtinput(self):
        datainfo = dict(
            img=np.random.random((10, 10)),
            img_shape=(10, 10),
            ori_shape=(10, 10),
            pad_shape=(10, 10),
            scale_factor=(1, 1),
            img_path='tmp/tmp.jpg',
            flip=True,
            flip_direction='left',
            gt_bboxes=np.array([[0, 0, 10, 10]]),
            gt_labels=np.array([0]),
            gt_polygons=[[0, 0, 0, 10, 10, 10, 10, 0]],
            gt_texts=['mmocr'],
        )
        with self.assertRaises(KeyError):
            transform = PackTextRecogInputs(meta_keys=('tmp', ))
            transform(copy.deepcopy(datainfo))
        transform = PackTextRecogInputs()
        results = transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertTupleEqual(tuple(results['inputs'].shape), (1, 10, 10))
        self.assertIn('data_samples', results)
        data_sample = results['data_samples']
        self.assertEqual(data_sample.gt_text.item, 'mmocr')
        self.assertIn('img_path', data_sample)
        self.assertIn('valid_ratio', data_sample)
        self.assertIn('pad_shape', data_sample)

        # test non-contiugous img
        nc_datainfo = copy.deepcopy(datainfo)
        nc_datainfo['img'] = nc_datainfo['img'].transpose(1, 0)
        results = transform(nc_datainfo)
        self.assertIn('inputs', results)
        self.assertEqual(results['inputs'].shape, torch.Size([1, 10, 10]))

        transform = PackTextRecogInputs(meta_keys=('img_path', ))
        results = transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertIn('data_samples', results)
        data_sample = results['data_samples']
        self.assertEqual(data_sample.gt_text.item, 'mmocr')
        self.assertIn('img_path', data_sample)
        self.assertNotIn('valid_ratio', data_sample)
        self.assertNotIn('pad_shape', data_sample)

        datainfo = dict(img_shape=(10, 10))
        transform = PackTextRecogInputs(meta_keys=('img_shape', ))
        results = transform(copy.deepcopy(datainfo))
        self.assertNotIn('inputs', results)
        data_sample = results['data_samples']
        self.assertNotIn('item', data_sample.gt_text)

    def test_repr(self):
        transform = PackTextRecogInputs()
        self.assertEqual(
            repr(transform),
            ("PackTextRecogInputs(meta_keys=('img_path', 'ori_shape', "
             "'img_shape', 'pad_shape', 'valid_ratio'))"))


class TestPackKIEInputs(TestCase):

    def setUp(self) -> None:
        self.transform = PackKIEInputs()

    def test_transform(self):
        datainfo = dict(
            img=np.random.random((10, 10)),
            img_shape=(10, 10),
            ori_shape=(10, 10),
            scale_factor=(1, 1),
            img_path='tmp/tmp.jpg',
            gt_bboxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]],
                               dtype=np.float32),
            gt_bboxes_labels=np.array([0, 0], np.int64),
            gt_edges_labels=np.array([[0, 0], [0, 0]], np.int64),
            gt_texts=['text1', 'text2'])

        with self.assertRaises(KeyError):
            transform = PackKIEInputs(meta_keys=('tmp', ))
            transform(copy.deepcopy(datainfo))

        results = self.transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertTupleEqual(tuple(results['inputs'].shape), (1, 10, 10))
        self.assertIn('data_samples', results)
        data_sample = results['data_samples']
        self.assertIsInstance(data_sample.gt_instances.bboxes, torch.Tensor)
        self.assertEqual(data_sample.gt_instances.bboxes.dtype, torch.float32)
        self.assertEqual(data_sample.gt_instances.labels.dtype, torch.int64)
        self.assertEqual(data_sample.gt_instances.edge_labels.dtype,
                         torch.int64)
        self.assertIsInstance(data_sample.gt_instances.texts, list)

        # test non-contiugous img
        nc_datainfo = copy.deepcopy(datainfo)
        nc_datainfo['img'] = nc_datainfo['img'].transpose(1, 0)
        results = self.transform(nc_datainfo)
        self.assertIn('inputs', results)
        self.assertEqual(results['inputs'].shape, torch.Size([1, 10, 10]))

        transform = PackKIEInputs(meta_keys=('img_path', ))
        results = transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertIn('data_samples', results)

        data_sample = results['data_samples']
        self.assertIn('bboxes', data_sample.gt_instances)
        self.assertIn('img_path', data_sample)

        datainfo.pop('img')
        results = self.transform(copy.deepcopy(datainfo))
        self.assertIn('inputs', results)
        self.assertEqual(results['inputs'].shape, torch.Size((0, 0, 0)))

    def test_repr(self):
        self.assertEqual(repr(self.transform), ('PackKIEInputs(meta_keys=())'))
