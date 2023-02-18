# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings

import lmdb
import mmcv
import numpy as np
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.loading import LoadAnnotations, LoadImageFromFile


@PIPELINES.register_module()
class LoadTextAnnotations(LoadAnnotations):
    """Load annotations for text detection.

    Args:
        with_bbox (bool): Whether to parse and load the bbox annotation.
             Default: True.
        with_label (bool): Whether to parse and load the label annotation.
            Default: True.
        with_mask (bool): Whether to parse and load the mask annotation.
             Default: False.
        with_seg (bool): Whether to parse and load the semantic segmentation
            annotation. Default: False.
        poly2mask (bool): Whether to convert the instance masks from polygons
            to bitmaps. Default: True.
        use_img_shape (bool): Use the shape of loaded image from
            previous pipeline ``LoadImageFromFile`` to generate mask.
    """

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 with_seg=False,
                 poly2mask=True,
                 use_img_shape=False):
        super().__init__(
            with_bbox=with_bbox,
            with_label=with_label,
            with_mask=with_mask,
            with_seg=with_seg,
            poly2mask=poly2mask)

        self.use_img_shape = use_img_shape

    def process_polygons(self, polygons):
        """Convert polygons to list of ndarray and filter invalid polygons.

        Args:
            polygons (list[list]): Polygons of one instance.

        Returns:
            list[numpy.ndarray]: Processed polygons.
        """

        polygons = [np.array(p).astype(np.float32) for p in polygons]
        valid_polygons = []
        for polygon in polygons:
            if len(polygon) % 2 == 0 and len(polygon) >= 6:
                valid_polygons.append(polygon)
        return valid_polygons

    def _load_masks(self, results):
        ann_info = results['ann_info']
        h, w = results['img_info']['height'], results['img_info']['width']
        if self.use_img_shape:
            if results.get('ori_shape', None):
                h, w = results['ori_shape'][:2]
                results['img_info']['height'] = h
                results['img_info']['width'] = w
            else:
                warnings.warn('"ori_shape" not in results, use the shape '
                              'in "img_info" instead.')
        gt_masks = ann_info['masks']
        if self.poly2mask:
            gt_masks = BitmapMasks(
                [self._poly2mask(mask, h, w) for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks(
                [self.process_polygons(polygons) for polygons in gt_masks], h,
                w)
        gt_masks_ignore = ann_info.get('masks_ignore', None)
        if gt_masks_ignore is not None:
            if self.poly2mask:
                gt_masks_ignore = BitmapMasks(
                    [self._poly2mask(mask, h, w) for mask in gt_masks_ignore],
                    h, w)
            else:
                gt_masks_ignore = PolygonMasks([
                    self.process_polygons(polygons)
                    for polygons in gt_masks_ignore
                ], h, w)
            results['gt_masks_ignore'] = gt_masks_ignore
            results['mask_fields'].append('gt_masks_ignore')

        results['gt_masks'] = gt_masks
        results['mask_fields'].append('gt_masks')
        return results


@PIPELINES.register_module()
class LoadImageFromNdarray(LoadImageFromFile):
    """Load an image from np.ndarray.

    Similar with :obj:`LoadImageFromFile`, but the image read from
    ``results['img']``, which is np.ndarray.
    """

    def __call__(self, results):
        """Call functions to add image meta information.

        Args:
            results (dict): Result dict with Webcam read image in
                ``results['img']``.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        assert results['img'].dtype == 'uint8'

        img = results['img']
        if self.color_type == 'grayscale' and img.shape[2] == 3:
            img = mmcv.bgr2gray(img, keepdim=True)
        if self.color_type == 'color' and img.shape[2] == 1:
            img = mmcv.gray2bgr(img)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['img_fields'] = ['img']
        return results


@PIPELINES.register_module()
class LoadImageFromLMDB(object):
    """Load an image from lmdb file.

    Similar with :obj:'LoadImageFromFile', but the image read from
    "results['img_info']['filename']", which is a data index of lmdb file.
    """

    def __init__(self, color_type='color'):
        self.color_type = color_type
        self.env = None
        self.txn = None

    def __call__(self, results):
        img_key = results['img_info']['filename']
        lmdb_path = results['img_prefix']

        # lmdb env
        if self.env is None:
            self.env = lmdb.open(
                lmdb_path,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
            )
        # read image
        with self.env.begin(write=False) as txn:
            imgbuf = txn.get(img_key.encode('utf-8'))
            try:
                img = mmcv.imfrombytes(imgbuf, flag=self.color_type)
            except IOError:
                print('Corrupted image for {}'.format(img_key))
                return None

            results['filename'] = img_key
            results['ori_filename'] = img_key
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            results['img_fields'] = ['img']
            return results

    def __repr__(self):
        return '{} (color_type={})'.format(self.__class__.__name__,
                                           self.color_type)

    def __del__(self):
        if self.env is not None:
            self.env.close()


@PIPELINES.register_module()
class LoadDualImageFromFile(object):
    """Load dual image from file. Tipically used for image to image tasks.

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): The flag argument for :func:`mmcv.imfrombytes`.
            Defaults to 'color'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='bgr',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.
                In dual image case, the dict contains `ori_prefix` and
                `tgt_prefix` keys. Original image and target image should
                have the same filename, but in different folders.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filename_ori = osp.join(results['ori_prefix'],
                                results['img_info']['filename'])
        filename_tgt = osp.join(results['tgt_prefix'],
                                results['img_info']['filename'])

        img_bytes_ori = self.file_client.get(filename_ori)
        img_bytes_tgt = self.file_client.get(filename_tgt)
        img_ori = mmcv.imfrombytes(
            img_bytes_ori,
            flag=self.color_type,
            channel_order=self.channel_order)
        img_tgt = mmcv.imfrombytes(
            img_bytes_tgt,
            flag=self.color_type,
            channel_order=self.channel_order)
        if self.to_float32:
            img_ori = img_ori.astype(np.float32)
            img_tgt = img_tgt.astype(np.float32)

        results['filename_ori'] = filename_ori
        results['filename_tgt'] = filename_tgt
        results['ori_filename'] = results['img_info']['filename']
        results['img_ori'] = img_ori
        results['img_tgt'] = img_tgt
        results['img_shape'] = img_ori.shape
        results['ori_shape'] = img_ori.shape
        results['img_fields'] = ['img_ori', 'img_tgt']
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"channel_order='{self.channel_order}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str
