import mmcv
from mmdet.datasets.builder import PIPELINES


@PIPELINES.register_module()
class DualResize:
    """Resize dual images."""

    def __init__(self, size, backend='cv2'):
        self.size = size
        self.backend = backend

    def __call__(self, results):
        img_ori = results['img_ori']
        ori_shape = img_ori.shape
        img_tgt = results['img_tgt']
        if self.backend == 'cv2':
            img_ori = mmcv.imresize(img_ori, self.size)
            img_tgt = mmcv.imresize(img_tgt, self.size)
        elif self.backend == 'pillow':
            img_ori = mmcv.pil_resize(img_ori, self.size)
            img_tgt = mmcv.pil_resize(img_tgt, self.size)
        else:
            raise ValueError('Invalid backend: {}'.format(self.backend))
        results['img_ori'] = img_ori
        results['img_tgt'] = img_tgt
        results['img_shape'] = img_ori.shape
        results['ori_shape'] = ori_shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, backend={self.backend})'
        return repr_str


@PIPELINES.register_module()
class DualNormalize:
    """Normalize dual images."""

    def __init__(self, mean, std, to_rgb=True):
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb

    def __call__(self, results):
        img_ori = results['img_ori']
        img_tgt = results['img_tgt']
        results['img_ori'] = mmcv.imnormalize(img_ori, self.mean, self.std,
                                              self.to_rgb)
        results['img_tgt'] = mmcv.imnormalize(img_tgt, self.mean, self.std,
                                              self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str
