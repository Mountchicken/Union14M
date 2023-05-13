# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import List, Optional, Tuple, Union


class BaseGatherer:
    """Base class for gatherer.

    Note: Gatherer assumes that all the annotation file is in the same
    directory and all the image files are in the same directory.

    Args:
        img_dir(str): The directory of the images. It is usually set
            automatically to f'text{task}_imgs/split' and users do not need to
            set it manually in config file in most cases. When the image files
            is not in 'text{task}_imgs/split' directory, users should set it.
            Defaults to ''.
        ann_dir (str): The directory of the annotation files. It is usually set
            automatically to 'annotations' and users do not need to set it
            manually in config file in most cases. When the annotation files
            is not in 'annotations' directory, users should set it. Defaults to
            'annotations'.
        split (str, optional): List of splits to gather. It' s the partition of
            the datasets. Options are 'train', 'val' or 'test'. It is usually
            set automatically and users do not need to set it manually in
            config file in most cases. Defaults to None.
        data_root (str, optional): The root directory of the image and
            annotation. It is usually set automatically and users do not need
            to set it manually in config file in most cases. Defaults to None.
    """

    def __init__(self,
                 img_dir: str = '',
                 ann_dir: str = 'annotations',
                 split: Optional[str] = None,
                 data_root: Optional[str] = None) -> None:
        self.split = split
        self.data_root = data_root
        self.ann_dir = osp.join(data_root, ann_dir)
        self.img_dir = osp.join(data_root, img_dir)

    def __call__(self) -> Union[Tuple[List[str], List[str]], Tuple[str, str]]:
        """The return value of the gatherer is a tuple of two lists or strings.

        The first element is the list of image paths or the directory of the
        images. The second element is the list of annotation paths or the path
        of the annotation file which contains all the annotations.
        """
        raise NotImplementedError
