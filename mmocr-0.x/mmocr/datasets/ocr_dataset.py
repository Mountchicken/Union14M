# Copyright (c) OpenMMLab. All rights reserved.
import json
import os

from mmdet.datasets.builder import DATASETS
from mmocr.core.evaluation.ocr_metric import eval_ocr_metric
from mmocr.datasets.base_dataset import BaseDataset
from mmocr.utils import is_type_list


@DATASETS.register_module()
class OCRDataset(BaseDataset):

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['text'] = results['img_info']['text']

    def evaluate(self, results, metric='acc', logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
        Returns:
            dict[str: float]
        """
        assert isinstance(metric, str) or is_type_list(metric, str)

        gt_texts = []
        pred_texts = []
        outs = []
        for i in range(len(self)):
            item_info = self.data_infos[i]
            text = item_info['text']
            gt_texts.append(text)
            pred_texts.append(results[i]['text'])
            outs.append(
                json.dumps(
                    dict(
                        pred=results[i]['text'],
                        target=text,
                        case='lower_case_symbols')))
        counter = 1
        if not os.path.exists(f'RobustScanner_{counter}'):
            with open(f'RobustScanner_{counter}.jsonl', 'w') as f:
                for result in outs:
                    f.write(result + '\n')
        else:
            counter += 1
        eval_results = eval_ocr_metric(pred_texts, gt_texts, metric=metric)

        return eval_results
