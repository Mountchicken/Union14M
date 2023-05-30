import json
from typing import List, Dict
from tqdm import tqdm

def mapper(axis_list: List[Dict], rotate_list: List[Dict]):
    """Map the filename in axis_list to the filename in rotate_list according
    to the annotation

    Args:
        axis_list (List[Dict]): List of axis annotation
        rotate_list (List[Dict]): List of rotate annotation
    """
    mapping_list = []
    for axis in tqdm(axis_list):
        axis_fn = axis['filename']
        axis_text = axis['text']
        find = False
        for rotate in rotate_list:
            rotate_fn = rotate['filename']
            rotate_text = rotate['text']
            if axis_text == rotate_text:
                mapping_list.append({
                    'axis': axis_fn,
                    'rotate': rotate_fn,
                    'text': axis_text
                })

                # remove the rotate from the list
                rotate_list.remove(rotate)
                find = True
                break
            else:
                continue
        if not find:
            print(f'fail to find the rotate for {axis_fn}')
            exit(0)
    with open('mapping.jsonl', 'w') as f:
        for mapping in mapping_list:
            f.write(json.dumps(mapping) + '\n')


if __name__ == '__main__':
    axis_root = '/media/jiangqing/jqssd/projects/research/DGDataset/Union14M/data/Union14M-L/full_images/TextOCR/annotation.jsonl'
    with open(axis_root, 'r') as f:
        axis_list = [json.loads(line) for line in f.readlines()]
    rotate_root = '/media/jiangqing/jqssd/projects/research/DGDataset/Union14M/add_data/textocr/rotate_crop/train_label.txt'
    with open(rotate_root, 'r') as f:
        rotate_list = [json.loads(line) for line in f.readlines()]
    mapper(axis_list, rotate_list)
