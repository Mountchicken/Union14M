eval_root = 'Union14M/eval'

train_img_prefix1 = eval_root
train_ann_file1 = f'{eval_root}/annotation.jsonl'

eval = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

eval_list = [eval]
