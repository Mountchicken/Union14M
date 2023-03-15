train_root = 'data/Union14M-L/'

challenging = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/train_challenging.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

hard = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/train_hard.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

medium = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/train_medium.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

normal = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/val_annos.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

easy = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/train_easy.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

val = dict(
    type='OCRDataset',
    img_prefix=train_root,
    ann_file=f'{train_root}/train_easy.jsonl',
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_list = [challenging, hard, medium, normal, easy]
val_list = [val]
