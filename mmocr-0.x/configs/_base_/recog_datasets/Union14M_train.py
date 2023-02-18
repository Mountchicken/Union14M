train_root = '../Union4M/Union4M/training_sets'

train_img_prefix1 = train_root
train_ann_file1 = f'{train_root}/hell.jsonl'

challenging = dict(
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

train_img_prefix2 = train_root
train_ann_file2 = f'{train_root}/hard.jsonl'

hard = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix2,
    ann_file=train_ann_file2,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_img_prefix3 = train_root
train_ann_file3 = f'{train_root}/difficult.jsonl'

medium = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix3,
    ann_file=train_ann_file3,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_img_prefix4 = train_root
train_ann_file4 = f'{train_root}/medium.jsonl'

normal = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix4,
    ann_file=train_ann_file4,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_img_prefix5 = train_root
train_ann_file5 = f'{train_root}/simple.jsonl'

easy = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix5,
    ann_file=train_ann_file5,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_list = [challenging, hard, medium, normal, easy]
