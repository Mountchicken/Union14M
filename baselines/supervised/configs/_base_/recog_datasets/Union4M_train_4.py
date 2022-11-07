train_root = '../Union4M/Union4M'

train_img_prefix1 = f'{train_root}/training_sets'
train_ann_file1 = f'{train_root}/training_sets/hell_4.jsonl'

hell = dict(
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

train_img_prefix2 = f'{train_root}/training_sets'
train_ann_file2 = f'{train_root}/training_sets/difficult_4.jsonl'

difficult = dict(
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

train_img_prefix3 = f'{train_root}/training_sets'
train_ann_file3 = f'{train_root}/training_sets/hard_4.jsonl'

hard = dict(
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

train_img_prefix4 = f'{train_root}/training_sets'
train_ann_file4 = f'{train_root}/training_sets/medium_4.jsonl'

medium = dict(
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

train_img_prefix5 = f'{train_root}/training_sets'
train_ann_file5 = f'{train_root}/training_sets/simple_4.jsonl'

simple = dict(
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

train_list = [hell, difficult, hard, medium, simple]
