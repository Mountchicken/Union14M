test_root8 = '../Union4M/Union4M/test_sets_TS/honest_ori'

test_img_prefix8 = f'{test_root8}/'
test_ann_file8 = f'{test_root8}/annotation.jsonl'

test8 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix8,
    ann_file=test_ann_file8,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=True)

test_list=[test8]
