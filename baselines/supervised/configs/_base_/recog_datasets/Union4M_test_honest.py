test_root = '../Union4M/Union4M/test_sets_TS'

test_img_prefix1 = f'{test_root}/honest'
test_img_prefix2 = f'{test_root}/honest_ori'

test_ann_file1 = f'{test_root}/honest/annotation.jsonl'
test_ann_file2 = f'{test_root}/honest_ori/annotation.jsonl'

test1 = dict(
    type='OCRDataset',
    img_prefix=test_img_prefix1,
    ann_file=test_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='txt',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=True)

test2 = {key: value for key, value in test1.items()}
test2['img_prefix'] = test_img_prefix2
test2['ann_file'] = test_ann_file2

test_list = [test1, test2]
