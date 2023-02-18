test_root = 'Union14M/benchmark/'

test_img_prefix1 = f'{test_root}/artistic'
test_img_prefix2 = f'{test_root}/curve'
test_img_prefix3 = f'{test_root}/salient'
test_img_prefix4 = f'{test_root}/contextless'
test_img_prefix5 = f'{test_root}/multi_words'
test_img_prefix6 = f'{test_root}/multi_oriented'
test_img_prefix7 = f'{test_root}/incomplete'
test_img_prefix8 = f'{test_root}/incomplete_ori'
test_img_prefix9 = f'{test_root}/general'

test_ann_file1 = f'{test_root}/art/annotation.jsonl'
test_ann_file2 = f'{test_root}/curve/annotation.jsonl'
test_ann_file3 = f'{test_root}/fcos/annotation.jsonl'
test_ann_file4 = f'{test_root}/meanless/annotation.jsonl'
test_ann_file5 = f'{test_root}/multi/annotation.jsonl'
test_ann_file6 = f'{test_root}/multi_oriented/annotation.jsonl'
test_ann_file7 = f'{test_root}/incomplete/annotation.jsonl'
test_ann_file8 = f'{test_root}/incomplete_ori/annotation.jsonl'
test_ann_file9 = f'{test_root}/general/annotation.jsonl'

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

test3 = {key: value for key, value in test1.items()}
test3['img_prefix'] = test_img_prefix3
test3['ann_file'] = test_ann_file3

test4 = {key: value for key, value in test1.items()}
test4['img_prefix'] = test_img_prefix4
test4['ann_file'] = test_ann_file4

test5 = {key: value for key, value in test1.items()}
test5['img_prefix'] = test_img_prefix5
test5['ann_file'] = test_ann_file5

test6 = {key: value for key, value in test1.items()}
test6['img_prefix'] = test_img_prefix6
test6['ann_file'] = test_ann_file6

test7 = {key: value for key, value in test1.items()}
test7['img_prefix'] = test_img_prefix7
test7['ann_file'] = test_ann_file7

test8 = {key: value for key, value in test1.items()}
test8['img_prefix'] = test_img_prefix7
test8['ann_file'] = test_ann_file7

test9 = {key: value for key, value in test1.items()}
test9['img_prefix'] = test_img_prefix7
test9['ann_file'] = test_ann_file7

test_list = [test1, test2, test3, test4, test5, test6, test7, test8, test9]
