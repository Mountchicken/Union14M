# Text Recognition Testing set, including:
# Regular Datasets: IIIT5K, SVT, IC13
# Irregular Datasets: IC15, SVTP, CT80

test_root = '../../benchmarks'

test_img_prefix1 = f'{test_root}/IIIT5K/'
test_img_prefix2 = f'{test_root}/IC13/'
test_img_prefix3 = f'{test_root}/IC15/'
test_img_prefix4 = f'{test_root}/SVT/'
test_img_prefix5 = f'{test_root}/SVTP/'
test_img_prefix6 = f'{test_root}/CUTE80/'

test_ann_file1 = f'{test_root}/IIIT5K/annotation.jsonl'
test_ann_file2 = f'{test_root}/IC13/annotation.jsonl'
test_ann_file3 = f'{test_root}/IC15/annotation.jsonl'
test_ann_file4 = f'{test_root}/SVT/annotation.jsonl'
test_ann_file5 = f'{test_root}/SVTP/annotation.jsonl'
test_ann_file6 = f'{test_root}/CUTE80/annotation.jsonl'

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

test_list = [test1, test2, test3, test4, test5, test6]
