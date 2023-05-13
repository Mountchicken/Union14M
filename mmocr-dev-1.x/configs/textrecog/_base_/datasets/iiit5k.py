iiit5k_textrecog_data_root = '../data/common_benchmarks/IIIT5K'

iiit5k_textrecog_train = dict(
    type='OCRDataset',
    data_root=iiit5k_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

iiit5k_textrecog_test = dict(
    type='OCRDataset',
    data_root=iiit5k_textrecog_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
