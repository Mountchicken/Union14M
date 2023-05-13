svt_textrecog_data_root = '../data/common_benchmarks/SVT'

svt_textrecog_train = dict(
    type='OCRDataset',
    data_root=svt_textrecog_data_root,
    ann_file='textrecog_train.json',
    pipeline=None)

svt_textrecog_test = dict(
    type='OCRDataset',
    data_root=svt_textrecog_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
