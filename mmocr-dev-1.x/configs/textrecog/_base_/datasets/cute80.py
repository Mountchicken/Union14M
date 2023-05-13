cute80_textrecog_data_root = '../data/common_benchmarks/CUTE80'

cute80_textrecog_test = dict(
    type='OCRDataset',
    data_root=cute80_textrecog_data_root,
    ann_file='annotation.json',
    test_mode=True,
    pipeline=None)
