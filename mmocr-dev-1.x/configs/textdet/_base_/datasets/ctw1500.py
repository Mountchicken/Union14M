ctw1500_textdet_data_root = 'data/ctw1500'

ctw1500_textdet_train = dict(
    type='OCRDataset',
    data_root=ctw1500_textdet_data_root,
    ann_file='textdet_train.json',
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=None)

ctw1500_textdet_test = dict(
    type='OCRDataset',
    data_root=ctw1500_textdet_data_root,
    ann_file='textdet_test.json',
    test_mode=True,
    pipeline=None)
