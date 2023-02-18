# Text Recognition Training set, including:
# Synthetic Datasets: SynthText, Syn90k

train_root = '../scene_text_train'

train_img_prefix1 = f'{train_root}/MJ'
train_ann_file1 = f'{train_root}/MJ'

train1 = dict(
    type='OCRDataset',
    img_prefix=train_img_prefix1,
    ann_file=train_ann_file1,
    loader=dict(
        type='AnnFileLoader',
        repeat=1,
        file_format='lmdb',
        parser=dict(type='LineJsonParser', keys=['filename', 'text'])),
    pipeline=None,
    test_mode=False)

train_img_prefix2 = f'{train_root}/ST'
train_ann_file2 = f'{train_root}/ST'

train2 = {key: value for key, value in train1.items()}
train2['img_prefix'] = train_img_prefix2
train2['ann_file'] = train_ann_file2

train_list = [train1, train2]
