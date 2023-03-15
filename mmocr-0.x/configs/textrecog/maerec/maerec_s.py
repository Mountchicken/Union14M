_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_datasets/Union14M_train.py',
    '../../_base_/recog_datasets/Union14M_benchmark.py',
    '../../_base_/schedules/schedule_adamw_cos_10e.py',
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

label_convertor = dict(
    type='AttnConvertor', dict_type='DICT91', with_unknown=True)

model = dict(
    type='MAERec',
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        pretrained='pretrained/vit_small/epoch_19.pth'),
    decoder=dict(
        type='MAERecDecoder',
        n_layers=6,
        d_embedding=384,
        n_head=8,
        d_model=384,
        d_inner=384 * 4,
        d_k=384 // 8,
        d_v=384 // 8),
    loss=dict(type='TFLoss'),
    label_convertor=label_convertor,
    max_seq_len=32)


data = dict(
    samples_per_gpu=256,
    workers_per_gpu=8,
    val_dataloader=dict(samples_per_gpu=128),
    test_dataloader=dict(samples_per_gpu=128),
    train=dict(
        type='UniformConcatDataset',
        datasets=train_list,
        pipeline=train_pipeline),
    val=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline),
    test=dict(
        type='UniformConcatDataset',
        datasets=test_list,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='acc')
