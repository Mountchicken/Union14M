_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_adam_step_20e.py',
    '../../_base_/recog_pipelines/abinet_pipeline.py',
    '../../_base_/recog_models/abinet.py',
    '../../_base_/recog_datasets/Union14M_train.py',
    '../../_base_/recog_datasets/Union14M_benchmark.py'
]

train_list = {{_base_.train_list}}
test_list = {{_base_.test_list}}

train_pipeline = {{_base_.train_pipeline}}
test_pipeline = {{_base_.test_pipeline}}

num_chars = 93
max_seq_len = 26

label_convertor = dict(
    type='ABIConvertor',
    dict_type='DICT91',
    with_unknown=True,
    with_padding=False,
    lower=False,
)

data = dict(
    samples_per_gpu=64,
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

evaluation = dict(interval=20, metric='acc')
