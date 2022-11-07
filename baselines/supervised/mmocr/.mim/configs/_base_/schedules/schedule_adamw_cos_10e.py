# optimizer
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr_ratio=0.0001, by_epoch=False)
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=10)
checkpoint_config = dict(interval=1)
