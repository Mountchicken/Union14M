# optimizer
optimizer = dict(type='Adadelta', lr=0.5)
optimizer_config = dict(grad_clip=dict(max_norm=0.5))
# learning policy
lr_config = dict(policy='step', step=[8, 14, 16])
# running settings
runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(interval=1)
