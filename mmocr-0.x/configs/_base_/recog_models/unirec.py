label_convertor = dict(
    type='AttnConvertor', dict_type='DICT91', with_unknown=True)

model = dict(
    type='UniRec',
    backbone=dict(
        type='VisionTransformer',
        img_size=(32, 128),
        patch_size=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.,
        qkv_bias=True,
        pretrained=None),
    decoder=dict(
        type='UniRecDecoder',
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
