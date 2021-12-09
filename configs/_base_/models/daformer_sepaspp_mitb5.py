# DAFormer (with context-aware feature fusion) in Tab. 7

_base_ = ['daformer_conv1_mitb5.py']
# 将 daformer_conv1_mitb5 里解码器的解码参数里的mlp换成了深度可分离的aspp
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    decode_head=dict(
        decoder_params=dict(
            fusion_cfg=dict(
                _delete_=True,
                type='aspp',
                sep=True,
                dilations=(1, 6, 12, 18),
                pool=False,
                act_cfg=dict(type='ReLU'),
                norm_cfg=norm_cfg))))
