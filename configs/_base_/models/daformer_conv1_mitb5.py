# This is the same as SegFormer but with 256 embed_dims
# SegF. with C_e=256 in Tab. 7

# model settings
# 标准化操作，使用Register BN
norm_cfg = dict(type='BN', requires_grad=True)         
# 找到未使用的参数
find_unused_parameters = True
# 模型具体参数
model = dict(
    # Register 编码解码
    type='EncoderDecoder',
    # 预训练权重
    pretrained='pretrained/mit_b5.pth',
    # 主干网络 register mit_b5 
    backbone=dict(type='mit_b5', style='pytorch'),
    # 修改解码头 DAFormerHead
    decode_head=dict(
        type='DAFormerHead',
        # 输入通道数：4层
        in_channels=[64, 128, 320, 512],
        # 通道索引
        in_index=[0, 1, 2, 3],
        # 输出的通道数
        channels=256,
        # 随机丢弃比例
        dropout_ratio=0.1,
        # 数据类数
        num_classes=19,
        # 标准化操作
        norm_cfg=norm_cfg,
        # 角对齐 false
        align_corners=False,
        # 剩下的字典类型参数
        decoder_params=dict(
            # 嵌入向量维度
            embed_dims=256,
            # 嵌入向量配置 mlp 没有激活函数 没有标准化
            embed_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            # 嵌入向量颈部配置 mlp 
            embed_neck_cfg=dict(type='mlp', act_cfg=None, norm_cfg=None),
            # 融合策略
            fusion_cfg=dict(
                # 卷积
                type='conv',
                # 卷积核大小
                kernel_size=1,
                # 激活函数 relu
                act_cfg=dict(type='ReLU'),
                # 标准化
                norm_cfg=norm_cfg),
        ),
        # 解码器损失函数 CrossEntropyLoss 不使用sigmoid 权重 1
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    # ？？？
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
