# dataset settings
# 数据集的设置
# 数据集类型
dataset_type = 'CityscapesDataset'
# 数据集根路径
data_root = 'data/cityscapes/'
# 图片归一化cfg：均值方差、是否转成rgb图片
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# 裁剪大小：512
crop_size = (512, 512)
# gta训练管道：transforms, formating
gta_train_pipeline = [
    # 载入图片
    dict(type='LoadImageFromFile'),
    # 载入标注
    dict(type='LoadAnnotations'),
    # resize大小为1280，720
    dict(type='Resize', img_scale=(1280, 720)),
    # 随机裁剪 裁剪预期的大小， 单一类别所占最大比例
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    # 随机翻转（如果没有指定的话，将由初始化函数里设置的概率决定执行的概率），翻转的概率，默认是水平翻转
    dict(type='RandomFlip', prob=0.5),
    # 光度失真
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    # 关键是 "img_norm_cfg"
    dict(type='Normalize', **img_norm_cfg),
    # 填充的大小
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    # 默认格式化捆绑操作，简化img,gt_semantic_seg的格式化
    dict(type='DefaultFormatBundle'),
    # 这通常是数据加载器管道的最后阶段。通常，keys 被设置为 "img"、"gt_semantic_seg "的某个子集
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(1024, 512)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    # dict(type='PhotoMetricDistortion'),  # is applied later in dacs.py
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 512),
        # MultiScaleFlipAug is disabled by not providing img_ratios and
        # setting flip=False
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    # 每个gpu计算图片数 batchsize = gpuNum * sampler_per_gpu
    samples_per_gpu=2,
    # 每个gpu分配的线程数
    workers_per_gpu=4,
    train=dict(
        type='UDADataset',                                          # 训练数据集类型
        source=dict(
            type='GTADataset',                                      # 源域数据集类型
            data_root='data/gta/',                                  # 源域数据集路径
            img_dir='images',                                       # 图片路径
            ann_dir='labels',                                       # 标签路径
            pipeline=gta_train_pipeline),                           # 源域数据集训练前的预操作
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=cityscapes_train_pipeline)),
    val=dict(                                                       # 验证和测试都是用的val，有什么区别没？？？
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=test_pipeline))
