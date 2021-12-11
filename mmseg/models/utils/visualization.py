# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image

# 调色板
Cityscapes_palette = [
    128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153,
    153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130,
    180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100, 0, 80, 100,
    0, 0, 230, 119, 11, 32, 128, 192, 0, 0, 64, 128, 128, 64, 128, 0, 192, 128,
    128, 192, 128, 64, 64, 0, 192, 64, 0, 64, 192, 0, 192, 192, 0, 64, 64, 128,
    192, 64, 128, 64, 192, 128, 192, 192, 128, 0, 0, 64, 128, 0, 64, 0, 128,
    64, 128, 128, 64, 0, 0, 192, 128, 0, 192, 0, 128, 192, 128, 128, 192, 64,
    0, 64, 192, 0, 64, 64, 128, 64, 192, 128, 64, 64, 0, 192, 192, 0, 192, 64,
    128, 192, 192, 128, 192, 0, 64, 64, 128, 64, 64, 0, 192, 64, 128, 192, 64,
    0, 64, 192, 128, 64, 192, 0, 192, 192, 128, 192, 192, 64, 64, 64, 192, 64,
    64, 64, 192, 64, 192, 192, 64, 64, 64, 192, 192, 64, 192, 64, 192, 192,
    192, 192, 192, 32, 0, 0, 160, 0, 0, 32, 128, 0, 160, 128, 0, 32, 0, 128,
    160, 0, 128, 32, 128, 128, 160, 128, 128, 96, 0, 0, 224, 0, 0, 96, 128, 0,
    224, 128, 0, 96, 0, 128, 224, 0, 128, 96, 128, 128, 224, 128, 128, 32, 64,
    0, 160, 64, 0, 32, 192, 0, 160, 192, 0, 32, 64, 128, 160, 64, 128, 32, 192,
    128, 160, 192, 128, 96, 64, 0, 224, 64, 0, 96, 192, 0, 224, 192, 0, 96, 64,
    128, 224, 64, 128, 96, 192, 128, 224, 192, 128, 32, 0, 64, 160, 0, 64, 32,
    128, 64, 160, 128, 64, 32, 0, 192, 160, 0, 192, 32, 128, 192, 160, 128,
    192, 96, 0, 64, 224, 0, 64, 96, 128, 64, 224, 128, 64, 96, 0, 192, 224, 0,
    192, 96, 128, 192, 224, 128, 192, 32, 64, 64, 160, 64, 64, 32, 192, 64,
    160, 192, 64, 32, 64, 192, 160, 64, 192, 32, 192, 192, 160, 192, 192, 96,
    64, 64, 224, 64, 64, 96, 192, 64, 224, 192, 64, 96, 64, 192, 224, 64, 192,
    96, 192, 192, 224, 192, 192, 0, 32, 0, 128, 32, 0, 0, 160, 0, 128, 160, 0,
    0, 32, 128, 128, 32, 128, 0, 160, 128, 128, 160, 128, 64, 32, 0, 192, 32,
    0, 64, 160, 0, 192, 160, 0, 64, 32, 128, 192, 32, 128, 64, 160, 128, 192,
    160, 128, 0, 96, 0, 128, 96, 0, 0, 224, 0, 128, 224, 0, 0, 96, 128, 128,
    96, 128, 0, 224, 128, 128, 224, 128, 64, 96, 0, 192, 96, 0, 64, 224, 0,
    192, 224, 0, 64, 96, 128, 192, 96, 128, 64, 224, 128, 192, 224, 128, 0, 32,
    64, 128, 32, 64, 0, 160, 64, 128, 160, 64, 0, 32, 192, 128, 32, 192, 0,
    160, 192, 128, 160, 192, 64, 32, 64, 192, 32, 64, 64, 160, 64, 192, 160,
    64, 64, 32, 192, 192, 32, 192, 64, 160, 192, 192, 160, 192, 0, 96, 64, 128,
    96, 64, 0, 224, 64, 128, 224, 64, 0, 96, 192, 128, 96, 192, 0, 224, 192,
    128, 224, 192, 64, 96, 64, 192, 96, 64, 64, 224, 64, 192, 224, 64, 64, 96,
    192, 192, 96, 192, 64, 224, 192, 192, 224, 192, 32, 32, 0, 160, 32, 0, 32,
    160, 0, 160, 160, 0, 32, 32, 128, 160, 32, 128, 32, 160, 128, 160, 160,
    128, 96, 32, 0, 224, 32, 0, 96, 160, 0, 224, 160, 0, 96, 32, 128, 224, 32,
    128, 96, 160, 128, 224, 160, 128, 32, 96, 0, 160, 96, 0, 32, 224, 0, 160,
    224, 0, 32, 96, 128, 160, 96, 128, 32, 224, 128, 160, 224, 128, 96, 96, 0,
    224, 96, 0, 96, 224, 0, 224, 224, 0, 96, 96, 128, 224, 96, 128, 96, 224,
    128, 224, 224, 128, 32, 32, 64, 160, 32, 64, 32, 160, 64, 160, 160, 64, 32,
    32, 192, 160, 32, 192, 32, 160, 192, 160, 160, 192, 96, 32, 64, 224, 32,
    64, 96, 160, 64, 224, 160, 64, 96, 32, 192, 224, 32, 192, 96, 160, 192,
    224, 160, 192, 32, 96, 64, 160, 96, 64, 32, 224, 64, 160, 224, 64, 32, 96,
    192, 160, 96, 192, 32, 224, 192, 160, 224, 192, 96, 96, 64, 224, 96, 64,
    96, 224, 64, 224, 224, 64, 96, 96, 192, 224, 96, 192, 96, 224, 192, 0, 0, 0
]

'''
PIL.Image.mode
1	1位像素，黑和白，存成8位的像素
L	8位像素，黑白

P	8位像素，使用调色板映射到任何其他模式

RGB	3× 8位像素，真彩
RGBA	4×8位像素，真彩+透明通道
CMYK	4×8位像素，颜色隔离
YCbCr	3×8位像素，彩色视频格式
I	32位整型像素
F	32位浮点型像素

im.convert(mode)
'''
# 颜色掩模
def colorize_mask(mask, palette):
    # 给调色板不够的，补0
    # 调色板模式 ("P")使用一个颜色调色板为每个像素定义具体的颜色值
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # fromarray:array转换成image
    # np.uint8()函数，但是这个函数仅仅是对原数据和0xff相与(和最低2字节数据相与)，这就容易导致如果原数据是大于255的，那么在直接使用np.uint8()后，比第八位更大的数据都被截断了
    # 网上更推荐先归一化再使用uint8
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    # 将调色板添加到图像中 
    new_mask.putpalette(palette)
    return new_mask


def _colorize(img, cmap, mask_zero=False):
    vmin = np.min(img)
    vmax = np.max(img)
    # 去掉图片小于0的维度
    mask = (img <= 0).squeeze()
    # 获取配色方案
    cm = plt.get_cmap(cmap)
    # numpy.clip(a, a_min, a_max, out=None)[source]
    # clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min。
    # 获得RGB的彩色图
    colored_image = cm(np.clip(img.squeeze(), vmin, vmax) / vmax)[:, :, :3]
    # Use white if no depth is available (<= 0)
    if mask_zero:
        colored_image[mask, :] = [1, 1, 1]
    return colored_image

# 显示图
def subplotimg(ax,
               img,
               title,
               range_in_title=False,
               palette=Cityscapes_palette,
               **kwargs):
    if img is None:
        return
    # 被该语句 wrap 起来的部分将不会track 梯度
    with torch.no_grad():
        # 如果img是tensor的话返回true，否则返回false。
        if torch.is_tensor(img):
            # 把图片放到cpu上
            img = img.cpu()
        # （nSample）x C x H x W
        if len(img.shape) == 2:
            if torch.is_tensor(img):
                img = img.numpy()
        elif img.shape[0] == 1:
            if torch.is_tensor(img):
                img = img.numpy()
            img = img.squeeze(0)
        elif img.shape[0] == 3:
            # H x W x C
            img = img.permute(1, 2, 0)
            if not torch.is_tensor(img):
                img = img.numpy()
        if kwargs.get('cmap', '') == 'cityscapes':
            kwargs.pop('cmap')
            if torch.is_tensor(img):
                img = img.numpy()
            img = colorize_mask(img, palette)

    if range_in_title:
        vmin = np.min(img)
        vmax = np.max(img)
        title += f' {vmin:.3f}-{vmax:.3f}'

    ax.imshow(img, **kwargs)
    ax.set_title(title)
