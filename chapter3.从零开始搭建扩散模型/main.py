#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 23:07
# @Author  : kunkun
# @File    : main.py.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")

# torchvision.transforms.ToTensor()是一个转换，它将PIL图像或nparray转换为tensor，并将像素值从0-255的整数范围缩放到0.0-1.0的浮点范围。
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x, y = next(iter(train_dataloader))
print("Input shape: ", x.shape)
print("Labels: ", y)
# plt.imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")
# plt.show()


def corrupt(x, amount):
    """根据amount为输入x加入噪声，这就是前向过程"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # 整理行形状以保证广播机制不出错
    return x * (1 - amount) + noise * amount


# 绘制输入数据
fig, axs = plt.subplots(2, 1, figsize=(12, 5))
axs[0].set_title("Input data")
axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap="Greys")

# 加入噪声
amount = torch.linspace(0, 1, x.shape[0])  # 从0到1，退化更强烈了
noised_x = corrupt(x, amount)

# 绘制加噪版本的图像
axs[1].set_title("corrupted data")
axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap="Greys")

plt.show()
