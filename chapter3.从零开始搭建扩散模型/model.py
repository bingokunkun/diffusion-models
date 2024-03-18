#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/18 23:26
# @Author  : kunkun
# @File    : model.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
from torch import nn


class BasicUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.down_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 64, kernel_size=5, padding=2)
        ])
        self.up_layers = nn.ModuleList([
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, out_channels, kernel_size=5, padding=2)
        ])
        self.act = nn.SiLU()
        self.downscale = nn.MaxPool2d(2)
        self.upscale = nn.Upsample(scale_factor=2)

    def forward(self, x):
        h = []
        for i, l in enumerate(self.down_layers):
            x = self.act(l(x))  # 通过运算层和激活函数
            if i < 2:  # 选择下行路径的前两层
                h.append(x)  # 排列供残差链接使用的数据
                x = self.downscale(x)  # 进行下采样，适配下一层的输入

        for i, l in enumerate(self.up_layers):
            if i > 0:  # 上行路径的后两层
                x = self.upscale(x)  # 上采样
                x += h.pop()  # 残差连接
            x = self.act(l(x))  # 运算层+激活函数

        return x


if __name__ == "__main__":
    net = BasicUNet()
    x = torch.rand(8, 1, 28, 28)
    print(net(x).shape)