#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 23:56
# @Author  : kunkun
# @File    : test.py
# @Project : diffusion-models
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def corrupt(x, amount):
    """根据amount为输入x加入噪声，这就是前向过程"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # 整理行形状以保证广播机制不出错
    return x * (1 - amount) + noise * amount


if __name__ == "__main__":
    batch_size = 8
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    x, y = next(iter(train_dataloader))
    net = torch.load("net.pth")
    amount = torch.linspace(0, 1, x.shape[0])
    noised_x = corrupt(x, amount)

    with torch.no_grad():
        preds = net(noised_x.to(device)).detach().cpu()

    fig, axs = plt.subplots(3, 1, figsize=(12, 7))
    axs[0].set_title("Input data")
    axs[0].imshow(torchvision.utils.make_grid(x)[0].clip(0, 1), cmap="Greys")
    axs[1].set_title("Corrupted data")
    axs[1].imshow(torchvision.utils.make_grid(noised_x)[0].clip(0, 1), cmap="Greys")
    axs[2].set_title("Network predictions")
    axs[2].imshow(torchvision.utils.make_grid(preds)[0].clip(0, 1), cmap="Greys")

    plt.show()

