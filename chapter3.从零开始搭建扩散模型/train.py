#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/19 23:31
# @Author  : kunkun
# @File    : train.py
# @Project : diffusion-models
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from model import BasicUNet


def corrupt(x, amount):
    """根据amount为输入x加入噪声，这就是前向过程"""
    noise = torch.rand_like(x)
    amount = amount.view(-1, 1, 1, 1)  # 整理行形状以保证广播机制不出错
    return x * (1 - amount) + noise * amount


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
n_epochs = 3
net = BasicUNet()
net.to(device)

loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
losses = []

# 训练
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        x = x.to(device)
        noise_amount = torch.rand(x.shape[0]).to(device)
        noisy_x = corrupt(x, noise_amount)

        pred = net(noisy_x)
        loss = loss_fn(pred, x)

        # 反向传播，更新参数
        opt.zero_grad()
        loss.backward()
        opt.step()

        # 存储损失，供后期查看
        losses.append(loss.item())

    # 输出在每个周期训练得到的损失的均值
    avg_loss = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
    print(f"Finished epoch {epoch}. Average loss for this epoch: {avg_loss:05f}")

plt.plot(losses)
plt.ylim(0, 0.1)

torch.save(net, "net.pth")
# model = torch.load("net.pth")
