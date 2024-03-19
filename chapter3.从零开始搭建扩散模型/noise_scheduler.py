#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 00:23
# @Author  : kunkun
# @File    : ddpm.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from matplotlib import pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def func1():
    """输入X和噪声趋势图"""
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
    plt.legend(fontsize="x-large")
    plt.show()


if __name__ == "__main__":
    dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True,
                                         transform=torchvision.transforms.ToTensor())
    train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    xb, yb = next(iter(train_dataloader))
    xb = xb.to(device)
    xb = xb * 2. - 1.  # 映射到(-1, 1)
    print("X shape", xb.shape)

    # 展示干净的原始输入
    axs[0].imshow(torchvision.utils.make_grid(xb[:8])[0].detach().cpu(), cmap="Greys")
    axs[0].set_title("Clean X")

    # 使用调度器加噪
    timesteps = torch.linspace(0, 999, 8).long().to(device)
    print(timesteps)
    noise = torch.randn_like(xb)
    noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
    print("Noisy X shape", noisy_xb.shape)

    # 展示“带噪”样本
    axs[1].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu().clip(-1, 1), cmap="Greys")
    axs[1].set_title("Noisy X (clipped to (-1, 1))")
    axs[2].imshow(torchvision.utils.make_grid(noisy_xb[:8])[0].detach().cpu(), cmap="Greys")
    axs[2].set_title("Noisy X")
    plt.show()