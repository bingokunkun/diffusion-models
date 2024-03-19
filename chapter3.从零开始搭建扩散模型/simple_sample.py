#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 00:08
# @Author  : kunkun
# @File    : simple_sample.py
# @Project : diffusion-models
# @Software: PyCharm
import matplotlib.pyplot as plt
import torch
import torchvision.utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    n_steps = 5
    x = torch.rand(8, 1, 28, 28).to(device)
    step_history = [x.detach().cpu()]
    pred_output_history = []
    net = torch.load("net.pth")

    for i in range(n_steps):
        with torch.no_grad():
            pred = net(x)
        pred_output_history.append(pred.detach().cpu())
        mix_factor = 1 / (n_steps - i)
        x = x * (1 - mix_factor) + pred * mix_factor
        step_history.append(x.detach().cpu())

    fig, axs = plt.subplots(n_steps, 2, figsize=(9, 4), sharex=True)
    axs[0, 0].set_title("x (model input)")
    axs[0, 1].set_title("model prediction")
    for i in range(n_steps):
        axs[i, 0].imshow(torchvision.utils.make_grid(step_history[i])[0].clip(0, 1), cmap="Greys")
        axs[i, 1].imshow(torchvision.utils.make_grid(pred_output_history[i])[0].clip(0, 1), cmap="Greys")

    plt.show()
