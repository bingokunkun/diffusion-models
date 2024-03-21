#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 00:50
# @Author  : kunkun
# @File    : train.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


device = "cuda"


# 载入管线
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

# 创建一个新的调度器并设置推理迭代次数
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

x = torch.randn(4, 3, 256, 256).to(device)
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)

    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    # 使用调度器计算更新后的样本是什么样子
    scheduler_output = scheduler.step(noise_pred, t, x)
    # 更新输入图像
    x = scheduler_output.prev_sample
    # 时不时看一下输入图像和预测的”去噪“图像
    if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
        axs[0].imshow(grid.spu().clip(-1, 1) * 0.5 + 0.5)
        axs[0].set_title(f"current x (step {i}")

        pred_x0 = scheduler_output.pred_original_sample
        grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
        axs[1].imshow(grid.spu().clip(-1, 1) * 0.5 + 0.5)
        axs[1].set_title(f"predicted denoised image (step {i}")
        plt.show()
