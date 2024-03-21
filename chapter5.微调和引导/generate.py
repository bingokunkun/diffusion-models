#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 01:24
# @Author  : kunkun
# @File    : generate.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
import torchvision
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from diffusers import DDIMScheduler, DDPMPipeline

device = "cuda"
scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
scheduler.set_timesteps(num_inference_steps=40)

image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)

x = torch.randn(8, 3, 256, 256).to("cuda")
for i, t in tqdm(enumerate(scheduler.timesteps)):
    model_input = scheduler.scale_model_input(x, t)
    with torch.no_grad():
        noise_pred = image_pipe.unet(model_input, t)["sample"]
    x = scheduler.step(noise_pred, t, x).prev_sample
grid = torchvision.utils.make_grid(x, nrow=4)
plt.show(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)