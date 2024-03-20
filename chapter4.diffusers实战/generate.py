#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/21 00:00
# @Author  : kunkun
# @File    : generate.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
from diffusers import DDPMPipeline, DDPMScheduler

from utils import show_images


model = torch.load("net.pth")
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
pipeline_output = image_pipe()

show_images(pipeline_output.images[0])

image_pipe.save_pretrained("my_pipeline")
# !ls my_pipeline
# model_index.json scheduler unet


# ###################################################### 第二种生成方法
sample = torch.randn(8, 3, 32, 32).to("cuda")
for i, t in enumerate(noise_scheduler.timesteps):
    with torch.no_grad():
        residual = model(sample, t).sample

    sample = noise_scheduler.step(residual, t, sample).prev_sample

show_images(sample)