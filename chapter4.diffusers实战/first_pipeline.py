#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 23:12
# @Author  : kunkun
# @File    : first_pipeline.py
# @Project : diffusion-models
# @Software: PyCharm
import torch
from diffusers import StableDiffusionPipeline

from utils import make_grid

device = "cuda"
model_id = "sd-dreambooth-library/mr-potato-head"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)

prompt = "an abstract oil painting of sks mr potato head by picasso"
images = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images
make_grid(images)
