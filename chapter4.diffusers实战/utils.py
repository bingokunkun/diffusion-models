#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 23:15
# @Author  : kunkun
# @File    : utils.py
# @Project : diffusion-models
# @Software: PyCharm
import numpy as np
import torchvision.utils
from PIL import Image


def show_images(x):
    x = x * 0.5 + 0.5
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im
