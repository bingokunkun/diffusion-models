#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/20 23:28
# @Author  : kunkun
# @File    : butterfly.py
# @Project : diffusion-models
# @Software: PyCharm
import torch.utils.data
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from torch.nn import functional as F
from diffusers import DDPMScheduler, UNet2DModel

from utils import show_images


device = "cuda"
dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
# # 或从本地下载
# dataset = load_dataset("image_folder", data_dir="path/to/folder")
image_size = 32
batch_size = 64

# 数据增强
preprocess = transforms.Compose([
    transforms.Resize((image_size, image_size)),  # 调整大小
    transforms.RandomHorizontalFlip(),  # 随机翻转
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # 映射到[-1, 1]
])


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # 可视化一小批数据
# xb = next(iter(train_dataloader))["images"].to(device)[:8]
# print("X shape:", xb.shape)
# show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)
# timesteps = torch.linspace(0, 999, 8).long().to("cuda")

# 可以控制超参数
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
# # noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
# noise = torch.randn_like(xb)
# noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
# show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)

# ################################################################## 定义模型

model = UNet2DModel(
    sample_size=image_size,
    in_channels=3, out_channels=3,
    layers_per_block=2,  # 每个UNet块的resnet层数
    block_out_channels=(64, 128, 128, 256),
    down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
).to(device)

# # 测试尺寸是否和输入相同
# with torch.no_grad():
#     model_prediction = model(noisy_xb, timesteps).sample
# print(model_prediction.shape)

# ################################################################## 创建训练循环
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
losses = []
for epoch in range(30):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        timesteps = torch.randint(0, noise_scheduler.num_inference_steps, (bs,), device=clean_images.device).long()
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
        noise_pred = model(noisy_images, timesteps, return_dict=False)

        loss = F.mse_loss(noise_pred, noise)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad()
    if epoch % 5 == 4:
        loss_last_epoch = sum(losses[-len(train_dataloader):]) / len(train_dataloader)
        print(f"Epoch: {epoch + 1}, loss: {loss_last_epoch}")