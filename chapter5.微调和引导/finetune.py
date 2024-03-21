#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/22 01:04
# @Author  : kunkun
# @File    : finetune.py
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

# #######################################################加载蝴蝶相关内容
dataset_name = "huggan/smithsonian_butterflies_subset"
dataset = load_dataset(dataset_name, split="train")
image_size, batch_size = 256, 4

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

num_epochs = 2
lr = 1e-5
grad_accumulation_steps = 2


# 载入管线
image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
image_pipe.to(device)


optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)
losses = []
for epoch in range(num_epochs):
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        clean_images = batch["images"].to(device)
        noise = torch.randn(clean_images.shape).to(device)
        bs = clean_images.shape[0]

        # 随机选取一个时间步
        timesteps = torch.randint(0, image_pipe.scheduler.num_train_timesteps, (bs,), device=device).long()
        noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

        noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        loss = F.mse_loss(noise_pred, noise)
        losses.append(loss.item())
        loss.backward()

        # 梯度累计，累积到一定步骤后更新模型参数
        if (step + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

    print(f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):]) / len(train_dataloader)}")

plt.plot(losses)

# 保存和载入微调过的管线
image_pipe.save_pretrained("my-finetuned-model")
