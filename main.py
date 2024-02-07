from torchvision import models
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.eval()

# 加载图像并进行预处理
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])

image_path = r'1.jpg'
image = Image.open(image_path)
image = transform(image)

# 将图像输入模型并获取输出
output = model(image.unsqueeze(0))

print(output)



























