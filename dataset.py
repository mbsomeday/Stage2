import matplotlib.pyplot as plt
import torch, os, cv2, argparse
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from torchvision import transforms

from cv_models import LOCAL

def get_image_transform(mode):
    '''
    Args:
        mode: 0, 1, 2 -> return single transformer
                -1 -> return transformer list
    '''
    image_transform = [
        transforms.Compose([
            transforms.Resize([224, 224]),  # [h, w])
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Resize([224, 224]),  # [h, w])

            transforms.RandomHorizontalFlip(p=0.5),  # 水平翻转
            transforms.ToTensor()
        ]),
        transforms.Compose([
            transforms.Resize([224, 224]),  # [h, w])

            transforms.RandomHorizontalFlip(p=1),  # 水平翻转
            transforms.ToTensor()
        ]),
    ]
    if mode >= 0:
        return image_transform[mode]
    else:
        return image_transform[: 3]


class MyDataset(Dataset):
    def __init__(self, running_on, txt_name, transformer_mode=None, multinput=False):

        super(MyDataset).__init__()
        self.base_dir = running_on['ECPD']['base_dir']
        self.txt_dir = os.path.join(self.base_dir, running_on['ECPD']['txt_dir'])
        self.txt_name = txt_name
        self.multinput = multinput
        self.transformer_mode = -1 if multinput else transformer_mode
        self.image_transformer = get_image_transform(self.transformer_mode)

        txt_path = os.path.join(self.txt_dir, txt_name)
        with open(txt_path, 'r') as f:
            data = f.readlines()

        images = []
        labels = []
        for line in data:
            line = line.strip()
            word = line.split()
            word_splits = word[0].split('\\')
            images.append(os.path.join(self.base_dir, word_splits[0], word_splits[1]))
            labels.append(word[2])

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images[item]
        label = self.labels[item]
        label = np.array(label).astype(np.int64)
        img = Image.open(image_name)  # PIL image shape:（C, W, H）
        # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
        # image = image.resize(size, Image.ANTIALIAS)
        if not self.multinput:
            # 单输入的情况
            img = self.image_transformer(img)
            return img, label, image_name
        else:
            # 需要对图片进行多种变化的情况
            image_list = []
            for trans in self.image_transformer:
                image_list.append(trans(img))
            return image_list, label, image_name




























