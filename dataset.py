# https://zhuanlan.zhihu.com/p/76893455  --> 讲解dataloader和sampler的关系


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
            # transforms.Resize([224, 224]),  # [h, w])
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

# 用于不同尺寸图像的预处理

class MyDataset(Dataset):
    def __init__(self, running_on, dataset_name, txt_name, transformer_mode=None, multinput=False):

        super(MyDataset).__init__()
        self.base_dir = running_on[dataset_name]['base_dir']
        self.txt_dir = os.path.join(self.base_dir, 'dataset_txt')
        self.txt_name = txt_name
        self.multinput = multinput
        self.transformer_mode = -1 if multinput else transformer_mode
        # print('mode:', self.transformer_mode)
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
        # print(image_name)
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



class DiversityDataset(Dataset):
    def __init__(self, running_on, dataset_name_list, txt_name, transformer_mode=None):

        super(DiversityDataset).__init__()
        self.base_dir = running_on['dataset_base_dir']
        self.dataset_dir_list = []
        # 获取所有数据集的base_dir
        for ds_name in dataset_name_list:
            self.dataset_dir_list.append(running_on[ds_name]['base_dir'])

        self.txt_name = txt_name
        self.transformer_mode = transformer_mode
        self.image_transformer = get_image_transform(self.transformer_mode)

        # 先存到temp中，这一步不可省
        temp = []
        for ds_dir in self.dataset_dir_list:
            txt_path = os.path.join(ds_dir, 'dataset_txt', txt_name)
            with open(txt_path, 'r') as f:
                temp.append(f.readlines())

        self.example_list = []
        for idx, big_split in enumerate(temp):
            cur_dataset_name = dataset_name_list[idx]
            for item in big_split:
                msg = cur_dataset_name + ' ' + item
                self.example_list.append(msg)

        images = []
        labels = []
        for line in self.example_list:
            line = line.strip()
            word = line.split()
            image_path = os.path.join(word[0], word[1])
            # word_splits = word[0].split('\\')
            images.append(image_path)
            labels.append(word[-1])

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_name = self.images[item]
        label = self.labels[item]
        label = np.array(label).astype(np.int64)
        img = Image.open(os.path.join(self.base_dir, image_name))  # PIL image shape:（C, W, H）
        # 利用image对图像大小重新设置, Image.ANTIALIAS为高质量的
        # image = image.resize(size, Image.ANTIALIAS)
        img = self.image_transformer(img)
        return img, label, image_name





if __name__ == '__main__':
    val_dataset = DiversityDataset(LOCAL, dataset_name_list=['D3_ECPNight', 'D4_BDD100K'], txt_name='test.txt', transformer_mode=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    for img, label, image_name in val_loader:
        print(label)
        print(img.shape)
        print(image_name)
        break
























