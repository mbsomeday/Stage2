# functions for cropping pedestrians and non-pedestrians in D3

import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import random
import numpy as np


# 分割 non pedestrian images
def crop_nonPeds(nonPed_txt, image_dir, save_to):
    with open(nonPed_txt) as f:
        data = f.readlines()

    for idx, item in enumerate(tqdm(data)):
        item = data[idx]
        item = item.strip()
        image_path = os.path.join(image_dir, item)
        image = Image.open(image_path)
        w, h = image.size
        nonPed_num = 0
        while nonPed_num < 16:
            x0 = random.randint(0, w - 250)
            y0 = random.randint(0, h - 250)
            cropped = image.crop((x0, y0, x0 + 250, y0 + 250))
            # 检验是否裁剪出重合度过高的图片或者黑图
            gray = cropped.convert('L')
            crop_array = np.array(gray)
            gray_min = crop_array.min()
            gray_max = crop_array.max()
            diff = gray_max - gray_min
            if diff > 0:
                # plt.imshow(cropped)
                # plt.title(nonPed_num)
                # plt.show()
                # 存储当前裁剪的图片
                cropped_name = item.split('\\')[1].split('.')[0] + '_' +str(nonPed_num) + '.png'
                save_ptah = os.path.join(save_to, cropped_name)
                cropped.save(save_ptah)
                nonPed_num += 1


# nonPed_txt = r'D:\my_phd\dataset\D3_ECPNight\nonPed.txt'
# image_dir = r'D:\my_phd\dataset\D3_ECPNight\ECP\night\img'
# save_to = r'D:\my_phd\dataset\D3_ECPNight\nonPedestrian'
# crop_nonPeds(nonPed_txt, image_dir, save_to)


# 划分数据集
def split_dataset(image_dir, txt_dir, cls_name, cls_code):
    image_list = os.listdir(image_dir)
    num_image = len(image_list)

    train_num = int(num_image * 0.6)
    val_num = int(num_image * 0.2)
    test_num = int(num_image * 0.2)

    indices = list(range(num_image))
    random.seed(13)
    random.shuffle(indices)

    # train set
    with open(os.path.join(txt_dir, 'train.txt'), 'a') as f:
        for idx in indices[: train_num]:
            image = image_list[idx]
            msg = os.path.join(cls_name, image) + ' ' + cls_name + ' ' + cls_code + '\n'
            f.write(msg)

    # val set
    with open(os.path.join(txt_dir, 'val.txt'), 'a') as f:
        for idx in indices[train_num: train_num+val_num]:
            image = image_list[idx]
            msg = os.path.join(cls_name, image) + ' ' + cls_name + ' ' + cls_code + '\n'
            f.write(msg)

    # test set
    with open(os.path.join(txt_dir, 'test.txt'), 'a') as f:
        for idx in indices[train_num+val_num: ]:
            image = image_list[idx]
            msg = os.path.join(cls_name, image) + ' ' + cls_name + ' ' + cls_code + '\n'
            f.write(msg)


# 划分 non pedestrian
image_dir = r'D:\my_phd\dataset\D3_ECPNight\nonPedestrian'
txt_dir = r'D:\my_phd\dataset\D3_ECPNight\dataset_txt'
split_dataset(image_dir, txt_dir, 'nonPedestrian', '0')

# 划分 pedestrian
image_dir = r'D:\my_phd\dataset\D3_ECPNight\pedestrian'
txt_dir = r'D:\my_phd\dataset\D3_ECPNight\dataset_txt'
split_dataset(image_dir, txt_dir, 'pedestrian', '1')
















