# functions for cropping pedestrians and non-pedestrians in D2

import ijson
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import numpy as np




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
# image_dir = r'D:\my_phd\dataset\D2_CityPersons\nonPedestrian'
# txt_dir = r'D:\my_phd\dataset\D2_CityPersons\dataset_txt'
# split_dataset(image_dir, txt_dir, 'nonPedestrian', '0')
#
# 划分 pedestrian
image_dir = r'D:\my_phd\dataset\D2_CityPersons\pedestrian'
txt_dir = r'D:\my_phd\dataset\D2_CityPersons\dataset_txt'
split_dataset(image_dir, txt_dir, 'pedestrian', '1')
#





















