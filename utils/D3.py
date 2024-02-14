# functions for cropping pedestrians and non-pedestrians in D3
# Note: json files in D3 is very big, 1G

import ijson
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import random
import numpy as np


# Only use train.json is enough, got 45638 nonPed images
# 获取不包含行人的图片，用于后续分割
def get_nonPed_examples(json_path, nonPed_txt):
    '''
    :param json_path: path to train.json
    :param nonPed_txt: path to save nonPed txt
    :return:
    '''
    nonPed_num = 0
    nonPed_list = []
    personInclude_cls = ['rider', 'motor', 'person', 'bike']
    with open(json_path) as f:
        for record in tqdm(ijson.items(f, "item")):
            nonPed_flag = True
            object_list = record['labels']
            for obj in object_list:
                category = obj['category']
                if category in personInclude_cls:
                    nonPed_flag = False
                    break
            if nonPed_flag:
                nonPed_num += 1
                nonPed_list.append(record['name'])
    print('num:', nonPed_num)
    # print(nonPed_list)

    with open(nonPed_txt, 'w') as f:
        for item in nonPed_list:
            msg = os.path.join('', item) + '\n'
            f.write(msg)


# json_path = r'D:\my_phd\dataset\D3_BDD\bdd100k\labels\bdd100k_labels_images_train.json'
# nonPed_txt = r'D:\my_phd\dataset\D3_BDD\nonPed.txt'
# get_nonPed_examples(json_path, nonPed_txt)

# 分割非行人图片
def crop_nonPeds(nonPed_txt, image_dir, save_to):
    with open(nonPed_txt) as f:
        data = f.readlines()

    # for idx, item in enumerate(data):
    for i in tqdm(range(7815)):
        item = data[i]
        item = item.strip()
        image_path = os.path.join(image_dir, item)
        image = Image.open(image_path)
        w, h = image.size
        # 由于这次的non Pedestrian数量很充足，每张图片裁剪出一张图就可以
        is_ok = False
        while not is_ok:
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
                # plt.show()
                # 存储当前裁剪的图片
                cropped_name = item.split('\\')[1]
                save_ptah = os.path.join(save_to, cropped_name)
                cropped.save(save_ptah)
                is_ok = True


# nonPed_txt = r'D:\my_phd\dataset\D3_BDD\nonPed.txt'
# image_dir = r'D:\my_phd\dataset\D3_BDD\bdd100k\images\100k'
# save_to = r'D:\my_phd\dataset\D3_BDD\nonPedestrian'
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


# # 划分 pedestrian
# image_dir = r'D:\my_phd\dataset\D3_BDD\pedestrian'
# txt_dir = r'D:\my_phd\dataset\D3_BDD\dataset_txt'
# split_dataset(image_dir, txt_dir, 'pedestrian', '1')

# # 划分 non pedestrian
# image_dir = r'D:\my_phd\dataset\D3_BDD\nonPedestrian'
# txt_dir = r'D:\my_phd\dataset\D3_BDD\dataset_txt'
# split_dataset(image_dir, txt_dir, 'nonPedestrian', '0')



















