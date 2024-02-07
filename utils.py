















# -------------------------------- TODO 数据集的分割 --------------------------------
# import random
# import os

# dataset_dir = r'D:\chrom_download\grouped2_ECP'
# pedestrian_dirName = 'pedestrian'
# background_dirName = 'background'
#
# dataseTxt_dirName = 'dataset_txt'
#
# ped_num = len(os.listdir(os.path.join(dataset_dir, pedestrian_dirName)))
# bg_num = len(os.listdir(os.path.join(dataset_dir, background_dirName)))
#
# ped_indices = list(range(ped_num))
#
# def wriTxt(indices, txtName, dirName, dir_code, image_list):
#     print('-' * 40 + txtName + '-' * 40)
#     txt_path = os.path.join(dataset_dir, dataseTxt_dirName, txtName)
#     with open(txt_path, 'a') as f:
#         for idx in indices:
#             cur_ped = os.path.join(dirName, image_list[idx])
#             print('Current Wrinting:', cur_ped)
#             msg = cur_ped + ' ' + dirName + ' ' + str(dir_code) + '\n'
#             f.write(msg)
#
# def split_oneCls(dirName, dir_code):
#     example_num = len(os.listdir(os.path.join(dataset_dir, dirName)))
#     indices = list(range(example_num))
#
#     # 打乱列表
#     random.seed(13)
#     random.shuffle(indices)
#
#     train_num = int(example_num * 0.6)
#     test_num = int(example_num * 0.2)
#     val_num = int(example_num * 0.2)
#
#     image_list = os.listdir(os.path.join(dataset_dir, dirName))
#
#     wriTxt(indices[: train_num], 'train.txt', dirName, dir_code, image_list)
#     wriTxt(indices[train_num: (train_num+test_num)], 'test.txt', dirName, dir_code, image_list)
#     wriTxt(indices[(train_num+test_num): (train_num+test_num+val_num)], 'val.txt', dirName, dir_code, image_list)












