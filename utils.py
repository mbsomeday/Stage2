


# # -------------------------------- TODO vbb转json --------------------------------
from scipy.io import loadmat
from collections import defaultdict
def read_vbb(path):
    """
    Read the data of a .vbb file to a dictionary.
    """
    assert path[-3:] == 'vbb'

    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if obj.shape[1] > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data


data = read_vbb(r'V000.vbb')
print(data.keys())

nFrame = data['nFrame']
maxObj = data['maxObj']
altered = data['altered']
frames = data['frames']




# # -------------------------------- TODO 读取json --------------------------------
# import json
#
# json_path = r'test.json'
# # data = json.loads(json_path)
# with open(json_path) as f:
#     data = json.load(f)
#
# print(data.keys())
# print(data['categories'])
#
# images = data['images']
# annotations = data['annotations']
#
# print(images[0])
# print(annotations[0])
#
# print(data['info'])

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












