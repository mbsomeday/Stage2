from math import floor, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialPyramidPooling2d(nn.Module):

    def __init__(self, num_level, pool_type='max_pool'):
        super(SpatialPyramidPooling2d, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):

        N, C, H, W = x.size()

        # print('多尺度提取信息，并进行特征融合...')
        # print()
        for i in range(self.num_level):
            level = i + 1
            # print('第', level, '次计算池化核：')
            kernel_size = (ceil(H / level), ceil(W / level))
            # print('kernel_size: ', kernel_size)
            stride = (ceil(H / level), ceil(W / level))
            # print('stride: ', stride)
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))
            # print('padding: ', padding)
            # print()

            # print('进行最大池化并将提取特征展开：')
            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)

            if i == 0:
                res = tensor
                # print('展开大小为： ', res.size())
                # print()
            else:
                res = torch.cat((res, tensor), 1)
                # print('合并为： ', res.size())
                # print()
        return res

