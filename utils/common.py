#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 11:55
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : common.py.py
# @software: PyCharm

def print_models(net):
    num_params =0
    for param in net.parameters():
        num_params+=param.numel()
    print(net)
    print("模型的参数量:", num_params)

def norm_img(image_array):
    # 归一化像素值到（0，1）之间，且将溢出值取边界值
    # image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    normal_image = (image_array - np.min(image_array)) / (np.max(image_array) - np.min(image_array))
    normal_image[normal_image > 1] = 1.
    normal_image[normal_image < 0] = 0.
    return normal_image