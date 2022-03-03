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