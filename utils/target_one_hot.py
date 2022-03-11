#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 14:52
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : target_one_hot.py
# @software: PyCharm

import torch



def one_hot_1(target, n_labels=17):
    n,h,w = target.size()
    one_hot = torch.zeros(n, n_labels, h, w).scatter_(1, target.view(n, 1, h, w), 1).to(target.device)
    return one_hot

def one_hot_2(target, n_labels=17):
    tensor_list=[]
    for i in range(n_labels):
        temp_prob = target==i
        tensor_list.append(temp_prob.unsqueeze(1))
    output_tensor = torch.cat(tensor_list,dim=1)
    return output_tensor.float()




