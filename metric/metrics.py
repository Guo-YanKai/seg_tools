#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/10 11:48
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : metrics.py.py
# @software: PyCharm

import torch
import numpy as np
from utils.target_one_hot import one_hot_1

class LossAverage(object):
    """计算并存储当前损失值和平均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)


class DiceAverage(object):
    def __init__(self, n_labels):
        self.n_labels = n_labels
        self.reset()

    def reset(self):
        # np.array和asarray都是转换为ndarray类型，但是array会将copy出一个副本，asarray不会
        self.value = np.asarray([0] * self.n_labels, dtype="float64")
        self.avg = np.asarray([0]* self.n_labels, dtype="float64")
        self.sum = np.asarray([0]* self.n_labels, dtype="float64")
        self.count=0

    def update(self, logits, target):
        self.value = DiceAverage.get_dices(logits, target)
        self.sum +=self.value
        self.count+=1
        self.avg  = np.around(self.sum/self.count, 4)



    @staticmethod
    def get_dices(logits, target):
        # 计算每个类别的dice分数
        target = one_hot_1(target)
        dices=[]
        for class_index in range(target.size()[1]):
            inter = torch.sum(logits[:,class_index,:,:] * target[:,class_index,:,:])
            union = torch.sum(logits[:,class_index,:,:])+torch.sum(target[:,class_index,:,:])
            dice = (2.* inter + 1) / (union+1)
            dices.append(dice.item())
        return np.array(dices)