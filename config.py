#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/3 11:48
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : config.py.py
# @software: PyCharm
import argparse

parser = argparse.ArgumentParser("hyper-parameters management")

# 硬件选项
parser.add_argument("--n_threads", type=int, default=0, help="number of threads")
# parser.add_argument("--GPU", action="store true", help="use gpu only")
parser.add_argument("--seed", type=int, default=2022, help="random seed")

# 各种文件存储路径
parser.add_argument("--data_path", default=r"D:\code\data\cbct\label\1", help="data path")
parser.add_argument("--test_data", default=r"D:\code\data\cbct\label", help="test data path")
parser.add_argument("--save_path", default="./experiments", help="save model path")

# 数据处理的参数
parser.add_argument("--n_labels", type=int, default=17, help="number of classes")
parser.add_argument("--valid_rate", type=float, default=0.2, help="验证集划分率")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")

# 训练过程的参数
parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.001,help="learning rate")
parser.add_argument("--early-stop", type=int, default=30, help="early stopping")


args = parser.parse_args()
# print(args)
