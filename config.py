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
parser.add_argument("--device", default="0", help="use gpu only")
parser.add_argument("--seed", type=int, default=2022, help="random seed")

# 各种文件存储路径
parser.add_argument("--data_path", default=r"D:\code\data\cbct\label\1", help="data path")
parser.add_argument("--save_path", default="./experiments", help="save model path")

# 数据处理的参数
parser.add_argument("--n_labels", type=int, default=17, help="number of classes")
parser.add_argument("--valid_rate", type=float, default=0.2, help="验证集划分率")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--class_weight", default=r"D:\code\work_code\github_code\seg_tools\class_weights.npy",
                    help="数据权重")

# 训练过程的参数
parser.add_argument("--net_name", type=str, default="Nested_UNet", help="选择模型结构:[Unet,Nested_UNet, Unet3D]")
parser.add_argument("--dsv", type=bool, default=True, help="deepsupervision for nested_unet /unet3+")
parser.add_argument("--load",default=r"D:\code\work_code\github_code\seg_tools\experiments\Nested_unet\CRE\SGD", help="导入的模型文件")

parser.add_argument("--optimizer", type=str, default="SGD", help="chose one optimizer:[SGD,Adam,RMSprop]")
parser.add_argument("--scheduler", type=str, default="StepLR",
                    help="学习率衰减方式:[StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR]")
parser.add_argument("--loss", type=str, default="DiceLoss",
                    help="损失函数:[CRE, DiceLoss, BCEDiceLoss, FocalLoss, GeneralizeDiceLoss]")
parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
parser.add_argument("--alpha", type=float, default=0.5, help="深度监督系数")

parser.add_argument("--early_stop", type=int, default=30, help="early stopping")

# 测试过程
parser.add_argument("--test_data", default=r"D:\code\data\cbct\3\dcm", help="test data path")
parser.add_argument("--test_output", default=r"D:\code\work_code\github_code\seg_tools\test_output", help="test data path")
parser.add_argument("--threshold", default=0.1, help="阈值")

args = parser.parse_args()
# print(args)
