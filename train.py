# 训练函数
# aaaa
import torch
import torch.nn as nn
from collections import Counter
from config import args
from torch.utils.data import DataLoader
from utils.CT_dataset import BasicDataset
from utils.common import split_data_val
from utils.weights_init import init_model
from utils.logger import Train_Logger
import os
from models.unet import Unet
from models.unet_nested import Nested_UNet
from tqdm import tqdm
import torch.optim as optim
from metric.loss import DiceLoss, CrossEntropyLoss, GeneralizeDiceLoss


def val(net, val_loader, loss, args):
    return 0


def train(net, train_loader, optimizer, loss, args):
    return 0


if __name__ == "__main__":
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    device = torch.device(f"cuda:{args.device}")
    print(device)
    dataset = BasicDataset(args.data_path, args)

    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=args.n_threads, sampler=train_sample)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.n_threads, sampler=val_sample)

    # 初始化模型结构/参数
    net = Unet(in_channels=1, n_labels=args.n_labels).to(device)
    net.apply(init_model)

    # 定义优化器
    if args.optimizer == "Adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == "RMSprop":
        optimizer = optim.RMSprop(net.parameters(), lr=args.lr, weight_decay=0.9)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    # 定义学习率衰减方式：
    if args.scheduler == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15, 20],
                                                   gamma=0.8)
    elif args.scheduler == "ExponentialLR":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # 定义损失函数
    if args.loss == "CRE":
        loss = CrossEntropyLoss
    elif args.loss == "diceLoss":
        loss = DiceLoss
    else:
        loss = GeneralizeDiceLoss

    log = Train_Logger(args.save_path, "train_log")
    best = [0, float("inf")]  # 初始化最优模型的epoch和dice分数
    trigger = 0  # 早停计数器

    for epoch in range(1, args.epochs + 1):
        train_log = train(net, train_loader, optimizer, loss, args)
        val_log = val(net, val_loader, loss, args)
