#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/8 11:50
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : train.py
# @software: PyCharm

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
from metric.loss import DiceLoss, GeneralizeDiceLoss
from metric.metrics import LossAverage, DiceAverage
from collections import OrderedDict

def val(net, val_loader, loss, args):
    val_log = OrderedDict({"Val_loss": 0})
    return val_log


def train(net, train_loader, optimizer, criterion, device, scheduler, args):
    print("=====Epoch:{}======lr:{}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
    net.train()

    train_loss = LossAverage()
    train_dice = DiceAverage(args.n_labels)

    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        # batch_images = batch["image"].float().to(device)
        # batch_masks = batch["mask"].long().to(device)

        batch_images = batch["image"].to(device,dtype=torch.float32)
        batch_masks = batch["mask"].to(device,dtype = torch.long)


        output = net(batch_images)

        if args.dsv:
            loss0 = criterion(output[0], batch_masks)
            loss1 = criterion(output[1], batch_masks)
            loss2 = criterion(output[2], batch_masks)
            loss3 = criterion(output[3], batch_masks)
            loss = (loss3 + loss0 + loss1 + loss2)
        else:
            loss = criterion(output, batch_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), batch_images.size(0))
        if args.dsv:
            train_dice.update(output[-1], batch_masks)
        else:
            train_dice.update(output, batch_masks)

        # 可视化模型参数的梯度
        if train_loss.avg is None:
            print([x.grad for x in optimizer.param_groups["0"]["params"]])
            break

    train_log = OrderedDict({"Train_Loss": train_loss.avg})
    for i in range(args.n_labels):
        train_log.update({f"Train_dice_{i}":train_dice.avg[i]})
    return train_log


if __name__ == "__main__":
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    device = torch.device(f"cuda:{args.device}")
    print(device)
    dataset = BasicDataset(args.data_path, args)

    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=args.n_threads, sampler=train_sample,pin_memory=True)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.n_threads, sampler=val_sample,pin_memory=True)

    # 初始化模型结构/参数
    if args.net_name == "Unet":
        net = Unet(in_channels=1, n_labels=args.n_labels).to(device)
    else:
        net = Nested_UNet(in_channels=1, n_labels=args.n_labels, deepsupervision=args.dsv).to(device)
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
    elif args.scheduler =="CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         T_max=args.epochs, eta_min=0)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # 定义损失函数
    if args.loss == "CRE":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "DiceLoss":
        criterion = DiceLoss
    else:
        criterion = GeneralizeDiceLoss

    log_save_path = os.path.join(args.save_path, args.net_name, args.loss,args.optimizer)
    os.makedirs(log_save_path,exist_ok=True)

    log = Train_Logger(log_save_path, "train_log", args)

    best = [0, float("inf")]  # 初始化最优模型的epoch和dice分数
    trigger = 0  # 早停计数器

    for epoch in range(1, args.epochs + 1):
        train_log = train(net, train_loader, optimizer, criterion, device, scheduler, args)
        val_log = val(net, val_loader, criterion, args)

        # 跟新学习率
        scheduler.step()
        # 获得当前的学习率
        print(scheduler.get_last_lr())
        log.update(epoch, train_log, val_log)

    torch.cuda.empty_cache()