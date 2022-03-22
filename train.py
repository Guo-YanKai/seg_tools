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
from utils.CT3D_dataset import Basic3D
from utils.common import split_data_val
from utils.weights_init import init_model
from utils.logger import Train_Logger
import os
from models.Unet3D import Unet3D
from models.unet import Unet
from models.unet_nested import Nested_UNet
from tqdm import tqdm
import torch.optim as optim
from metric import loss
# from metric.loss import DiceLoss, BCEDiceLoss, GeneralizeDiceLoss
from metric.metrics import LossAverage, DiceAverage
from collections import OrderedDict
import numpy as np


def val(net, val_loader, criterion, device, args):
    net.eval()
    val_loss = LossAverage()
    val_dice = DiceAverage(args.n_labels)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            batch_images = batch["image"].to(device, dtype=torch.float32)
            batch_masks = batch["mask"].to(device, dtype=torch.long)

            output = net(batch_images)
            if args.dsv:
                loss = criterion(output[-1], batch_masks)
            else:
                loss = criterion(output, batch_masks)
            val_loss.update(loss.item(), batch_images.size(0))

            if args.dsv:
                val_dice.update(output[-1], batch_masks)
            else:
                val_dice.update(output, batch_masks)

            val_log = OrderedDict({"Val_Loss": val_loss.avg})
            for i in range(args.n_labels):
                val_log.update({f"Val_dice_{i}": val_dice.avg[i]})
    return val_log


def train(net, train_loader, optimizer, criterion, device, scheduler, args):
    print("=====Epoch:{}======lr:{}".format(epoch, optimizer.state_dict()["param_groups"][0]["lr"]))
    net.train()

    train_loss = LossAverage()
    train_dice = DiceAverage(args.n_labels)

    for idx, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
        # batch_images = batch["image"].float().to(device)
        # batch_masks = batch["mask"].long().to(device)
        optimizer.zero_grad()

        batch_images = batch["image"].to(device, dtype=torch.float32)
        batch_masks = batch["mask"].to(device, dtype=torch.long)
        # print(batch_images.shape)
        # print(batch_masks.shape)

        output = net(batch_images)

        if args.dsv:
            loss0 = criterion(output[0], batch_masks)
            print("loss0:", loss0)
            loss1 = criterion(output[1], batch_masks)
            print("loss1:", loss1)
            loss2 = criterion(output[2], batch_masks)
            print("loss2:", loss2)
            loss3 = criterion(output[3], batch_masks)
            print("loss3:", loss3)
            # loss = loss3 + 0.4 * (loss0 + loss1 + loss2)
            loss = loss3 + args.alpha * (loss0 + loss1 + loss2)
        else:
            loss = criterion(output, batch_masks)

        print(loss)

        # loss.backward()
        # optimizer.step()
        #
        # train_loss.update(loss.item(), batch_images.size(0))
        # if args.dsv:
        #     train_dice.update(output[-1], batch_masks)
        # else:
        #     train_dice.update(output, batch_masks)
        print("lr:", type(scheduler.get_last_lr()), scheduler.get_last_lr()[-1])
        break

    # train_log = OrderedDict({"Train_Loss": train_loss.avg})
    # for i in range(args.n_labels):
    #     train_log.update({f"Train_dice_{i}":train_dice.avg[i]})
    # 添加学习率
    # train_log.update({"lr":optimizer.state_dict()["param_groups"][0]["lr"]})
    # return train_log
    return 0


if __name__ == "__main__":
    if not os.path.exists(args.save_path): os.mkdir(args.save_path)
    device = torch.device(f"cuda:{args.device}")
    print(device)
    if args.net_name != "Unet3D":
        dataset = BasicDataset(args.data_path, args)
    elif args.net_name == "Unet3D":
        dataset = Basic3D(args.data_path, args)

    train_sample, val_sample = split_data_val(dataset, args)
    train_loader = DataLoader(dataset, batch_size=args.batch_size,
                              num_workers=args.n_threads, sampler=train_sample, pin_memory=False)
    val_loader = DataLoader(dataset, batch_size=args.batch_size,
                            num_workers=args.n_threads, sampler=val_sample, pin_memory=False)

    # 初始化模型结构/参数
    if args.net_name == "Unet":
        net = Unet(in_channels=1, n_labels=args.n_labels).to(device)
    elif args.net_name == "Unet3D":
        net = Unet3D(in_channels=1, n_labels=args.n_labels).to(device)
    else:
        net = Nested_UNet(in_channels=1, n_labels=args.n_labels, deepsupervision=args.dsv).to(device)

    init_model(net)

    # 是否从已有的模型参数开始训练
    if args.load is not None:
        print(f"model loaded form {args.load}")
        ckpt = torch.load("{}/best_model.pth".format(args.load), map_location=device)
        net.load_state_dict(ckpt["net"])
        print("Model loaded！")
        # net.load_state_dict(torch.load(args.load, map_location=device))

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
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)

    # 导入数据类别权重
    if args.class_weight is not None:
        np_weight = np.load(args.class_weight)
        for i in range(len(np_weight)):
            print(f"类别{i}的权重是:", np_weight[i])
        weights = torch.from_numpy(np_weight).float().to(device)
    else:
        weights = None

    # 定义损失函数

    if args.loss == "CRE":
        criterion = nn.CrossEntropyLoss(weight=weights)
    elif args.loss == "DiceLoss":
        criterion = loss.DiceLoss(args, weight=weights)
    elif args.loss == "BCEDiceLoss":
        criterion = loss.BCEDiceLoss(args, alpha=1, beta=0.5, weight=weights)
    elif args.loss == "FocalLoss":
        criterion = loss.FocalLoss(weight=weights)
    else:
        criterion = loss.GeneralizeDiceLoss()

    log_save_path = os.path.join(args.save_path, args.net_name, args.loss, args.optimizer)
    os.makedirs(log_save_path, exist_ok=True)

    log = Train_Logger(log_save_path, "train_log", args)

    best = [0, float("inf")]  # 初始化最优模型的epoch和dice分数
    trigger = 0  # 早停计数器

    for epoch in range(1, args.epochs + 1):
        train_log = train(net, train_loader, optimizer, criterion, device, scheduler, args)

        break
        # val_log = val(net, val_loader, criterion, device, args)
        #
        # 更新学习率
        # scheduler.step()

        # log.update(epoch, train_log, val_log)
        #
        # # save checkpoints
        # state = {"net": net.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch}
        # torch.save(state, os.path.join(log_save_path, "latest_model.pth"))
        # trigger += 1
        #
        # if val_log["Val_Loss"] < best[-1]:
        #     print("save best model")
        #     torch.save(state, os.path.join(log_save_path, "best_model.pth"))
        #     best[0]=epoch
        #     best[1]=val_log["Val_Loss"]
        #     trigger=0
        # print("Best Performance at Epoch:{}|{}".format(best[0],best[1]))
        # #深度监督系数进行衰减
        # if epoch % 20 == 0:
        #     args.alpha *= 0.8
        # #早停
        # if trigger >= args.early_stop:
        #     print("=>early stopping")
        #     break
    torch.cuda.empty_cache()
