#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/14 17:41
# @Author  : guoyankai
# @Email   : 392759421@qq.com
# @File    : test.py.py
# @software: PyCharm
import torch
import torch.nn as nn
from models.Unet3D import Unet3D
from models.unet import Unet
from models.unet_nested import Nested_UNet
import os
from config import args
from tqdm import tqdm
from utils.CT_dataset import TestDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter
from PIL import Image
import numpy as np

def dcm_to_png(dcm_path):
    """待处理"""
    png =0
    return png

def test(net, image):
    net.eval()
    with torch.no_grad():
        output = net(image)
        if args.dsv:
            output = output[-1]
        if args.n_labels>1:
            probs = F.softmax(output, dim=1)
        else:
            probs = F.sigmoid(output)
        probs = probs.squeeze(0)

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask>0.1
            masks.append(mask)
        return masks


if __name__=="__main__":
    os.makedirs(args.test_output, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    print("device:",device)
    dataset = TestDataset(args.test_data, args)
    test_loader = DataLoader(dataset, batch_size=1,
                              num_workers=args.n_threads, pin_memory=False)
    if args.net_name == "Unet":
        net = Unet(in_channels=1, n_labels=args.n_labels).to(device)
    elif args.net_name=="Unet3D":
        net = Unet3D(in_channels=1, n_labels=args.n_labels).to(device)
    else:
        net = Nested_UNet(in_channels=1, n_labels=args.n_labels, deepsupervision=args.dsv).to(device)

    model_pth = os.path.join(args.save_path, args.net_name, args.loss,args.optimizer)
    print("model_pth:", model_pth)
    ckpt = torch.load("{}/best_model.pth".format(model_pth),map_location=device)
    net.load_state_dict(ckpt["net"])
    print("Model loaded！")

    for idx, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = batch["image"].to(device, dtype=torch.float32)
        masks = test(net, image)


        img_name_to_ext= batch["name"][0]
        output_img_dir = os.path.join(args.test_output, img_name_to_ext)
        os.makedirs(output_img_dir, exist_ok=True)

        if args.n_labels==1:
            image_idex = Image.fromarray((masks*255).astype(np.uint8))
            image_idex.save(os.path.join(output_img_dir, img_name_to_ext))
        else:
            for idx in range(0,len(masks)):
                img_name_idx = img_name_to_ext +"_"+str(idx)+".png"
                image_idx = Image.fromarray((masks[idx]*255).astype(np.uint8))
                image_idx.save(os.path.join(output_img_dir, img_name_idx))


