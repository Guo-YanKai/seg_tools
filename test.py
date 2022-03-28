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
from utils.Test_CT_Data import TestDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms
from collections import Counter
from PIL import Image
import numpy as np
from utils.colors import get_colors
import cv2
import matplotlib.pyplot as plt
import pydicom
import vtk
from collections import Counter
import shutil

def test(net, image):
    net.eval()

    with torch.no_grad():
        output = net(image)
        if args.dsv:
            output = output[-1]

        # 这个地方有疑问？？
        if args.n_labels > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = F.sigmoid(output)

        probs = probs.squeeze(0)

        # 第一种：采用argmax保存每个像素得最大类别值。
        masks = F.one_hot(probs.argmax(dim=0), args.n_labels).permute(2, 0, 1).cpu().numpy()

        # 第二种，对于每个像素，采用阈值得方法。
        # tf = transforms.Compose([
        #     transforms.ToPILImage(),
        #     transforms.ToTensor(),
        # ])
        # masks = []2
        # for prob in probs:
        #     prob = tf(prob.cpu())
        #     mask = prob.squeeze().cpu().numpy()
        #     mask = mask > 0.1
        #     masks.append(mask)

        return masks


def dcm_to_stl(out_dcm_path, out_stl_path, n_labels):
    """进行stl重建"""
    os.makedirs(out_stl_path, exist_ok=True)

    for i in range(n_labels):
        sig_dcm_path = os.path.join(out_dcm_path, str(i))
        # 这段代码用来判断是否有牙齿的dcm没有数值，就不进行重建
        dcm_list = os.listdir(sig_dcm_path)
        for j in range(len(dcm_list)):
            dcm = pydicom.read_file(sig_dcm_path + '/' + dcm_list[j])
            dcm_data = dcm.pixel_array
            if len(Counter(dcm_data.ravel())) != 1:
                break
        if j >= 511:
            print(i)
            continue
        else:
            filename = os.path.join(out_stl_path, "{}.stl".format(str(i)))
            # 读取dcm数据，对应source
            v16 = vtk.vtkDICOMImageReader()
            v16.SetDataByteOrderToLittleEndian()
            v16.SetDirectoryName(sig_dcm_path)
            v16.SetDataSpacing(1.0, 1.0, 1.0)
            v16.Update()
            # 利用封装好的MC算法抽取等值面，对应filter
            marchingCubes = vtk.vtkMarchingCubes()
            marchingCubes.SetInputConnection(v16.GetOutputPort())
            marchingCubes.SetValue(0, -10)
            marchingCubes.Update()
            # 剔除旧的或废除的数据单元，提高绘制速度，对应filter
            Stripper = vtk.vtkStripper()
            Stripper.SetInputConnection(marchingCubes.GetOutputPort())
            Stripper.Update()
            # 平滑滤波
            SmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter()
            SmoothPolyDataFilter.SetInputConnection(Stripper.GetOutputPort())
            # SmoothPolyDataFilter.SetRelaxationFactor(0.05)
            SmoothPolyDataFilter.SetRelaxationFactor(0.5)
            PolyDataNormals = vtk.vtkPolyDataNormals()
            PolyDataNormals.SetInputConnection(SmoothPolyDataFilter.GetOutputPort())
            # 将模型输出到STL文件
            STLWriter = vtk.vtkSTLWriter()
            STLWriter.SetFileName(filename.__str__())
            STLWriter.SetInputConnection(PolyDataNormals.GetOutputPort())
            STLWriter.Write()

    print("所有的单个牙齿stl重建完成!!!!")


if __name__ == "__main__":
    os.makedirs(args.test_output, exist_ok=True)
    device = torch.device(f"cuda:{args.device}")
    print("device:", device)
    dataset = TestDataset(args)

    dcm3d, slices_dcm = dataset.get_attribute()
    test_loader = DataLoader(dataset, batch_size=1,
                             num_workers=args.n_threads, pin_memory=False)
    if args.net_name == "Unet":
        net = Unet(in_channels=1, n_labels=args.n_labels).to(device)
    elif args.net_name == "Unet3D":
        net = Unet3D(in_channels=1, n_labels=args.n_labels).to(device)
    else:
        net = Nested_UNet(in_channels=1, n_labels=args.n_labels,
                          deepsupervision=args.dsv).to(device)

    model_pth = os.path.join(args.save_path, args.net_name, args.loss, args.optimizer)
    print("model_pth:", model_pth)
    ckpt = torch.load("{}\\best_model.pth".format(model_pth), map_location=device)
    net.load_state_dict(ckpt["net"])
    print("Model loaded！")


    out_dcm_path = os.path.join(args.test_output, "out_dcm")
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = batch["image"].to(device, dtype=torch.float32)
        dcm_data = np.array(batch["dcm_data"].squeeze(0))
        name = batch["name"][0]
        masks = test(net, image)
        img_name_to_ext = name.split(".")[0]

        # 保存预测输出的png(已弃用，不保存pred的图片了，直接处理)
        # output_img_dir = os.path.join(args.test_output, img_name_to_ext)
        # os.makedirs(output_img_dir, exist_ok=True)

        if args.n_labels == 1:
            # image_idex = Image.fromarray((masks * 255).astype(np.uint8))
            # image_idex.save(os.path.join(output_img_dir, img_name_to_ext))
            masks.squeeze()
            masks[masks == 1] = 255
            masks[masks == 0] = 1
            masks[masks == 255] = 0
        else:
            for idx in range(0, len(masks)):
                # img_name_idx = img_name_to_ext + "_" + str(idx) + ".png"
                # image_idx = Image.fromarray((masks[idx] * 255).astype(np.uint8))
                # image_idx.save(os.path.join(output_img_dir, img_name_idx))

                # 对照dcm，输出每个牙齿的stl
                mask = masks[idx]
                if idx == 0:
                    # 如果是0背景类需要将0变成1，1变成0，然后乘以原dcm的data
                    mask[mask == 1] = 255
                    mask[mask == 0] = 1
                    mask[mask == 255] = 0

                    new_dcm_data = (mask * dcm_data).astype("int16")
                    out_sig_dcm_path = os.path.join(out_dcm_path, f"{idx}")
                    os.makedirs(out_sig_dcm_path, exist_ok=True)

                    slice_ori_dcm =pydicom.read_file(os.path.join(args.test_data, name))
                    slice_ori_dcm.PixelData = new_dcm_data.tobytes()
                    slice_ori_dcm.save_as(os.path.join(out_sig_dcm_path, name))

                else:
                    new_dcm_data = (mask * dcm_data).astype("int16")
                    out_sig_dcm_path = os.path.join(out_dcm_path, f"{idx}")
                    os.makedirs(out_sig_dcm_path, exist_ok=True)

                    slice_ori_dcm = pydicom.read_file(os.path.join(args.test_data, name))
                    slice_ori_dcm.PixelData = new_dcm_data.tobytes()
                    slice_ori_dcm.save_as(os.path.join(out_sig_dcm_path, name))

        # 是否进行彩色打印
        if args.colors:
            output_img_dir = os.path.join(args.test_output, img_name_to_ext)
            os.makedirs(output_img_dir, exist_ok=True)
            img = np.array(image[0, 0].cpu().detach()).astype(np.uint8)
            colors = get_colors(args.n_labels)
            w, h = img.shape
            image_mask = np.zeros([h, w, 3], np.uint8)
            for idx in range(0, len(masks)):
                image_idx = Image.fromarray((masks[idx] * 255).astype(np.uint8))
                array_image = np.asarray(image_idx)
                image_mask[np.where(array_image == 255)] = colors[idx]
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            image_mask = cv2.cvtColor((np.asarray(image_mask)), cv2.COLOR_RGB2BGR)
            output = cv2.addWeighted(img, 0.7, image_mask, 0.3, 0)
            color_name = img_name_to_ext + ".png"
            cv2.imwrite(os.path.join(output_img_dir, color_name), output)

    print("----------------第三步：重建每个牙齿的dcm ==> stl--------------------")
    # 如何将dcm得预测输出直接保存为stl(整体得牙齿stl,单个类别的牙齿stl)
    dcm_to_stl(out_dcm_path, out_stl_path=args.test_output, n_labels=args.n_labels)
    shutil.rmtree(out_dcm_path)

    torch.cuda.empty_cache()
