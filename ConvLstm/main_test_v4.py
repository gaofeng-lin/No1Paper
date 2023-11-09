#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2020/03/09
@Author  :   jhhuang96
@Mail    :   hjh096@126.com
@Version :   1.0
@Description:   
'''






import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from net_params import convlstm_encoder_params_3224, convlstm_decoder_params_3224
from net_params import convlstm_encoder_params_80_20, convlstm_decoder_params_80_20

from data.mm import MovingMNIST
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
import sys
from earlystopping import EarlyStopping
from tqdm import tqdm
import numpy as np
# from tensorboardX import SummaryWriter
import argparse
from torch.utils.tensorboard import SummaryWriter


import pickle


from torch.utils.data import Dataset

from openstl.api import BaseExperiment
from openstl.utils import create_parser


import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from os.path import join
from os import listdir
import math
import struct
from scipy import stats
import datetime
from collections import namedtuple
import subprocess

import time
from multiprocessing import Pool
import random

# 在main_test_v3的基础上，额外预测一部分数据，来替代小块边界处的数据
# -----------------------

TestParams = namedtuple('TestParams', ['testStart', 'test_valSample', 'totalStep', 'save_model_name', 'isTest'])

def process_file(file_info):
    data_pathLow, i, dimLow = file_info
    dlow = getBData(data_pathLow + 'RESU' + '{:04d}'.format(i) + '.DAT', dimLow)
    dul = dlow[1]
    dul = 2 * (dul - np.min(dul)) / (np.max(dul) - np.min(dul)) - 1
    dul = dul.reshape(dimLow[1] + 1, dimLow[0] + 1).transpose()
    return dul[0:dimLow[0], 0:dimLow[1]]

def getBData_acceleration(dataPath, dim):
    num_entries = (dim[0]+1)*(1+dim[1])
    value = np.zeros([4, num_entries])
    
    with open(dataPath, 'rb') as res:
        for i in range(num_entries):
            _ = res.read(4)
            data = res.read(16)  # Read 4 floats at once
            values = struct.unpack('4f', data)
            for j, v in enumerate(values):
                value[j][i] = v
            _ = res.read(4)
    
    return value


def getBData(dataPath,dim):
    value=np.zeros([4,(dim[0]+1)*(1+dim[1])])
    index=0
    res=open(dataPath,'rb')
    for i in range((dim[0]+1)*(1+dim[1])):
        _=res.read(4)
        n1=res.read(4)
        n2=res.read(4)
        n3=res.read(4)
        n4=res.read(4)
        _=res.read(4)
        v1=struct.unpack('1f',n1)
        v2=struct.unpack('1f',n2)
        v3=struct.unpack('1f',n3)
        v4=struct.unpack('1f',n4)
        
        value[0][index]=float(v1[0])
        value[1][index]=float(v2[0])
        value[2][index]=float(v3[0])
        value[3][index]=float(v4[0])
        # print(i," ",value[0][index]," ",value[1][index]," ",value[2][index]," ",value[3][index])
        index=index+1
    res.close()
    return value


class ScalarDataSet(Dataset):
    def __init__(self, args):
        # dataset 是数据集的名字，这里就取'flowDM'
        # scale 是放大倍数。 比如我这里输入的是 480和960的数据集，它们是2倍关系，那么scale就是2
        # f 选取训练数据的比例。一般取0.6，这里总共有500个数据，0.6就是300个。
        # crop 直接取yes
        # croptimes 一般取4，和训练次数有关
        self.dataset = args.dataset
        self.scale = args.scale
        self.f = args.f
        self.crop = args.crop
        self.croptimes = args.croptimes
        # self.group = args.group
        self.start = args.start
        self.end = args.end
        self.totalStep = args.totalStep
        if self.dataset == 'HR':
            self.dim = [480,720,120]
            self.total_samples = 100
            self.cropsize = [30,40,15]
            self.data_path = 'Data/jet_hr_'                
        elif self.dataset == 'H':
            self.dim = [600,248,248]
            self.total_samples = 100
            self.cropsize = [32,32,32]
            self.data_path = 'Data/GT-'
        elif self.dataset == 'Vortex':
            self.dim = [128,128,128]
            self.total_samples = 90
            self.cropsize = [16,16,16]
            self.data_path = 'Data/vorts'
        elif self.dataset == 'flowDM':
            self.dimLow = [480,120]
            self.dimHigh = [960,240]       # self.dimHigh = [1920,480]
            self.total_samples = 500
            self.cropsize = [32,32]
            # self.data_pathLow = './weno_data/weno3/DM480/'#'../../../../nvme0/dx/dl/weno_code/weno3/DM480/'
            self.data_pathLow = './weno_data/weno5/DM480/'
            self.data_pathHigh ='./weno_data/weno3/DM960/' #'../../../../nvme0/dx/dl/weno_code/weno3/DM1920/'

        # 加载数据
        self.features, self.labels = self.ReadDataV2()
        # self.features, self.labels = self.read_data_test()

    # 加载测试数据，不要划分，留给后面的函数来划分
    def read_data_test(self):
        file_infos = [(self.data_pathLow, self.start + j + i, self.dimLow) 
                      for j in range(self.end) 
                      for i in range(self.totalStep)]

        with Pool(processes=8) as pool:
            results = pool.map(process_file, file_infos)

        # 调整数组维度以匹配数据结构
        results = np.asarray(results).reshape(self.end, self.totalStep, 1, self.dimLow[0], self.dimLow[1])

        # 直接划分特征和标签
        feature = results[:, :10]
        label = results[:, 10:]
        
        
        # 这个时候不要转变为tensor，转变后也是在cpu上
        # feature = torch.FloatTensor(feature)
        # label = torch.FloatTensor(label)   

        # 均匀的切割为80*20的图片
        # feature = split_data_v2(feature)
        # label = split_data_v2(label)

        return feature, label    

    def ReadDataV2(self):
        file_infos = [(self.data_pathLow, self.start + j + i, self.dimLow) 
                      for j in range(self.end) 
                      for i in range(self.totalStep)]

        with Pool(processes=8) as pool:
            results = pool.map(process_file, file_infos)

        # 调整数组维度以匹配数据结构
        results = np.asarray(results).reshape(self.end, self.totalStep, 1, self.dimLow[0], self.dimLow[1])

        # 直接划分特征和标签
        feature = results[:, :10]
        label = results[:, 10:]
        # feature = torch.FloatTensor(feature)
        # label = torch.FloatTensor(label)

        # print('feature shape is : ', feature.shape)

        # # 对数据进行32*32大小的滑动窗口进行切割
        # input_patches = sliding_window_v2(feature)
        # target_patches = sliding_window_v2(label)
        # print('input_patches shape: ', input_patches.shape)

        # selected_feature_patches, selected_label_patches = select_random_patches(input_patches, target_patches)
        # print('selected_feature_patches shape : ', selected_feature_patches.shape)


        # 要进行维度的合并，合并到第一维上面去


        return feature, label

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        inputVar = self.features[idx]
        targetVar = self.labels[idx]
        return inputVar, targetVar
        
def getDataLoader(dataType, setpStart, test_valSample, totalStep, isTest):
    args = argparse.Namespace()
    args.dataset='flowDM'
    args.crop='yes'
    args.scale=2
    args.f=0.6
    args.croptimes=4
    args.totalStep = totalStep
    args.isTest = isTest
    if dataType == "test":
        if isTest == 0:
            #下面的两行参数是用来测试流程的
            args.start = 20
            args.end = 3
        else:
            args.start = setpStart
            args.end = test_valSample
    
    if dataType == "train":
        if isTest == 0:
            #下面的两行参数是用来测试流程的
            args.start = 20
            args.end = 3
        
        else:
            args.start = setpStart
            args.end = test_valSample * 3
        #下面的两行参数是用来测试流程的
        # args.start = 20
        # args.end = 3

    if dataType == "val":
        if isTest == 0:
            #下面的两行参数是用来测试流程的
            args.start = 20
            args.end = 3
        else:
            args.start = setpStart
            args.end = test_valSample
 
        #下面的两行参数是用来测试流程的
        # args.start = 30
        # args.end = 3
    
    DateSet = ScalarDataSet(args)
    data_loader = DataLoader(dataset=DateSet, batch_size=1, shuffle=True)
    # torch.save(data_loader,'/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/test_60.npy')
    return data_loader



# 这个代码是用于测试模型的，把图片均匀的切割为32*32，多余的部分补充8
def split_dataFotTestv2(images, block_size=(32, 32)):
    total_num, seq_length, channels, height, width = images.shape
    
    # new_height = height // block_size[0] * block_size[0]
    # new_width = (width // block_size[1] - 1) * block_size[1]
    
    images_proper = images[:, :, :, :, :(width // block_size[1]) * block_size[1]]
    images_last_column = images[:, :, :, :, (width // block_size[1]) * block_size[1] - 8:]
    
    
    height_splits = height // block_size[0]
    width_splits = width // block_size[1]
    
    images_proper = images_proper.reshape(total_num, seq_length, channels, height_splits, block_size[0], width_splits, block_size[1])
    images_last_column = images_last_column.reshape(total_num, seq_length, channels, height_splits, block_size[0], 1, block_size[1])
    
    # print('images_proper shape is :', images_proper.shape)
    
    # images_proper = images_proper.transpose(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    
    images_proper = images_proper.permute(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    images_last_column = images_last_column.permute(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    

    
    return images_proper, images_last_column



# 这个代码是用于测试模型的，把图片均匀的切割为32*32，
def split_dataFotTest(images, block_size=(32, 32)):
    total_num, seq_length, channels, height, width = images.shape
    
    # new_height = height // block_size[0] * block_size[0]
    # new_width = (width // block_size[1] - 1) * block_size[1]
    
    images_proper = images[:, :, :, :, :(width // block_size[1]) * block_size[1]]
    images_last_column = images[:, :, :, :, (width // block_size[1]) * block_size[1]:]
    
    
    height_splits = height // block_size[0]
    width_splits = width // block_size[1]
    
    images_proper = images_proper.reshape(total_num, seq_length, channels, height_splits, block_size[0], width_splits, block_size[1])
    images_last_column = images_last_column.reshape(total_num, seq_length, channels, height_splits, block_size[0], -1)
    
    # print('images_proper shape is :', images_proper.shape)
    
    # images_proper = images_proper.transpose(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    
    images_proper = images_proper.permute(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    images_last_column = images_last_column.permute(0, 1, 2, 3, 5, 4).reshape(total_num, seq_length, channels, -1, block_size[0], width - (width_splits * block_size[0]))
    
    print('images_proper shape is :', images_proper.shape)
    print('images_last_column is :', images_last_column.shape)
    
    # images = np.concatenate((images_proper. images_last_column), axis=5)
    
    # images = images.transpose(0, 1, 2, 3, 5, 4, 6)
    
    # images = images.reshape(total_num, seq_length, channels, -1, block_size[0], images.shape[-1])
    
    return images_proper, images_last_column

# 把32*32的图片重构起来
def reconstruct_thirtytwo(images_32_32, images_32_24, height_splits, width_splits, height_final):

    
    
    # 先把images_32_32的图片拼接为480*96
    images_32_32 = reconstruct_data(images_32_32, (480, 96))
    
    
    # 把images_32_24改变形状，从15,32,32还原为15,32,24，再变为480,24
    
    images_32_24 = images_32_24[:, :, :, :, :, 8:]
    print('images_32_24 shape:', images_32_24.shape)
    
    total_num, seq_length, channels, n_patches, patch_h, patch_w = images_32_24.shape
    images_32_24 = torch.reshape(images_32_24, (total_num, seq_length, channels, n_patches * patch_h, patch_w))
    
    # 再把480*24拼接在480*96的下方。
    reassembled_images = torch.cat((images_32_32, images_32_24), dim=-1)
    
    print('reassembled_images shape:', reassembled_images.shape)
    return reassembled_images

# 均匀的方式切割数据，切割大小为80*20。新增的功能是合并第0维和第3维
def split_data_v2(flowData, patch_h=80, patch_w=20):
    # patches = []
    print('flowData shape : ', flowData.shape)
    batch_size, seq_length, channels, h, w = flowData.shape
    n_h = h // patch_h
    n_w = w // patch_w
    
    # 调整数据形状以形成图像块
    images = flowData.view(batch_size, seq_length, channels, n_h, patch_h, n_w, patch_w)
    images = images.permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, seq_length, channels, patch_h, patch_w)

    return images

# 切割大小为32*32，不均匀边界处记录索引。新增的功能是合并第0维和第3维
def split_data_v3(flowData, patch_h=32, patch_w=32):
    print('flowData shape : ', flowData.shape)
    # flowData.shape torch.Size([1, 10, 1, 480, 120])
    batch_size, seq_length, channels, h, w = flowData.shape
    
    images = []
    overlap_indices = []
    
    for i in range(0, h, patch_h):
        for j in range(0, w, patch_w):
            end_i = min(i+patch_h, h)
            end_j = min(j+patch_w, w)
            
            # 如果达到边界，则计算重叠的起始索引
            if end_i != i+patch_h or end_j != j+patch_w:
                start_i = max(0, end_i - patch_h)
                start_j = max(0, end_j - patch_w)
                overlap_indices.append((start_i, start_j))
            else:
                start_i = i
                start_j = j
            
            patch = flowData[:, :, :, start_i:end_i, start_j:end_j]
            images.append(patch)
    
    images = torch.cat(images, dim=0)
    return images, overlap_indices

# 切割大小为32*32，不均匀边界处记录索引。新增的功能是合并第0维和第3维
def split_data_v3_v2(flowData, patch_w=32, patch_h=32, height_ranges=[(0, 31), (32, 63), (64, 95), (88, 119)]):
    print('flowData shape: ', flowData.shape)
    batch_size, seq_length, channels, w, h = flowData.shape
    
    images = []

    for i in range(0, w, patch_w):
        for start_j, end_j in height_ranges:
            if end_j > h :
                continue
            patch = flowData[:, :, :, i:i+patch_w, start_j:end_j+1]
            images.append(patch)

    images = torch.cat(images, dim=0)

    return images

# 切割大小为32*32，宽度的其实位置发生了变化，具体为：15-46，47-78，79-110
# 这部分数据是为了替换均匀32*32预测不好的边界数据
def split_data_v4(flowData, patch_h=32, patch_w=32, width_ranges=[(15, 46), (47, 78), (79, 110)]):
    print('flowData shape: ', flowData.shape)
    batch_size, seq_length, channels, h, w = flowData.shape
    
    images = []
    
    for start_j, end_j in width_ranges:
        for i in range(0, h, patch_h):
            end_i = min(i + patch_h, h)
            for j in range(start_j, end_j, patch_w):
                end_j = min(j + patch_w, w)
                
                patch = flowData[:, :, :, i:end_i, j:end_j]
                images.append(patch)
    
    images = torch.cat(images, dim=0)
    return images

def split_data_v5(flowData, patch_h=32, patch_w=32, height_ranges=[(15, 46), (47, 78), (79, 110)]):
    print('flowData shape: ', flowData.shape)
    batch_size, seq_length, channels, h, w = flowData.shape
    
    images = []

    flowData = flowData.reshape(-1, 10, 1,  32, 120)

    for start_j, end_j in height_ranges:
        patch = flowData[:, :, :, :, start_j: end_j+1]
        images.append(patch)
    images = torch.cat(images, dim=0)

    return images

def split_data_v6(flowData, patch_w=32, patch_h=32, height_ranges=[(15, 46), (47, 78), (79, 110)]):
    print('flowData shape: ', flowData.shape)
    batch_size, seq_length, channels, w, h = flowData.shape
    
    images = []
    for i in range(0, w, patch_w):
        for start_j, end_j in height_ranges:
            if end_j > h :
                continue
            patch = flowData[:, :, :, i:i+patch_w, start_j:end_j+1]
            images.append(patch)

    images = torch.cat(images, dim=0)

    return images

def merge_data_v6(patches, original_shape=(1, 10, 1, 480, 96), patch_w=32, patch_h=32, height_ranges=[(15, 46), (47, 78), (79, 110)]):
    # Initialize the tensor to hold the merged data
    merged_data = torch.zeros(original_shape)
    
    # Calculate how many patches fit horizontally
    patches_per_row = original_shape[3] // patch_w

    # Iterate over each patch
    for k, patch in enumerate(patches):
        # Calculate the position of the patch
        row_idx = k // (patches_per_row * len(height_ranges))
        within_row_idx = k % (patches_per_row * len(height_ranges))
        col_idx = within_row_idx // len(height_ranges)
        height_idx = within_row_idx % len(height_ranges)
        start_j, end_j = height_ranges[height_idx]

        # Calculate the exact position where the patch should be placed
        i = row_idx * patch_h
        j = col_idx * patch_w

        # Place the patch back into its position
        merged_data[:, :, :, i:i+patch_h, j:j+patch_w] = patch

    return merged_data

# 均匀的方式切割数据，切割大小为80*20
def split_data(flowData, patch_h=80, patch_w=20):
    # patches = []
    print('flowData shape : ', flowData.shape)
    batch_size, seq_length, channels, h, w = flowData.shape
    n_h = h // patch_h
    n_w = w // patch_w
    
    images = flowData.reshape(batch_size, seq_length, channels, n_h, patch_h, n_w, patch_w)
    images = images.permute(0,1,2,3,5,4,6).reshape(batch_size, seq_length, channels, n_h*n_w, patch_h, patch_w)

    return images




def reconstruct_data(patch_data, original_shape):
    
    total_num, seq_length, channels, n_patches, patch_h, patch_w = patch_data.shape
    
    
    n_h = original_shape[0] // patch_h
    n_w = original_shape[1] // patch_w
    
    imgs = patch_data.view(total_num, seq_length, channels, n_h, n_w, patch_h, patch_w)
    imgs = imgs.permute(0,1,2,3,5,4,6)
    # imgs = imgs.shape(total_num, seq_length, channels, original_shape[0], original_shape[1])
    imgs = torch.reshape(imgs,(total_num, seq_length, channels, original_shape[0], original_shape[1]))
    
    return imgs

# 滑动窗口的形式切割数据，窗口大小为32*32，步长为2
# def sliding_window(data, patch_size=32, stride=2):
#     total_num, seq_length, channels, h, w = data.shape
#     output = []
#     for i in range(0, h - patch_size + 1, stride):
#         for j in range(0, w- patch_size + 1, stride):
#             patch = data[..., i:i+patch_size, j:j+patch_size]
#             output.append(patch)
#     return torch.stack(output, dim=3)

# 滑动窗口的形式切割数据，窗口大小为32*32，步长为2
def sliding_window_v2(data, patch_size=32, stride=2):
    if data.is_cuda:
        data = data.cpu()
    
    # 将tensor转换为numpy array
    data_np = data.numpy()

    
    total_num, seq_length, channels, h, w = data_np.shape
    output = [] 
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w- patch_size + 1, stride):
            patch = data_np[..., i:i+patch_size, j:j+patch_size]
            output.append(patch)
    output_np = np.stack(output, axis=3)
    return torch.from_numpy(output_np)

# 这个函数是配合sliding_window一起使用。从对应的标签和特征中随机挑选185个小块（这些小块也是对应的）
def select_random_patches(feature_patches, label_patches, num_patches=185):
    total_num, seq_length, channels, split_num, h, w = feature_patches.shape
    indices = torch.randperm(split_num)[:num_patches]
    selected_feature_patches = feature_patches[:, :, :, indices, :, :]
    selected_label_patches = label_patches[:, :, :, indices, :, :]
    return selected_feature_patches, selected_label_patches

# 把80*20的小块还原为480*120
def merge_patches(images, original_h=480, original_w=120, patch_h=80, patch_w=20):
    batch_size, seq_length, channels, patch_h, patch_w = images.shape
    n_h = original_h // patch_h
    n_w = original_w // patch_w
    
    # 首先将batch_size维度分解为原始尺寸
    images = images.reshape(1, seq_length, channels, n_h, n_w, patch_h, patch_w)
    
    # 重新排列张量以匹配原始图片的形状
    images = images.permute(0, 1, 2, 3, 5, 4, 6).reshape(1, seq_length, channels, original_h, original_w)
    
    return images

# 把32*32的图像还原为480*120，不均匀部分单独出来
def merge_patches_v2(images, overlap_indices, original_h=480, original_w=120, patch_h=32, patch_w=32):
    _, seq_length, channels, _, _ = images.shape
    merged_image = torch.zeros((1, seq_length, channels, original_h, original_w), dtype=images.dtype, device=images.device)
    patch_idx = 0
    
    for i in range(0, original_h, patch_h):
        for j in range(0, original_w, patch_w):
            end_i = min(i+patch_h, original_h)
            end_j = min(j+patch_w, original_w)
            
            # 如果块是重叠的，则使用其索引位置
            if (i, j) in overlap_indices:
                start_i = max(0, end_i - patch_h)
                start_j = max(0, end_j - patch_w)
            else:
                start_i = i
                start_j = j
            
            if patch_idx < images.shape[0]:
                merged_image[:, :, :, start_i:end_i, start_j:end_j] = images[patch_idx, :, :, :end_i-start_i, :end_j-start_j]
                patch_idx += 1
    
    return merged_image

# 和merge_patches_v2相比，对重叠的区域取平均值，而不是舍弃其中一部分
def merge_patches_v3(images, overlap_indices, original_h=480, original_w=120, patch_h=32, patch_w=32):
    _, seq_length, channels, _, _ = images.shape
    merged_image = torch.zeros((1, seq_length, channels, original_h, original_w), dtype=images.dtype, device=images.device)
    overlap_counter = torch.zeros((1, seq_length, channels, original_h, original_w), dtype=torch.int32, device=images.device)
    patch_idx = 0
    
    for i in range(0, original_h, patch_h):
        for j in range(0, original_w, patch_w):
            end_i = min(i + patch_h, original_h)
            end_j = min(j + patch_w, original_w)
            
            if (i, j) in overlap_indices:
                start_i = max(0, end_i - patch_h)
                start_j = max(0, end_j - patch_w)
            else:
                start_i = i
                start_j = j
            
            if patch_idx < images.shape[0]:
                merged_image[:, :, :, start_i:end_i, start_j:end_j] += images[patch_idx, :, :, :end_i - start_i, :end_j - start_j]
                overlap_counter[:, :, :, start_i:end_i, start_j:end_j] += 1
                patch_idx += 1
    
    # 避免除以零
    overlap_counter[overlap_counter == 0] = 1
    # 计算重叠区域的平均值
    merged_image = merged_image / overlap_counter.float()
    
    return merged_image

def merge_patches_v3_v2(images, original_w=480, original_h=120, patch_w=32, patch_h=32):

    # images ([60, 10, 1, 32, 32])
    print('images shape : ', images.shape)
    _, seq_length, channels, _, _ = images.shape
    merged_image = torch.zeros((1, seq_length, channels, original_w, original_h), dtype=images.dtype, device=images.device)
    patch_idx = 0
    
    for i in range(0, original_w, patch_w):
        for j in range(0, original_h, patch_h):
            # end_i = min(i+patch_w, original_w)
            end_i = i+patch_w
            # end_j = min(j+patch_h, original_h)
            end_j = j+patch_h
            
            start_i = i
            start_j = j
            
            # if patch_idx < images.shape[0]:
            if patch_idx % 4 == 3:
                merged_image[:, :, :, start_i:end_i, start_j-8:end_j-8] = images[patch_idx, :, :, :, :]
                patch_idx += 1
            else:    
                merged_image[:, :, :, start_i:end_i, start_j:end_j] = images[patch_idx, :, :, :, :]
                patch_idx += 1
    
    return merged_image


def merge_patches_v6(images, original_w=480, original_h=96, patch_w=32, patch_h=32):

    # images ([45, 10, 1, 32, 32])
    print('images shape : ', images.shape)
    _, seq_length, channels, _, _ = images.shape
    merged_image = torch.zeros((1, seq_length, channels, original_w, original_h), dtype=images.dtype, device=images.device)
    patch_idx = 0
    
    for i in range(0, original_w, patch_w):
        for j in range(0, original_h, patch_h):
            end_i = min(i+patch_w, original_w)
            end_j = min(j+patch_h, original_h)
            
            start_i = i
            start_j = j
            
            if patch_idx < images.shape[0]:
                merged_image[:, :, :, start_i:end_i, start_j:end_j] = images[patch_idx, :, :, :, :]
                patch_idx += 1

    # for i in range(0, original_w, patch_w):
    #     for j in range(0, original_h, patch_h):
    #         end_i = min(i+patch_w, original_w)
    #         end_j = min(j+patch_h, original_h)
            
    #         start_i = i
    #         start_j = j
            
    #         if patch_idx < images.shape[0]:
    #             merged_image[:, :, :, start_i:end_i, start_j:end_j] = images[patch_idx, :, :, :end_i-start_i, :end_j-start_j]
    #             patch_idx += 1
    
    return merged_image

def test(params):

    testStart, test_valSample, totalStep, save_model_name, isTest = params.testStart, params.test_valSample, params.totalStep, params.save_model_name, params.isTest
    testLoader = getDataLoader("test", testStart, test_valSample, totalStep, isTest)


    # testLoader = torch.load('/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/train_180.npy')
    '''
    test function to test the model
    '''
    # 输入尺寸是80*20-----------------------------------------
    # encoder_params = convlstm_encoder_params_80_20
    # decoder_params = convlstm_decoder_params_80_20
    # 输入尺寸是80*20-----------------------------------------

    # # 输入尺寸是32*32-----------------------------------------
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
    # # 输入尺寸是32*32-----------------------------------------

    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    
    model = ED(encoder, decoder)
    
    

    # model_info = torch.load(current_path +'/save_model/' + save_model_name + '/' + output)
    model_info = torch.load('./save_model/4_tmp/checkpoint_44_0.000028.pth.tar')
    model.load_state_dict(model_info['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    
    criterion = nn.MSELoss().cuda()
    
    
    # test_loss = 0
    test_loss = []
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    TIMESTAMP = formatted_time
    save_model_name = TIMESTAMP
    
    print('start up test....................')
    with torch.no_grad():    
        # t = tqdm(testLoader, leave=False, total=len(testLoader))
        for i, (inputs, labels) in enumerate(testLoader):
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            model.eval()


            # 把真值可视化出来-----------------------------------------------------
            # print('labels shape : ', labels.shape)
            # for t in range(labels.shape[1]):
            #     outputs_subset_data = labels[0, t, 0, :, :]

            #     # 构造文件名
          
            #     # 写入.dat文件
            #     with open('/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/convlstm/labels.dat', 'w') as f:
            #         f.write('variables="x","y","u"\n')
            #         f.write('ZONE T="Small Rect", I=480 J=120\n')
            #         for y in range(0, 120):
            #             for x in range(0, 480):
            #                 f.write(f'{x} {y} {outputs_subset_data[x, y].item()}\n')
            #     breakpoint()


            # 把真值可视化出来-----------------------------------------------------

            # 对数据进行切割，按照32*32切割-----------------------------------
            input_patch  = split_data_v3_v2(inputs)
            label_patch  = split_data_v3_v2(labels)
            # 对数据进行切割，按照32*32切割-----------------------------------

            # r_images shape :  torch.Size([45, 10, 1, 32, 32])
            r_images_inputs = split_data_v6(inputs)
            r_images = model(r_images_inputs)

            # 画出r_images-------------------------------------------------
            # r_images = r_images.reshape(3,10,1,480,32)
            # r_images = r_images.reshape(-1,10,1,32,96)
            # print('r_images shape : ', r_images.shape)
            # for t in range(r_images.shape[1]):
            #     outputs_subset_data = r_images[0, t, 0, :, :]

            #     # 构造文件名
          
            #     # 写入.dat文件
            #     with open('/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/convlstm/r_images.dat', 'w') as f:
            #         f.write('variables="x","y","u"\n')
            #         f.write('ZONE T="Small Rect", I=32 J=96\n')
            #         for y in range(0, 96):
            #             for x in range(0, 32):
            #                 f.write(f'{x} {y} {outputs_subset_data[x, y].item()}\n')
            #     breakpoint()
                # 画出r_images-------------------------------------------------

            # 对数据进行切割，按照80*20切割-----------------------------------
            # input_patch = split_data_v2(inputs)
            # label_patch= split_data_v2(labels)
            # 对数据进行切割，按照80*20切割-----------------------------------
            
        
            losses = []
            
            pred_all_32_32 = []
            pred_all_32_24 = []
            loss_split_all = 0
            # split_num = input_patches.shape[3]
            # breakpoint()
            # inputs = input_patch.to(device)
            # labels  = label_patch.to(device)
            print('inputs shape : ', inputs.shape)
            pred = model(input_patch)
            print('pred shape : ', pred.shape)
          

            # # 把32*32的小块重构为480*120-----------------------------------------
            reconstruct_pred  = merge_patches_v3_v2(pred)
            reconstruct_label = merge_patches_v3_v2(label_patch)
            # # 把32*32的小块重构为480*120-----------------------------------------

            # 把80*20的小块重构为480*120-----------------------------------------
            # reconstruct_pred  = merge_patches(pred)
            # reconstruct_label = merge_patches(label_patch)
            # 把80*20的小块重构为480*120-----------------------------------------
            

            



          

            
            # loss_split_all /=  split_num

            current_path = os.getcwd()
            # 把预测值存入文件夹中
            save_outputs_directory = current_path + '/save_outputs_file/' + save_model_name
            if not os.path.exists(save_outputs_directory):
                os.makedirs(save_outputs_directory)

            # 把真实值存入文件夹中
            save_groundtruth_directory = current_path + '/save_groundtruth_file/' + save_model_name
            if not os.path.exists(save_groundtruth_directory):
                os.makedirs(save_groundtruth_directory)

            # 把差值存入文件夹中
            save_diff_directory = current_path + '/save_diff_file/' + save_model_name
            if not os.path.exists(save_diff_directory):
                os.makedirs(save_diff_directory)

            # 定义替换的宽度范围
            # original_width_ranges = [(29, 33), (61, 65), (93, 97)]
            # r_image_width_ranges = [(14, 18), (46, 50), (78, 82)]

            original_width_ranges = [(30, 33), (62, 65), (94, 97)]
            r_image_width_ranges = [(15, 18), (47, 50), (79, 82)]
            replace_patch_w = 5  # 替换的小块宽度

            r_images = merge_patches_v6(r_images)
            # 假设 reconstruct_pred 的形状是 torch.Size([1, 10, 1, 480, 120])
            for t in range(reconstruct_pred.shape[1]):  # 遍历每个时间步
                # outputs_subset_data = reconstruct_pred[0, t, 0, :, :]
                labels_subset_data  = reconstruct_label[0, t, 0, :, :]

                # 创建一个副本以便修改
                reconstructed_data = reconstruct_pred.clone()
                

                # 方法二-------------------------------------------------------------------------------
                
                # print('r_images shape : ', r_images.shape)
                # breakpoint()
                for (original_start_w, original_end_w), (r_image_start_w, r_image_end_w) in zip(original_width_ranges, r_image_width_ranges):
                    print('reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] shape : ', reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1].shape)
                    print('r_images[:, t, 0, :, r_image_start_w:r_image_end_w+1] shape : ', r_images[:, t, 0, :, r_image_start_w:r_image_end_w+1].shape)
                    reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] = r_images[:, t, 0, :, r_image_start_w:r_image_end_w+1]
                print('reconstructed_data shape: ', reconstructed_data.shape)
                reconstructed_data = reconstructed_data[0, t, 0, :, :]
                # 方法二-------------------------------------------------------------------------------


                # 方法1,对应实验22---------------------------------------------------------------
                # reconstructed_data = reconstructed_data.reshape(-1, 10, 1, 32, 120)

                # r_images = r_images.reshape(-1,10,1,32,96)

                # # 遍历替换的宽度范围
                # for (original_start_w, original_end_w), (r_image_start_w, r_image_end_w) in zip(original_width_ranges, r_image_width_ranges):
                #     print('reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] shape : ', reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1].shape)
                #     reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] = r_images[:, t, 0, :, r_image_start_w:r_image_end_w+1]

                # reconstructed_data = reconstructed_data.reshape(1, 10, 1, 480, 120)[0, t, 0, :, :]
                # 方法1---------------------------------------------------------------

                # 方法3，方法1的基础上平移---------------------------------------------------------------
                
                # reconstructed_data = reconstructed_data.reshape(-1, 10, 1, 32, 120)

                # r_images = r_images.reshape(-1,10,1,32,96)
                # height_shift = 5

                # # 遍历替换的宽度范围
                # for (original_start_w, original_end_w), (r_image_start_w, r_image_end_w) in zip(original_width_ranges, r_image_width_ranges):
                #     print('reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] shape : ', reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1].shape)
                #     reconstructed_data[:, t, 0, :, original_start_w:original_end_w+1] = r_images[:, t, 0, :, r_image_start_w:r_image_end_w+1]

                    
                
                # reconstructed_data = reconstructed_data.reshape(1, 10, 1, 480, 120)

                # reconstructed_data = circular_shift(reconstructed_data, 30, 34, 5)
                # reconstructed_data = reconstructed_data[0, t, 0, :, :]

                # 方法3---------------------------------------------------------------
                        
                       




                # 构造文件名
                file_name = os.path.join(save_outputs_directory, f'{i}_{t}.dat')
                # 写入.dat文件
                with open(file_name, 'w') as f:
                    f.write('variables="x","y","u"\n')
                    f.write('ZONE T="Small Rect", I=480 J=120\n')
                    for y in range(0, 120):
                        for x in range(0, 480):
                            f.write(f'{x} {y} {reconstructed_data[x, y].item()}\n')


                file_name = os.path.join(save_groundtruth_directory, f'{i}_{t}.dat')
                with open(file_name, 'w') as f:
                    f.write('variables="x","y","u"\n')
                    f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                    for y in range(0, 120):
                        for x in range(0, 480):
                            f.write(f'{x} {y} {labels_subset_data[x, y]}\n')



                file_name = os.path.join(save_diff_directory, f'{i}_{t}.dat')
                with open(file_name, 'w') as f:
                    f.write('variables="x","y","u"\n')
                    f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                    for y in range(0, 120):
                        for x in range(0, 480):
                            f.write(f'{x} {y} {abs(reconstructed_data[x, y] - labels_subset_data[x, y])}\n')
                
            breakpoint()

            with open(f'{current_path}/save_outputs_file/{save_model_name}/{i}.dat', 'w') as f:
                f.write('variables="x","y","u"\n')
                f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                for y in range(0, 120):
                    for x in range(0, 480):
                        f.write(f'{x} {y} {outputs_subset_data[x, y]}\n')
            
            
            
            # 把真实值存入文件夹中
            save_groundtruth_directory = current_path + '/save_groundtruth_file/' + save_model_name
            if not os.path.exists(save_groundtruth_directory):
                os.makedirs(save_groundtruth_directory)
            with open(f'{current_path}/save_groundtruth_file/{save_model_name}/{i}.dat', 'w') as f:
                f.write('variables="x","y","u"\n')
                f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                for y in range(0, 120):
                    for x in range(0, 480):
                        f.write(f'{x} {y} {labels_subset_data[x, y]}\n')
            
            
            
            # 把差值存入文件夹中
            save_diff_directory = current_path + '/save_diff_file/' + save_model_name
            if not os.path.exists(save_diff_directory):
                os.makedirs(save_diff_directory)
            
            with open(f'{current_path}/save_diff_file/{save_model_name}/{i}.dat', 'w') as f:
                f.write('variables="x","y","u"\n')
                f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                for y in range(0, 120):
                    for x in range(0, 480):
                        f.write(f'{x} {y} {abs(outputs_subset_data[x, y] - labels_subset_data[x, y])}\n')
                
                # loss = criterion(pred, label)
                # tmp_loss += loss
            # one_epoch_loss = tmp_loss/
            
            # outputs = model(inputs)
            
            # print('outputs shape is :', outputs.shape)
            # print('labels is :', labels)
            
            
            # labels_subset_data = labels[0, 0, 0, :, :]
            # with open(f'/home/nvme0/lgf/TianCi/ConvLSTM-PyTorch/save_groundtruth_file/{i}.dat', 'w') as f:
                # f.write('variables="x","y","u"\n')
                # f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                # for y in range(0, 120):
                    # for x in range(0, 480):
                        # f.write(f'{x} {y} {labels_subset_data[x, y]}\n')
            
            
            # 
            # outputs_subset_data = outputs[0, 0, 0, :, :]

            

            
            # print('outputs shape is :', outputs.shape)
            # print('labels shape is :', labels.shape)
            # loss = criterion(outputs, labels)
            
            # print('i count is :', i)
            
            
            # test_loss += loss_split_all
        # test_loss /= len(testLoader)
        
    # print("Test Loss: ", test_loss)
    print("test is over!")
    

# 定义执行循环平移的函数
def circular_shift(tensor, start_idx, end_idx, shift_amount):
    # 提取完整的宽度
    complete_width = tensor.shape[-1]

    # 提取需要平移的切片
    slice_to_shift = tensor[..., start_idx:end_idx]

    # 执行循环平移
    shifted_slice = torch.roll(slice_to_shift, shifts=-shift_amount, dims=-1)

    # 创建一个与原始张量形状相同的新张量用于存放结果
    result_tensor = torch.zeros_like(tensor)

    # 复制原始张量中不变的部分
    result_tensor[..., :start_idx] = tensor[..., :start_idx]
    result_tensor[..., end_idx:] = tensor[..., end_idx:]

    # 插入平移后的部分，考虑环绕
    wrap_around = end_idx - complete_width
    if wrap_around < 0:  # 不需要环绕
        result_tensor[..., start_idx-shift_amount:end_idx-shift_amount] = shifted_slice
    else:  # 需要环绕
        result_tensor[..., start_idx-shift_amount:complete_width] = shifted_slice[..., :wrap_around]
        result_tensor[..., :wrap_around] = shifted_slice[..., wrap_around:]

    return result_tensor

if __name__ == "__main__":

    # trainStep = int(input("请输入特征步数："))
    # predStep = int(input("请输入标签步数："))

    trainStep = 10
    predStep = 10
    
    
    save_model_name = str(trainStep)+'Pred'+str(predStep)
    print('save_model_name is:', save_model_name)
    # isTest = int(input("测试流程输入0，否则输入1："))

    isTest = 1
    
    
    totalStep = trainStep + predStep
    
    if isTest == 0:
        trainStart = valStart = test_valSample = testStart = 1
    
    else:
        # trainSample = int(input("请输入训练样本的数目："))
        trainSample = 180
        test_valSample = int(trainSample/3)
        
        testStart = 500 - totalStep + 1 - test_valSample + 1
        valStart = testStart - 1 - totalStep + 1 - test_valSample + 1
        trainStart = valStart - 1 - totalStep + 1 - trainSample + 1
   
    
        

    # user_input = input("输入1进行训练，输入2进行测试, 输入q退出：")
    user_input = "2"

    params = TestParams(testStart, test_valSample, totalStep, save_model_name, isTest)
    test(params)

