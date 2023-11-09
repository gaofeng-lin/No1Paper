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

# 这段代码是基于mainv5.py，主要的修改在于把数据按照32*32切割后
# 每个epoch都从里面选一部分，而不是完全固定。


# 数据提取了训练集和验证集，暂时没有提取测试集，测试集会单独写一个脚本来运行
# 训练集从（114-123）-（124-143）为一组，直到（293-302）-（303-322）结束

# 180个训练、验证和测试各60

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
from net_params import convlstm_encoder_params_3224, convlstm_decoder_params_3224
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


from torch.utils.data import Dataset, SubsetRandomSampler

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

# -----------------------

TrainParams = namedtuple('TrainParams', ['trainStart', 'valStart', 'test_valSample', 'totalStep', 'save_model_name', 'isTest'])

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
            self.data_pathLow = './weno_data/weno3/DM480/'#'../../../../nvme0/dx/dl/weno_code/weno3/DM480/'
            self.data_pathHigh ='./weno_data/weno3/DM960/' #'../../../../nvme0/dx/dl/weno_code/weno3/DM1920/'

        # 加载数据
        self.features, self.labels = self.ReadDataV2()

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
        feature = torch.FloatTensor(feature)
        label = torch.FloatTensor(label)

        print('feature shape is : ', feature.shape)

        # 对数据进行32*32大小的滑动窗口进行切割
        input_patches = sliding_window(feature)
        target_patches = sliding_window(label)
        

        # 进行维度的合并，把小块数量合并到样本

        input_patches = dim_reshape(input_patches)
        target_patches = dim_reshape(target_patches)
        print('input_patches shape: ', input_patches.shape)


        return input_patches, target_patches

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # print('getitem中的self.features', self.features.shape)
        inputVar = self.features[idx]
        targetVar = self.labels[idx]

        # print('getitem中的inputvar', inputVar.shape)

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
    
    # DateSet = ScalarDataSet(args)
    # data_loader = DataLoader(dataset=DateSet, batch_size=32, shuffle=False)
    # # torch.save(data_loader,'/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/test_60.npy')
    # return data_loader

    DateSet = ScalarDataSet(args)
    # data_loader = DataLoader(dataset=DateSet, batch_size=32, shuffle=False)
    # torch.save(data_loader,'/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/test_60.npy')
    return DateSet
        

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
    
    # print('images_proper shape is :', images_proper.shape)
    # print('images_last_column is :', images_last_column.shape)
    
    # images = np.concatenate((images_proper. images_last_column), axis=5)
    
    # images = images.transpose(0, 1, 2, 3, 5, 4, 6)
    
    # images = images.reshape(total_num, seq_length, channels, -1, block_size[0], images.shape[-1])
    
    # print()
    
    return images_proper, images_last_column

# 这个代码是用于测试模型的，把图片均匀的切割为32*32，
def split_dataFotTest(images, block_size=(32, 32)):
    total_num, seq_length, channels, height, width = images.shape
    
    
    images_proper = images[:, :, :, :, :(width // block_size[1]) * block_size[1]]
    images_last_column = images[:, :, :, :, (width // block_size[1]) * block_size[1]:]
    
    
    height_splits = height // block_size[0]
    width_splits = width // block_size[1]
    
    images_proper = images_proper.reshape(total_num, seq_length, channels, height_splits, block_size[0], width_splits, block_size[1])
    images_last_column = images_last_column.reshape(total_num, seq_length, channels, height_splits, block_size[0], -1)


    images_proper = images_proper.permute(0, 1, 2, 3, 5, 4, 6).reshape(total_num, seq_length, channels, -1, block_size[0], block_size[0])
    images_last_column = images_last_column.permute(0, 1, 2, 3, 5, 4).reshape(total_num, seq_length, channels, -1, block_size[0], width - (width_splits * block_size[0]))
    
    print('images_proper shape is :', images_proper.shape)
    print('images_last_column is :', images_last_column.shape)
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

# 均匀的方式切割数据，切割大小为80*20
def split_data(flowData, patch_h=80, patch_w=20):
    # patches = []
    batch_size, seq_length, channels, h, w = flowData.shape
    n_h = h // patch_h
    n_w = w // patch_w
    
    images = flowData.view(batch_size, seq_length, channels, n_h, patch_h, n_w, patch_w)
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

def sliding_window(data, patch_size=32, stride=2):
    # 如果数据在GPU上，将其移动到CPU
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

# 相比于第一版，把得到的小块数量合并到样本总数
def sliding_window_v2(data, patch_size=32, stride=2):
    # 如果数据在GPU上，将其移动到CPU
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

# 把得到的小块数量合并到样本总数
def dim_reshape(tensor):

    # 获取tensor的形状
    bs, timesteps, channels, num_patches, height, width = tensor.shape

    # 重新排列维度顺序
    tensor_permuted = tensor.permute(3, 0, 1, 2, 4, 5)

    # 合并第0维和第1维
    tensor_reshaped = tensor_permuted.reshape(bs * num_patches, timesteps, channels, height, width)   

    return tensor_reshaped

# 这个函数是配合sliding_window一起使用。从对应的标签和特征中随机挑选185个小块（这些小块也是对应的）
def select_random_patches(feature_patches, label_patches, num_patches=185):
    total_num, seq_length, channels, split_num, h, w = feature_patches.shape
    indices = torch.randperm(split_num)[:num_patches]
    selected_feature_patches = feature_patches[:, :, :, indices, :, :]
    selected_label_patches = label_patches[:, :, :, indices, :, :]

    # print('selected_feature_patches shape: ', selected_feature_patches.shape)

    selected_feature_patches = dim_reshape(selected_feature_patches)
    selected_label_patches = dim_reshape(selected_label_patches)


    return selected_feature_patches, selected_label_patches

def create_data_loader(dataset, num_samples, batch_size, shuffle=True):
    # 获取数据集大小
    dataset_size = len(dataset)

    # 创建所有可能索引的列表
    indices = list(range(dataset_size))

    # 如果需要打乱数据，则打乱索引
    if shuffle:
        np.random.shuffle(indices)

    # 获取所需数量的样本索引
    selected_indices = indices[:num_samples]

    # 创建Sampler
    sampler = SubsetRandomSampler(selected_indices)

    # 创建DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return data_loader


# 这个函数是每个epoch的时候，从数据集中随机选择一部分
def create_data_two_loaders(dataset, num_samples_train, num_samples_val, batch_size):
    # 获取数据集大小
    dataset_size = len(dataset)

    # 创建所有可能索引的列表
    indices = list(range(dataset_size))

    # 打乱索引
    np.random.shuffle(indices)

    # 根据需要的训练集和验证集的样本数目，分割索引列表
    train_indices = indices[:num_samples_train]
    val_indices = indices[num_samples_train:num_samples_train + num_samples_val]

    # 创建Sampler
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    # 创建训练集和验证集的DataLoader
    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader


def train(params):
    '''
    main function to run the training
    '''
    
    '''
    1. 加载数据集
    '''
    
    trainStart, valStart, test_valSample, totalStep, save_model_name, isTest = params.trainStart, params.valStart, params.test_valSample, params.totalStep, params.save_model_name, params.isTest
    
    print('trainStart: ', trainStart)
    print('valStart: ', valStart)
    print('test_valSample: ', test_valSample)
    print('totalStep: ', totalStep)
    print('isTest: ', isTest)
    
    # trainLoader = getDataLoader("train", trainStart, test_valSample, totalStep, isTest)
    

    # validLoader = getDataLoader("val", valStart, test_valSample, totalStep, isTest)

    # trainLoader = torch.load('/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/train_180.npy')
    # validLoader = torch.load('/home/bingxing2/gpuuser638/lgf/tianchi/time_dimension_enhancement/data_base/weno3/DM480/10pred10/val_60.npy')

    train_dataset = getDataLoader("train", trainStart, test_valSample, totalStep, isTest)
    valid_dataset = getDataLoader("val", valStart, test_valSample, totalStep, isTest)
    
    train_num_samples = 21600
    val_num_samples   = 7200
    batch_size        = 128




    '''
    2. 初始化训练参数
    '''
    current_time = datetime.datetime.now()

    formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')
    
    # TIMESTAMP = save_model_name
    TIMESTAMP = formatted_time
    
    # TIMESTAMP = "2020-03-09T00-00-00"
    parser = argparse.ArgumentParser()
    parser.add_argument('-clstm',
                        '--convlstm',
                        help='use convlstm as base cell',
                        action='store_true')
    # parser.add_argument('-cgru',
    #                     '--convgru',
    #                     help='use convgru as base cell',
    #                     action='store_true')
    parser.add_argument('--batch_size',
                        default=4,
                        type=int,
                        help='mini-batch size')
    parser.add_argument('-lr', default=1e-3, type=float, help='G learning rate')
    parser.add_argument('-frames_input',
                        default=10,
                        type=int,
                        help='sum of input frames')
    parser.add_argument('-frames_output',
                        default=5,
                        type=int,
                        help='sum of predict frames')
    parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
    args = parser.parse_args()
    
    print('参数初始化完毕')


    random_seed = 1996
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(random_seed)
    else:
        torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    save_dir = './save_model/' + TIMESTAMP
    
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params
    
    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    
    print('encoder_params[0], encoder_params[1]:', encoder_params[0], encoder_params[1])
    
    print('decoder_params[0], decoder_params[1]:', decoder_params[0], decoder_params[1])
    net = ED(encoder, decoder)
    run_dir = './runs/' + TIMESTAMP
    if not os.path.isdir(run_dir):
        os.makedirs(run_dir)
    tb = SummaryWriter(run_dir)
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=20, verbose=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        # net = nn.DataParallel(net)
        net = nn.DataParallel(net)
        # print("我可以使用多gpu训练...........")
    net.to(device)
    # 载入模型
    model_info = torch.load('./save_model/4_tmp/4_tmp.pth.tar')
    net.load_state_dict(model_info['state_dict'])
    if os.path.exists(os.path.join(save_dir, 'checkpoint.pth.tar')):
        # load existing model
        print('==> loading existing model')
        model_info = torch.load(os.path.join(save_dir, 'checkpoin.pth.tar'))
        net.load_state_dict(model_info['state_dict'])
        optimizer = torch.optim.Adam(net.parameters())
        optimizer.load_state_dict(model_info['optimizer'])
        cur_epoch = model_info['epoch'] + 1
    else:
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        cur_epoch = 0
    lossfunction = nn.MSELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    pla_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                      factor=0.5,
                                                      patience=4,
                                                      verbose=True)

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    # mini_val_loss = np.inf
    # first_batch = next(iter(trainLoader))
    # print('first_batch is: ',first_batch)
    for epoch in range(cur_epoch, args.epochs + 1):
        trainLoader = create_data_loader(train_dataset, train_num_samples, batch_size)
        validLoader = create_data_loader(valid_dataset, val_num_samples, batch_size)
        losses = []
        t = tqdm(trainLoader, leave=False, total=len(trainLoader))
        for i, (input_patch, target_patch) in enumerate(t):
            inputs = input_patch.to(device)  # B,S,C,H,W
            label  = target_patch.to(device)  # B,S,C,H,W
            optimizer.zero_grad()
            net.train()
            print('train inputs shape : ', inputs.shape)
            # print('label shape : ', label.shape)
            pred = net(inputs)  # B,S,C,H,W
            loss = lossfunction(pred, label) 
            loss_aver = loss.item() / args.batch_size
            losses.append(loss_aver)
            loss.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value=10.0)
            optimizer.step()
        train_loss_aver = sum(losses) / len(losses)
        train_losses.append(train_loss_aver)
        t.set_postfix({
            'trainloss': '{:.6f}'.format(loss_aver),
            'epoch': '{:02d}'.format(epoch)
        })
        tb.add_scalar('TrainLoss', train_loss_aver, epoch)
        ######################
        # validate the model #
        ######################
        print('我要开始验证模型了')
        with torch.no_grad():
            net.eval()
            t = tqdm(validLoader, leave=False, total=len(validLoader))
            # for i, (idx, targetVar, inputVar, _, _) in enumerate(t):
            losses = []
            for i, (inputVar, targetVar) in enumerate(t):
                if i == 3000:
                    break
                # input_patches = sliding_window(inputVar)
                # target_patches = sliding_window(targetVar)
                

                inputs = inputVar.to(device)  # B,S,C,H,W
                label  = targetVar.to(device)  # B,S,C,H,W
                pred = net(inputs)  # B,S,C,H,W
                # pred = reconstruct_data(pred_patches, img_size=inputVar.shape[2:])
                loss = lossfunction(pred, label) 
                loss_aver = loss.item() / args.batch_size
                losses.append(loss_aver)
            
            
            # record validation loss
            valid_loss_aver = sum(losses) / len(losses)
            valid_losses.append(valid_loss_aver)
            #print ("validloss: {:.6f},  epoch : {:02d}".format(loss_aver,epoch),end = '\r', flush=True)
            t.set_postfix({
                'validloss': '{:.6f}'.format(loss_aver),
                'epoch': '{:02d}'.format(epoch)
            })

        tb.add_scalar('ValidLoss', valid_loss_aver, epoch)
        torch.cuda.empty_cache()
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(args.epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{args.epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.6f} ' +
                     f'valid_loss: {valid_loss:.6f}')

        print(print_msg)
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        pla_lr_scheduler.step(valid_loss)  # lr_scheduler
        model_dict = {
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        early_stopping(valid_loss.item(), model_dict, epoch, save_dir)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    with open("avg_train_losses.txt", 'wt') as f:
        for i in avg_train_losses:
            print(i, file=f)

    with open("avg_valid_losses.txt", 'wt') as f:
        for i in avg_valid_losses:
            print(i, file=f)


def test(params):
    '''
    test function to test the model
    '''
    encoder_params = convlstm_encoder_params
    decoder_params = convlstm_decoder_params

    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    
    model = ED(encoder, decoder)
    
    # 载入模型
    model_info = torch.load('./save_model/4_tmp/4_tmp.pth.tar')
    model.load_state_dict(model_info['state_dict'])
    
    
    testStart, test_valSample, totalStep, save_model_name, isTest = params.testStart, params.test_valSample, params.totalStep, params.save_model_name, params.isTest
    
    current_path = os.getcwd()
    
    os.chdir('./save_model/'+save_model_name)
    command = 'ls -1 checkpoint_*.pth.tar | sort -t "_" -k3 -n | head -1'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    out, err = process.communicate()
    # output = out.decode('utf-8').strip()
    output = 'checkpoint_70_0.000000.pth.tar'
    print('output is: ',output)
    
    # model_info = torch.load(current_path +'/save_model/' + save_model_name + '/' + output)
    
    
    
    
    # model.load_state_dict(model_info['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    

    
    lossfunction = nn.MSELoss().cuda()
    
    criterion = lossfunction
    
    # test_loss = 0
    test_loss = []
    
    testLoader = getDataLoader("test", testStart, test_valSample, totalStep, isTest)
    
    print('start up test....................')
    with torch.no_grad():    
        for i, (inputs, labels) in enumerate(testLoader):
            # inputs = inputs.to(device)
            # labels = labels.to(device)
            
            
            # 对输入的数据进行切割，按照80*20的大小
            # input_patches = split_data(inputs)
            # target_patches = split_data(labels)
            
            # 对输入的数据进行切割，按照32*32的大小，一部分特殊的输32*24
            input_patches_32_32,  input_patches_32_24= split_dataFotTestv2(inputs)
                
            target_patches_32_32,  target_patches_32_24= split_dataFotTestv2(labels)
            
            
            losses = []
            
            pred_all_32_32 = []
            pred_all_32_24 = []
            loss_split_all = 0
            # split_num = input_patches.shape[3]
            # breakpoint()
            for input_patch, target_patch in zip(input_patches_32_32.unbind(dim=3), target_patches_32_32.unbind(dim=3)):
                inputs = input_patch.to(device)
                label = target_patch.to(device)
                
                pred = model(inputs)
                # loss = criterion(pred, label)
                # loss_split_all += loss
                # losses.append(loss_aver)

                pred_all_32_32.append(pred)
            
 
            for input_patch, target_patch in zip(input_patches_32_24.unbind(dim=3), target_patches_32_24.unbind(dim=3)):
                inputs = input_patch.to(device)
                label = target_patch.to(device)
                
                pred = model(inputs)
              

                pred_all_32_24.append(pred)
            
            # breakpoint()
            
            pred_all_32_32_tensor = torch.stack(pred_all_32_32, dim = 3)
            pred_all_32_24_tensor = torch.stack(pred_all_32_24, dim = 3)
            
            # breakpoint()
            reconstruct_pred = reconstruct_thirtytwo(pred_all_32_32_tensor, pred_all_32_24_tensor, 15, 3, 15)
            # breakpoint()
            
            # reconstruct_pred = reconstruct_data(pred_all_tensor)
            
            outputs_subset_data = reconstruct_pred[0, 0, 0, :, :]
            labels_subset_data = labels[0, 0, 0, :, :]
            
            # breakpoint()
            
            # loss_split_all /=  split_num
            save_model_name = '32_32_10Pred5'
            # 把预测值存入文件夹中
            save_outputs_directory = current_path + '/save_outputs_file/' + save_model_name
            if not os.path.exists(save_outputs_directory):
                os.makedirs(save_outputs_directory)
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
   
    
        

    
    while True:
        # user_input = input("输入1进行训练，输入2进行测试, 输入q退出：")
        user_input = "1"
        if user_input == "1":
            params = TrainParams(trainStart, valStart, test_valSample, totalStep, save_model_name, isTest)
            print('准备进入训练函数')
            train(params)
        elif user_input == "2":
            # testModel = 
            params = TestParams(testStart, test_valSample, totalStep, save_model_name, isTest)
            test(params)
        elif user_input.lower() == "q":
            break
        else:
            print("无效输入")
