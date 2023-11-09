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

# 这段代码是为了测试模型，测试的是10步到20步的预测



import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from encoder import Encoder
from decoder import Decoder
from model import ED
from net_params import convlstm_encoder_params, convlstm_decoder_params, convgru_encoder_params, convgru_decoder_params
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


# -----------------------

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



class ScalarDataSet():
    def __init__(self,args):
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
            self.data_pathLow = '/home/nvme0/lgf/transformDat/DM480/'#'../../../../nvme0/dx/dl/weno_code/weno3/DM480/'
            self.data_pathHigh ='/home/nvme0/lgf/transformDat/DM960/' #'../../../../nvme0/dx/dl/weno_code/weno3/DM1920/'


        self.samples = int(self.total_samples*self.f)

    def ReadDataV2(self):
        self.low = []
        tmp = []
        # low是一个（1,480,120）的数组，全是0元素
        # low = np.zeros((1,self.dimLow[0],self.dimLow[1]))
        for j in range(0, self.end):
            low = np.zeros((1,self.dimLow[0],self.dimLow[1]))
            for i in range(self.start+j, self.start+30+j):
                # 得到的dlow 是一个二维数组， 维度为(4,481*121)
                dlow = getBData(self.data_pathLow +'RESU'+'{:04d}'.format(i)+'.DAT',self.dimLow)
                dul=dlow[1]
                dul = 2*(dul-np.min(dul))/(np.max(dul)-np.min(dul))-1
                
                dul = dul.reshape(self.dimLow[1]+1,self.dimLow[0]+1).transpose()
                # print('reshape 后的 dul shape: ', dul.shape)
                # low[0] 是 [480, 120]
                low[0] = dul[0:self.dimLow[0],0:self.dimLow[1]]
                # low1 = dul[0:160,0:self.dimLow[1]]
                # low2 = dul[160:320,0:self.dimLow[1]]
                # low3 = dul[320:480,0:self.dimLow[1]]
                # low4 = np.array([low1, low2, low3])
                            
                # print('low[0] shape is :', np.array(low[0].shape))
                # print('low shape is :', np.array(low.shape))
                # print('low1 shape is :', low1.shape)
                # print('low2 shape is :', low2.shape)
                # print('low3 shape is :', low3.shape)                
                # print('low4 shape is :', low4.shape)
                # low = low.tolist()
                # self.low.append(low4)
                self.low.append(low)
                # tmp.append(low)
            
            tmp.append(self.low)
            self.low = []
        self.low = np.asarray(tmp)
        # self.low = [:, :10]
        # self.high = [:, 20:]
        
        feature = self.low[:, :10]
        label = self.low[:, 10:]
        
        feature = torch.FloatTensor(feature)
        label = torch.FloatTensor(label)
        dataset = torch.utils.data.TensorDataset(feature,label)
        train_loader = DataLoader(dataset=dataset,batch_size=4, shuffle=True)
        # print('feature shape is ', np.array(feature).shape)
        # print('label shape is ', np.array(label).shape)
        return train_loader


            




def getDataLoader(dataType):
    args = argparse.Namespace()
    args.dataset='flowDM'
    args.crop='yes'
    args.scale=2
    args.f=0.6
    args.croptimes=4
    if dataType == "test":
        args.start = 412
        args.end = 60
        
        #下面的两行参数是用来测试流程的
        # args.start = 402
        # args.end = 5
        testDateSet = ScalarDataSet(args)
        data_loader = testDateSet.ReadDataV2()
    if dataType == "train":
        args.start = 82
        args.end = 240
        
        #下面的两行参数是用来测试流程的
        # args.start = 20
        # args.end = 10
        DateSet = ScalarDataSet(args)
        data_loader = DateSet.ReadDataV2()
    if dataType == "val":
        args.start = 322
        args.end = 80
 
        #下面的两行参数是用来测试流程的
        # args.start = 30
        # args.end = 10
        DateSet = ScalarDataSet(args)
        data_loader = DateSet.ReadDataV2()
    return data_loader
        


# -----------------------

current_time = datetime.datetime.now()

formatted_time = current_time.strftime('%Y-%m-%d-%H-%M-%S')

TIMESTAMP = formatted_time

# TIMESTAMP = "2020-03-09T00-00-00"
parser = argparse.ArgumentParser()
parser.add_argument('-clstm',
                    '--convlstm',
                    help='use convlstm as base cell',
                    action='store_true')
parser.add_argument('-cgru',
                    '--convgru',
                    help='use convgru as base cell',
                    action='store_true')
parser.add_argument('--batch_size',
                    default=4,
                    type=int,
                    help='mini-batch size')
parser.add_argument('-lr', default=1e-4, type=float, help='G learning rate')
parser.add_argument('-frames_input',
                    default=10,
                    type=int,
                    help='sum of input frames')
parser.add_argument('-frames_output',
                    default=20,
                    type=int,
                    help='sum of predict frames')
parser.add_argument('-epochs', default=100, type=int, help='sum of epochs')
args = parser.parse_args()

random_seed = 1996
np.random.seed(random_seed)
torch.manual_seed(random_seed)
if torch.cuda.device_count() > 1:
    torch.cuda.manual_seed_all(random_seed)
else:
    torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




testLoader = getDataLoader("test")

# ---------------------------------------------------------------- 这一块是随机生成的数据，主要是为了检验下流程行不行
# batch = 16
# sequence_length = 10
# num_channels = 1
# height = 64
# width = 64

# feature = torch.randn(batch, sequence_length, num_channels, height, width)
# labels = torch.randn(batch, 20, num_channels, height, width)


# dataset = torch.utils.data.TensorDataset(feature,labels)
# trainLoader = DataLoader(dataset=dataset,batch_size=4, shuffle=True)

# validLoader = DataLoader(dataset=dataset,batch_size=4, shuffle=True)
# -------------------------------------------------------------------------


encoder_params = convlstm_encoder_params
decoder_params = convlstm_decoder_params


def train():
    '''
    main function to test the model
    '''
  

    encoder = Encoder(encoder_params[0], encoder_params[1]).cuda()
    decoder = Decoder(decoder_params[0], decoder_params[1]).cuda()
    
    model_info = torch.load('./save_model/2023-07-24-10-03-08/checkpoint_69_0.000019.pth.tar')
    
    model = ED(encoder, decoder)
    
    
    model.load_state_dict(model_info['state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    model.eval()
    
    lossfunction = nn.MSELoss().cuda()
    
    criterion = lossfunction
    
    test_loss = 0
    
    print('start up test....................')
    with torch.no_grad():    
        for i, (inputs, labels) in enumerate(testLoader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            

            
            
            labels_subset_data = labels[0, 0, 0, :, :]
            with open(f'/home/nvme0/lgf/TianCi/ConvLSTM-PyTorch/save_groundtruth_file/2023-07-25-12-54-28/{i}.dat', 'w') as f:
                f.write('variables="x","y","u"\n')
                f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                for y in range(0, 120):
                    for x in range(0, 480):
                        f.write(f'{x} {y} {labels_subset_data[x, y]}\n')
            
            
            # 
            outputs_subset_data = outputs[0, 0, 0, :, :]
            # with open(f'/home/nvme0/lgf/TianCi/ConvLSTM-PyTorch/save_outputs_file/{i}.dat', 'w') as f:
                # f.write('variables="x","y","u"\n')
                # f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                # for y in range(0, 120):
                    # for x in range(0, 480):
                        # f.write(f'{x} {y} {outputs_subset_data[x, y]}\n')
                        
            # with open(f'/home/nvme0/lgf/TianCi/ConvLSTM-PyTorch/save_diff_file/2023-07-25-12-54-28/{i}.dat', 'w') as f:
                # f.write('variables="x","y","u"\n')
                # f.write('    ZONE T="Small Rect", I= 480 J= 120\n')
                # for y in range(0, 120):
                    # for x in range(0, 480):
                        # f.write(f'{x} {y} {abs(outputs_subset_data[x, y] - labels_subset_data[x, y])}\n')
            
            # print('outputs shape is :', outputs.shape)
            # print('labels shape is :', labels.shape)
            loss = criterion(outputs, labels)
            
            # print('i count is :', i)
            
            
            test_loss += loss.item()
        test_loss /= len(testLoader)
        
    print("Test Loss: ", test_loss)
    
    
    



if __name__ == "__main__":
    train()
