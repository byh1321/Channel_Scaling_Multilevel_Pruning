import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import progress_bar

import os
import argparse
#import VGG16

import struct
import random

parser = argparse.ArgumentParser(description='load and make new network')
parser.add_argument('--mode', default=0, type=int, help='mode 1 -> for 0.125, mode 2 -> for 0.25, mode 3 -> for full channel')
parser.add_argument('--block1', default='NULL', help='input block1 ckpt name', metavar="FILE")
parser.add_argument('--block2', default='NULL', help='input block2 ckpt name', metavar="FILE")
parser.add_argument('--block3', default='NULL', help='input block3 ckpt name', metavar="FILE")
parser.add_argument('--outputfile', default='NULL', help='output file name', metavar="FILE")


use_cuda = torch.cuda.is_available()
args = parser.parse_args()

class CNN(nn.Module):
	def __init__(self):
		super(CNN,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3,64,3,padding=1,bias=False), #layer0
			nn.BatchNorm2d(64), # batch norm is added because dataset is changed
			nn.ReLU(inplace=True),
		)
		self.conv2 = nn.Sequential(
			nn.Conv2d(64,64,3,padding=1, bias=False), #layer3
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
		)
		self.maxpool1 = nn.Sequential(
			nn.MaxPool2d(2,2), # 16*16* 64
		)
		self.conv3 = nn.Sequential(
			nn.Conv2d(64,128,3,padding=1, bias=False), #layer7
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.conv4 = nn.Sequential(
			nn.Conv2d(128,128,3,padding=1, bias=False),#layer10
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
		)
		self.maxpool2 = nn.Sequential(
			nn.MaxPool2d(2,2), # 8*8*128
		)
		self.conv5 = nn.Sequential(
			nn.Conv2d(128,256,3,padding=1, bias=False), #layer14
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv6 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer17
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.conv7 = nn.Sequential(
			nn.Conv2d(256,256,3,padding=1, bias=False), #layer20
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
		)
		self.maxpool3 = nn.Sequential(
			nn.MaxPool2d(2,2), # 4*4*256
		)
		self.conv8 = nn.Sequential(
			nn.Conv2d(256,512,3,padding=1, bias=False), #layer24
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv9 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer27
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv10 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer30
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool4 = nn.Sequential(
			nn.MaxPool2d(2,2), # 2*2*512
		)
		self.conv11 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer34
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv12 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer37
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.conv13 = nn.Sequential(
			nn.Conv2d(512,512,3,padding=1, bias=False), #layer40
			nn.BatchNorm2d(512),
			nn.ReLU(inplace=True),
		)
		self.maxpool5 = nn.Sequential(
			nn.MaxPool2d(2,2) # 1*1*512
		)
		self.fc1 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer1
			nn.ReLU(inplace=True),
		)
		self.fc2 = nn.Sequential(
			nn.Dropout(p=0.5),
			nn.Linear(512,512, bias=False), #fc_layer4
			nn.ReLU(inplace=True),
		)
		self.fc3 = nn.Sequential(
			nn.Linear(512,100, bias=False) #fc_layer6
		)

	def forward(self,x):
		isfixed = 0
		x = roundmax(x)
		out1 = self.conv1(x) # 1250*64*32*32
		
		if isfixed:
			out1 = quant(out1) 
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if isfixed:
			out2 = quant(out2)
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)
		out4 = self.conv3(out3) # 1250*128*16*16
		if isfixed:
			out4 = quant(out4) 
			out4 = roundmax(out4)
		out5 = self.conv4(out4) # 1250*128*16*16
		if isfixed:
			out5 = quant(out5) 
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)
		out7 = self.conv5(out6) # 1250*256*8*8
		if isfixed:
			out7 = quant(out7) 
			out7 = roundmax(out7)
		out8 = self.conv6(out7) # 1250*256*8*8
		if isfixed:
			out8 = quant(out8) 
			out8 = roundmax(out8)
		out9 = self.conv7(out8) # 1250*256*8*8
		if isfixed:
			out9 = quant(out9) 
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)
		out11 = self.conv8(out10) # 1250*512*4*4
		if isfixed:
			out11 = quant(out11) 
			out11 = roundmax(out11)
		out12 = self.conv9(out11) # 1250*512*4*4
		if isfixed:
			out12 = quant(out12) 
			out12 = roundmax(out12)
		out13 = self.conv10(out12) # 1250*512*4*
		if isfixed:
			out13 = quant(out13) 
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*
		if isfixed:
			out15 = quant(out15) 
			out15 = roundmax(out15)
		out16 = self.conv12(out15) # 1250*512*2*
		if isfixed:
			out16 = quant(out16) 
			out16 = roundmax(out16)
		out17 = self.conv13(out16) # 1250*512*2*
		if isfixed:
			out17 = quant(out17) 
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)
		out20 = self.fc1(out19) # 1250*512
		if isfixed:
			out20 = quant(out20) 
			out20 = roundmax(out20)
		out21 = self.fc2(out20) # 1250*512
		if isfixed:
			out21 = quant(out21) 
			out21 = roundmax(out21)
		out22 = self.fc3(out21) # 1250*10
		if isfixed:
			out22 = quant(out22) 
			out22 = roundmax(out22)

		return out22

def roundmax(input):
	'''maximum = 2**iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)'''
	return input	

def quant(input):
	#input = torch.round(input / (2 ** (-aprec))) * (2 ** (-aprec))
	return input


def set_mask(block, val):
	if block == 0:
		mask[0][:,:,:,:] = val
		mask[1][:,:,:,:] = val 
		mask[2][:,:,:,:] = val 
		mask[3][:,:,:,:] = val
		mask[4][:,:,:,:] = val
		mask[5][:,:,:,:] = val
		mask[6][:,:,:,:] = val
		mask[7][:,:,:,:] = val
		mask[8][:,:,:,:] = val
		mask[9][:,:,:,:] = val
		mask[10][:,:,:,:] = val
		mask[11][:,:,:,:] = val
		mask[12][:,:,:,:] = val
		mask[13][:,:] = val 
		mask[14][:,:] = val 
		mask[15][:,:] = val 
	elif block == 1:
		for i in range(56):
			mask[0][i,:,:,:] = val
			mask[1][i,0:55,:,:] = val 
		for i in range(112):
			mask[2][i,0:55,:,:] = val 
			mask[3][i,0:111,:,:] = val
		for i in range(224):
			mask[4][i,0:111,:,:] = val
			mask[5][i,0:223,:,:] = val
			mask[6][i,0:223,:,:] = val
		for i in range(448):
			mask[7][i,0:223,:,:] = val
			mask[8][i,0:447,:,:] = val
			mask[9][i,0:447,:,:] = val
			mask[10][i,0:447,:,:] = val
			mask[11][i,0:447,:,:] = val
			mask[12][i,0:447,:,:] = val
			mask[13][i,0:447] = val 
			mask[14][i,0:447] = val 
		mask[15][:,0:447] = val 
	elif block == 2:
		for i in range(48):
			mask[0][i,:,:,:] = val
			mask[1][i,0:47,:,:] = val 
		for i in range(96):
			mask[2][i,0:47,:,:] = val 
			mask[3][i,0:95,:,:] = val
		for i in range(192):
			mask[4][i,0:95,:,:] = val
			mask[5][i,0:191,:,:] = val
			mask[6][i,0:191,:,:] = val
		for i in range(384):
			mask[7][i,0:191,:,:] = val
			mask[8][i,0:383,:,:] = val
			mask[9][i,0:383,:,:] = val
			mask[10][i,0:383,:,:] = val
			mask[11][i,0:383,:,:] = val
			mask[12][i,0:383,:,:] = val
			mask[13][i,0:383] = val 
			mask[14][i,0:383] = val 
		mask[15][:,0:383] = val 
	elif block == 3:
		for i in range(40):
			mask[0][i,:,:,:] = val
			mask[1][i,0:39,:,:] = val 
		for i in range(80):
			mask[2][i,0:39,:,:] = val 
			mask[3][i,0:79,:,:] = val
		for i in range(160):
			mask[4][i,0:79,:,:] = val
			mask[5][i,0:159,:,:] = val
			mask[6][i,0:159,:,:] = val
		for i in range(320):
			mask[7][i,0:159,:,:] = val
			mask[8][i,0:319,:,:] = val
			mask[9][i,0:319,:,:] = val
			mask[10][i,0:319,:,:] = val
			mask[11][i,0:319,:,:] = val
			mask[12][i,0:319,:,:] = val
			mask[13][i,0:319] = val 
			mask[14][i,0:319] = val 
		mask[15][:,0:319] = val 
	elif block == 4:
		for i in range(32):
			mask[0][i,:,:,:] = val
			mask[1][i,0:31,:,:] = val 
		for i in range(64):
			mask[2][i,0:31,:,:] = val 
			mask[3][i,0:63,:,:] = val
		for i in range(128):
			mask[4][i,0:63,:,:] = val
			mask[5][i,0:127,:,:] = val
			mask[6][i,0:127,:,:] = val
		for i in range(256):
			mask[7][i,0:127,:,:] = val
			mask[8][i,0:255,:,:] = val
			mask[9][i,0:255,:,:] = val
			mask[10][i,0:255,:,:] = val
			mask[11][i,0:255,:,:] = val
			mask[12][i,0:255,:,:] = val
			mask[13][i,0:255] = val 
			mask[14][i,0:255] = val 
		mask[15][:,0:255] = val 
	return mask

def net_mask_mul(net, mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.mul(param.data,mask[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.mul(param.data,mask[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.mul(param.data,mask[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.mul(param.data,mask[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.mul(param.data,mask[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.mul(param.data,mask[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.mul(param.data,mask[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.mul(param.data,mask[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.mul(param.data,mask[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.mul(param.data,mask[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.mul(param.data,mask[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.mul(param.data,mask[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.mul(param.data,mask[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.mul(param.data,mask[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.mul(param.data,mask[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.mul(param.data,mask[15])
	return net

def add_network(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,layer[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.add(param.data,layer[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.add(param.data,layer[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.add(param.data,layer[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.add(param.data,layer[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.add(param.data,layer[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.add(param.data,layer[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.add(param.data,layer[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.add(param.data,layer[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.add(param.data,layer[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.add(param.data,layer[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.add(param.data,layer[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.add(param.data,layer[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.add(param.data,layer[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.add(param.data,layer[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.add(param.data,layer[15])

def add_mask(net, mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.add(param.data,mask[0])
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.add(param.data,mask[1])
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.add(param.data,mask[2])
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.add(param.data,mask[3])
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.add(param.data,mask[4])
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.add(param.data,mask[5])
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.add(param.data,mask[6])
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.add(param.data,mask[7])
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.add(param.data,mask[8])
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.add(param.data,mask[9])
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.add(param.data,mask[10])
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.add(param.data,mask[11])
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.add(param.data,mask[12])

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.add(param.data,mask[13])
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.add(param.data,mask[14])
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.add(param.data,mask[15])

def save_network(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			layer[0] = param.data
	for child in net.children():
		for param in child.conv2[0].parameters():
			layer[1] = param.data		
	for child in net.children():
		for param in child.conv3[0].parameters():
			layer[2] = param.data		
	for child in net.children():
		for param in child.conv4[0].parameters():
			layer[3] = param.data		
	for child in net.children():
		for param in child.conv5[0].parameters():
			layer[4] = param.data	
	for child in net.children():
		for param in child.conv6[0].parameters():
			layer[5] = param.data
	for child in net.children():
		for param in child.conv7[0].parameters():
			layer[6] = param.data
	for child in net.children():
		for param in child.conv8[0].parameters():
			layer[7] = param.data
	for child in net.children():
		for param in child.conv9[0].parameters():
			layer[8] = param.data
	for child in net.children():
		for param in child.conv10[0].parameters():
			layer[9] = param.data
	for child in net.children():
		for param in child.conv11[0].parameters():
			layer[10] = param.data
	for child in net.children():
		for param in child.conv12[0].parameters():
			layer[11] = param.data
	for child in net.children():
		for param in child.conv13[0].parameters():
			layer[12] = param.data

	for child in net.children():
		for param in child.fc1[1].parameters():
			layer[13] = param.data
	for child in net.children():
		for param in child.fc2[1].parameters():
			layer[14] = param.data
	for child in net.children():
		for param in child.fc3[0].parameters():
			layer[15] = param.data

def getNonzeroPoints(net):
	mask = torch.load('mask_null.dat')
	mask[0] = (net.conv1[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[1] = (net.conv2[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[2] = (net.conv3[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[3] = (net.conv4[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[4] = (net.conv5[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[5] = (net.conv6[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[6] = (net.conv7[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[7] = (net.conv8[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[8] = (net.conv9[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[9] = (net.conv10[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[10] = (net.conv11[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[11] = (net.conv12[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[12] = (net.conv13[0].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[13] = (net.fc1[1].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[14] = (net.fc2[1].weight.data == 0).type(torch.FloatTensor).cuda()
	mask[15] = (net.fc3[0].weight.data == 0).type(torch.FloatTensor).cuda()
	return mask

def printweight(net):
	for child in net.children():
		for param in child.conv1[0].parameters():
			f = open('test_1.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv2[0].parameters():
			f = open('test_2.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convolution layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv3[0].parameters():
			f = open('test_3.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 3 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv4[0].parameters():
			f = open('test_4.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 4 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv5[0].parameters():
			f = open('test_5.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 5 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv6[0].parameters():
			f = open('test_6.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 6 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv7[0].parameters():
			f = open('test_7.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 7 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv8[0].parameters():
			f = open('test_8.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 8 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv9[0].parameters():
			f = open('test_9.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 9 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv10[0].parameters():
			f = open('test_10.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 10 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv11[0].parameters():
			f = open('test_11.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 11 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv12[0].parameters():
			f = open('test_12.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 12 weight printed')
			f.close()
	for child in net.children():
		for param in child.conv13[0].parameters():
			f = open('test_13.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('convoultion layer 13 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc1[1].parameters():
			f = open('test_fc1.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 1 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc2[1].parameters():
			f = open('test_fc2.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 2 weight printed')
			f.close()
	for child in net.children():
		for param in child.fc3[0].parameters():
			f = open('test_fc3.txt','w+')
			param_out = param.clone()
			param_out = param_out.view(1,-1)
			for i in range(0,param_out.size()[1]):
				print(param_out[0,i].data[0], file = f)
			print('fc layer 3 weight printed')
			f.close()

if __name__ == '__main__':
	if use_cuda:
		cudnn.benchmark = True

	mask = torch.load('mask_null.dat')
	mask_rand = torch.load('mask_rand.dat')
	layer = torch.load('mask_null.dat')
	try:
		checkpoint = torch.load('./checkpoint/'+args.block1)
		net1 = checkpoint['net']
		if use_cuda:
			net1.cuda() 
			net1 = torch.nn.DataParallel(net1, device_ids=range(0,8))
	except:
		print("Error : Failed to load net1. End program.")
		exit()

	try:
		if args.block2 == 'NULL':
			pass
		else:
			checkpoint = torch.load('./checkpoint/'+args.block2)
			net2 = checkpoint['net'] 
			if use_cuda:
				net2.cuda() 
				net2 = torch.nn.DataParallel(net2, device_ids=range(0,8))
	except:
		print("Error : Failed to load net2. End program.")
		exit()

	try:
		if args.block3 == 'NULL':
			pass
		else:
			checkpoint = torch.load('./checkpoint/'+args.block3)
			net3 = checkpoint['net'] 
			if use_cuda:
				net3.cuda() 
				net3 = torch.nn.DataParallel(net3, device_ids=range(0,8))
	except:
		print("Error : Failed to load net3. End program.")
		exit()
	
	#######################################################
	# load network, give randn values to nonzero param.
	#'''
	if args.mode == 1:
		mask_zero = getNonzeroPoints(net1)
		mask_rand = torch.load('mask_rand.dat')
		mask = torch.load('mask_null.dat')
		for i in range(16):
			mask[i] = mask_rand[i] * mask_zero[i]
		add_mask(net1, mask)
		
	#'''
	#######################################################

	#######################################################
	# load network, give randn values to nonzero param(group version).
	#'''
	if args.mode == 2:
		'''
		mask_nonzero = torch.load('./mask_60_4bunch_cifar100.dat')
		mask_zero = []
		for i in range(16):
			mask_zero.append((mask_nonzero[i] == 0).type(torch.FloatTensor).cuda())
		'''
		mask_rand = torch.load('mask_rand_100.dat')
		add_mask(net1, mask_rand)
		'''
		print(mask_zero[0][0])
		print(mask_rand[0][0])
		for child in net1.children():
			for param in child.conv1[0].parameters():
				print(param[0])
		'''

	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 08, gau 016
	#'''
	if args.mode == 4:
		mask = set_mask(3,1)
		#print(type(mask))
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(2,1)
		mask = set_mask(3,0)
		for i in range(16):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 08, gau 016 threshold check
	#'''
	if args.mode == 5:
		mask = set_mask(2,1)
		mask = set_mask(3,0)
		net1 = net_mask_mul(net1,mask)
	#'''
	#######################################################
	
	#######################################################
	#Enable this part for blur 10, gau 025
	#'''	
	if args.mode == 3:
		mask = set_mask(2,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(0,1)
		mask = set_mask(2,0)
		for i in range(16):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#Enable this part for blur 10, gau 025 threshold check
	#'''
	if args.mode == 6:
		mask = set_mask(0,1)
		mask = set_mask(2,0)
		net1 = net_mask_mul(net1,mask)
	#'''
	#######################################################

	#######################################################
	#'''	
	if args.mode == 7:
		mask = set_mask(2,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(1,1)
		mask = set_mask(2,0)
		for i in range(16):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################

	#######################################################
	#'''	
	if args.mode == 8:
		mask = set_mask(1,1)
		net1 = net_mask_mul(net1, mask)
		mask = set_mask(0,1)
		mask = set_mask(1,0)
		for i in range(16):
			mask[i] = torch.mul(mask[i],mask_rand[i])
		add_mask(net1,mask) 
	#'''
	#######################################################
	
	#######################################################
	#Enable this part for gaussian 016
	'''
	mask = set_mask(3,1)
	net1 = net_mask_mul(net1)
	mask = set_mask(0,0)
	mask = set_mask(2,1)
	mask = set_mask(3,0)
	for i in range(16):
		mask[i] = torch.mul(mask[i],mask_rand[i])
	add_mask(net1,mask) 
	'''
	#######################################################

	#######################################################
	#Enable this part for gaussian 025
	'''
	mask = set_mask(2,1)
	net1 = net_mask_mul(net1)
	mask = set_mask(0,1)
	mask = set_mask(2,0)
	for i in range(16):
		mask[i] = torch.mul(mask[i],mask_rand[i])
	add_mask(net1,mask) 
	'''
	#######################################################
	
	#######################################################
	#Check if training works
	'''
	f = open('testout1.csv','a+')
	for child in net1.children():
		for param in child.conv10[0].parameters():
			print(torch.sum(torch.abs(param)), file=f)
	f.close()

	mask = set_mask(0,1)
	mask = set_mask(4,0)
	net1 = net_mask_mul(net1)
	#net2 = net_mask_mul(net2)
	#save_network(net2)
	#net1 = add_network(net1)

	f = open('testout2.csv','a+')
	for child in net1.children():
		for param in child.conv10[0].parameters():
			print(torch.sum(torch.abs(param)),file=f)
	f.close()
	'''
	#######################################################
	#'''
	f = open('testout.txt','a+')
	for child in net1.children():
		for param in child.conv2[0].parameters():
			for i in range(0,64):
				for j in range(0,64):
					print("data[{},{},:,:] = {}".format(i,j,param.data[i,j,:,:]), file=f)
	f.close()
	#'''

	if args.outputfile == 'NULL':
		pass
	else:
		#torch.save(net1, './checkpoint/ckpt_20180613_half_clean_0.125_gaussian.t0')
		print('Saving..')
		state = {
			'net': net1.module if use_cuda else net1,
			'acc': 0,
		}
	
		torch.save(state, './checkpoint/'+args.outputfile)
	#printweight(net1)
