'''
some parts of code are extracted from "https://github.com/kuangliu/pytorch-cifar"
I modified some parts for our experiment
'''

from __future__ import print_function

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

import struct
import random
import cifar_dirty_test
import cifar_dirty_train
#import VGG16_yh 

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--ldpr', default=0, type=int, help='previously pruned network') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--thres', default=0, type=float)
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--network', default='ckpt_20181130_half_clean.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--network2', default='ckpt_20181130_half_clean_prune_80.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--outputfile', default='ckpt_20181130_half_clean.t0', help='output file name', metavar="FILE")
parser.add_argument('--imgprint', default=0, type=int, help='print input and dirty img to png') #mode=1 is train, mode=0 is inference
parser.add_argument('--gau', type=float, default=0, metavar='N',help='gaussian noise standard deviation')
parser.add_argument('--blur', type=float, default=0, metavar='N',help='blur noise standard deviation')

args = parser.parse_args()

global glob_gau
global glob_blur

glob_gau = args.gau
glob_blur = args.blur

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy

transform_train = transforms.Compose([transforms.RandomCrop(32,padding=4),
									  transforms.RandomHorizontalFlip(),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
transform_test = transforms.Compose([transforms.ToTensor(),
									 transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

cifar_train = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=True, transform=transform_train, target_transform=None, download=True)
cifar_test = dset.CIFAR100("/home/yhbyun/Dataset/CIFAR100/", train=False, transform=transform_test, target_transform=None, download=True)

#train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset([cifar_train, cifar_train_blur_033]),batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
train_loader = torch.utils.data.DataLoader(cifar_train,batch_size=args.bs, shuffle=True,num_workers=8,drop_last=False)
test_loader = torch.utils.data.DataLoader(cifar_test,batch_size=10000, shuffle=False,num_workers=8,drop_last=False)

mode = args.mode

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
		#self._initialize_weights()

	def forward(self,x):
		global glob_gau
		global glob_blur
		if args.imgprint == 1:
			npimg = np.array(x,dtype=float)
			npimg = npimg.squeeze(0)
			scipy.misc.toimage(npimg).save("img0.png")
		#Noise generation part
		if (glob_gau==0)&(glob_blur==0):
			#no noise
			pass

		elif (glob_blur == 0)&(glob_gau == 1):
			#gaussian noise add
			
			gau_kernel = torch.randn(x.size())*args.gau
			x = Variable(gau_kernel.cuda()) + x
			

		elif (glob_gau == 0)&(glob_blur == 1):
			#blur noise add
			blur_kernel_partial = torch.FloatTensor(utils.genblurkernel(args.blur))
			blur_kernel_partial = torch.matmul(blur_kernel_partial.unsqueeze(1),torch.transpose(blur_kernel_partial.unsqueeze(1),0,1))
			kernel_size = blur_kernel_partial.size()[0]
			zeros = torch.zeros(kernel_size,kernel_size)
			blur_kernel = torch.cat((blur_kernel_partial,zeros,zeros,
			zeros,blur_kernel_partial,zeros,
			zeros,zeros,blur_kernel_partial),0)
			blur_kernel = blur_kernel.view(3,3,kernel_size,kernel_size)
			blur_padding = int((blur_kernel_partial.size()[0]-1)/2)
			#x = torch.nn.functional.conv2d(x, weight=blur_kernel.cuda(), padding=blur_padding)
			x = torch.nn.functional.conv2d(x, weight=Variable(blur_kernel.cuda()), padding=blur_padding)

		elif (glob_gau == 1) & (glob_blur == 1):
			#both gaussian and blur noise added
			blur_kernel_partial = torch.FloatTensor(utils.genblurkernel(args.blur))
			blur_kernel_partial = torch.matmul(blur_kernel_partial.unsqueeze(1),torch.transpose(blur_kernel_partial.unsqueeze(1),0,1))
			kernel_size = blur_kernel_partial.size()[0]
			zeros = torch.zeros(kernel_size,kernel_size)
			blur_kernel = torch.cat((blur_kernel_partial,zeros,zeros,
			zeros,blur_kernel_partial,zeros,
			zeros,zeros,blur_kernel_partial),0)
			blur_kernel = blur_kernel.view(3,3,kernel_size,kernel_size)
			blur_padding = int((blur_kernel_partial.size()[0]-1)/2)
			x = torch.nn.functional.conv2d(x, weight=Variable(blur_kernel.cuda()), padding=blur_padding)
			gau_kernel = torch.randn(x.size())*args.gau
			x = Variable(gau_kernel.cuda()) + x
		else:
			print("Something is wrong in noise adding part")
			exit()
		if args.imgprint == 1:
			npimg = np.array(x,dtype=float)
			npimg = npimg.squeeze(0)
			scipy.misc.toimage(npimg).save("img1.png")
			exit()

		if args.fixed:
			x = roundmax(x)
			x = quant(x)
		out1 = self.conv1(x) # 1250*64*32*32
		if args.fixed:
			out1 = quant(out1) 
			out1 = roundmax(out1)

		out2 = self.conv2(out1) # 1250*64*32*32
		if args.fixed:
			out2 = quant(out2)
			out2 = roundmax(out2)

		out3 = self.maxpool1(out2)
		out4 = self.conv3(out3) # 1250*128*16*16
		if args.fixed:
			out4 = quant(out4) 
			out4 = roundmax(out4)
		out5 = self.conv4(out4) # 1250*128*16*16
		if args.fixed:
			out5 = quant(out5) 
			out5 = roundmax(out5)

		out6 = self.maxpool2(out5)
		out7 = self.conv5(out6) # 1250*256*8*8
		if args.fixed:
			out7 = quant(out7) 
			out7 = roundmax(out7)
		out8 = self.conv6(out7) # 1250*256*8*8
		if args.fixed:
			out8 = quant(out8) 
			out8 = roundmax(out8)
		out9 = self.conv7(out8) # 1250*256*8*8
		if args.fixed:
			out9 = quant(out9) 
			out9 = roundmax(out9)

		out10 = self.maxpool3(out9)
		out11 = self.conv8(out10) # 1250*512*4*4
		if args.fixed:
			out11 = quant(out11) 
			out11 = roundmax(out11)
		out12 = self.conv9(out11) # 1250*512*4*4
		if args.fixed:
			out12 = quant(out12) 
			out12 = roundmax(out12)
		out13 = self.conv10(out12) # 1250*512*4*
		if args.fixed:
			out13 = quant(out13) 
			out13 = roundmax(out13)

		out14 = self.maxpool4(out13)

		out15 = self.conv11(out14) # 1250*512*2*
		if args.fixed:
			out15 = quant(out15) 
			out15 = roundmax(out15)
		out16 = self.conv12(out15) # 1250*512*2*
		if args.fixed:
			out16 = quant(out16) 
			out16 = roundmax(out16)
		out17 = self.conv13(out16) # 1250*512*2*
		if args.fixed:
			out17 = quant(out17) 
			out17 = roundmax(out17)

		out18 = self.maxpool5(out17)

		out19 = out18.view(out18.size(0),-1)
		out20 = self.fc1(out19) # 1250*512
		if args.fixed:
			out20 = quant(out20) 
			out20 = roundmax(out20)
		out21 = self.fc2(out20) # 1250*512
		if args.fixed:
			out21 = quant(out21) 
			out21 = roundmax(out21)
		out22 = self.fc3(out21) # 1250*10
		'''
		if args.fixed:
			out22 = quant(out22) 
			out22 = roundmax(out22)
		'''
		return out22

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				#print(m)
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

				#if m.bias is not None:
					#nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				#nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				#print(m)
				nn.init.normal_(m.weight, 0, 0.01)
				#nn.init.constant_(m.bias, 0)

def roundmax(input):
	'''
	maximum = 2**args.iwidth-1
	minimum = -maximum-1
	input = F.relu(torch.add(input, -minimum))
	input = F.relu(torch.add(torch.neg(input), maximum-minimum))
	input = torch.add(torch.neg(input), maximum)
	'''
	return input	

def quant(input):
	#input = torch.round(input / (2 ** (-args.aprec))) * (2 ** (-args.aprec))
	return input

def paramsget(net):
	try:
		params = net.conv1[0].weight.view(-1,)
		params = torch.cat((params,net.conv2[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv3[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv4[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv5[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv6[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv7[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv8[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv9[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv10[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv11[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv12[0].weight.view(-1,)),0)
		params = torch.cat((params,net.conv13[0].weight.view(-1,)),0)
		params = torch.cat((params,net.fc1[1].weight.view(-1,)),0)
		params = torch.cat((params,net.fc2[1].weight.view(-1,)),0)
		params = torch.cat((params,net.fc3[0].weight.view(-1,)),0)
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				params = param.view(-1,)
		for child in net.children():
			for param in child.conv2[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv4[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv5[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv6[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv7[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv8[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv9[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv10[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv11[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv12[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.conv13[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)

		for child in net.children():
			for param in child.fc1[1].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.fc2[1].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.fc3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)>thres).type(torch.FloatTensor)
		#result = torch.sum(tmp)/params.size()[0]*64/28
		#result = torch.sum(tmp)/params.size()[0]*64/11
		#result = torch.sum(tmp)/params.size()[0]*64/9
		result = torch.sum(tmp)/params.size()[0]*4 #for half clean
		#result = torch.sum(tmp)/params.size()[0] # for full size
		if ((100-args.pr)/100)>result:
			print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(thres):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = (torch.abs(net.conv1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[1] = (torch.abs(net.conv2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[2] = (torch.abs(net.conv3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[3] = (torch.abs(net.conv4[0].weight.data)>thres).type(torch.FloatTensor)
		mask[4] = (torch.abs(net.conv5[0].weight.data)>thres).type(torch.FloatTensor)
		mask[5] = (torch.abs(net.conv6[0].weight.data)>thres).type(torch.FloatTensor)
		mask[6] = (torch.abs(net.conv7[0].weight.data)>thres).type(torch.FloatTensor)
		mask[7] = (torch.abs(net.conv8[0].weight.data)>thres).type(torch.FloatTensor)
		mask[8] = (torch.abs(net.conv9[0].weight.data)>thres).type(torch.FloatTensor)
		mask[9] = (torch.abs(net.conv10[0].weight.data)>thres).type(torch.FloatTensor)
		mask[10] = (torch.abs(net.conv11[0].weight.data)>thres).type(torch.FloatTensor)
		mask[11] = (torch.abs(net.conv12[0].weight.data)>thres).type(torch.FloatTensor)
		mask[12] = (torch.abs(net.conv13[0].weight.data)>thres).type(torch.FloatTensor)
		mask[13] = (torch.abs(net.fc1[1].weight.data)>thres).type(torch.FloatTensor)
		mask[14] = (torch.abs(net.fc2[1].weight.data)>thres).type(torch.FloatTensor)
		mask[15] = (torch.abs(net.fc3[0].weight.data)>thres).type(torch.FloatTensor)
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv2[0].parameters():
				mask[1] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv3[0].parameters():
				mask[2] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv4[0].parameters():
				mask[3] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv5[0].parameters():
				mask[4] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv6[0].parameters():
				mask[5] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv7[0].parameters():
				mask[6] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv8[0].parameters():
				mask[7] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv9[0].parameters():
				mask[8] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv10[0].parameters():
				mask[9] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv11[0].parameters():
				mask[10] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv12[0].parameters():
				mask[11] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv13[0].parameters():
				mask[12] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	

		for child in net.children():
			for param in child.fc1[1].parameters():
				mask[13] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.fc2[1].parameters():
				mask[14] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.fc3[0].parameters():
				mask[15] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
	return mask

def getGroupPruningMask(params):
	thres = 0
	bunch = 4
	params = params.view(-1,bunch)
	while 1:
		partialmask = torch.mean(torch.abs(params),1)
		partialmask = torch.ge(partialmask, thres)
		fullmask = torch.unsqueeze(partialmask, 1)
		fullmask = torch.cat((fullmask, fullmask, fullmask, fullmask),1)
		#print(fullmask.size())
		#print(torch.sum(fullmask).item())
		#print(params.size())
		#print(type(params.size()))
		if (torch.sum(fullmask).item()/15285952*100*4) < (100-args.pr):
			#print((torch.sum(fullmask).item()/15285952*100))
			#print('thres : ', thres, ', ',(torch.sum(fullmask).item()/15285952))
			#print('break!')
			fullmask = fullmask.view(-1,1)
			break;
		else:
			#print((torch.sum(fullmask).item()/15285952*100))
			#print('thres : ', thres, ', ',(torch.sum(fullmask).item()/15285952))
			thres += 0.0001
	#print(fullmask.size())
	#exit()
	return splitFullMask(fullmask)

def splitFullMask(fullmask):
	mask = []
	mask.append(fullmask[0:1728].unsqueeze(1).unsqueeze(1).view(64,3,3,3).type(torch.FloatTensor))
	mask.append(fullmask[1728:38592].unsqueeze(1).unsqueeze(1).view(64,64,3,3).type(torch.FloatTensor))
	mask.append(fullmask[38592:112320].unsqueeze(1).unsqueeze(1).view(128,64,3,3).type(torch.FloatTensor))
	mask.append(fullmask[112320:259776].unsqueeze(1).unsqueeze(1).view(128,128,3,3).type(torch.FloatTensor))
	mask.append(fullmask[259776:554688].unsqueeze(1).unsqueeze(1).view(256,128,3,3).type(torch.FloatTensor))
	mask.append(fullmask[554688:1144512].unsqueeze(1).unsqueeze(1).view(256,256,3,3).type(torch.FloatTensor))
	mask.append(fullmask[1144512:1734336].unsqueeze(1).unsqueeze(1).view(256,256,3,3).type(torch.FloatTensor))
	mask.append(fullmask[1734336:2913984].unsqueeze(1).unsqueeze(1).view(512,256,3,3).type(torch.FloatTensor))
	mask.append(fullmask[2913984:5273280].unsqueeze(1).unsqueeze(1).view(512,512,3,3).type(torch.FloatTensor))
	mask.append(fullmask[5273280:7632576].unsqueeze(1).unsqueeze(1).view(512,512,3,3).type(torch.FloatTensor))
	mask.append(fullmask[7632576:9991872].unsqueeze(1).unsqueeze(1).view(512,512,3,3).type(torch.FloatTensor))
	mask.append(fullmask[9991872:12351168].unsqueeze(1).unsqueeze(1).view(512,512,3,3).type(torch.FloatTensor))
	mask.append(fullmask[12351168:14710464].unsqueeze(1).unsqueeze(1).view(512,512,3,3).type(torch.FloatTensor))
	mask.append(fullmask[14710464:14972608].view(512,512).type(torch.FloatTensor))
	mask.append(fullmask[14972608:15234752].view(512,512).type(torch.FloatTensor))
	mask.append(fullmask[15234752:15285952].view(100,512).type(torch.FloatTensor))
	return mask

def getNonzeroPoints(net):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = (net.conv1[0].weight.data != 0).type(torch.FloatTensor)
		mask[1] = (net.conv2[0].weight.data != 0).type(torch.FloatTensor)
		mask[2] = (net.conv3[0].weight.data != 0).type(torch.FloatTensor)
		mask[3] = (net.conv4[0].weight.data != 0).type(torch.FloatTensor)
		mask[4] = (net.conv5[0].weight.data != 0).type(torch.FloatTensor)
		mask[5] = (net.conv6[0].weight.data != 0).type(torch.FloatTensor)
		mask[6] = (net.conv7[0].weight.data != 0).type(torch.FloatTensor)
		mask[7] = (net.conv8[0].weight.data != 0).type(torch.FloatTensor)
		mask[8] = (net.conv9[0].weight.data != 0).type(torch.FloatTensor)
		mask[9] = (net.conv10[0].weight.data != 0).type(torch.FloatTensor)
		mask[10] = (net.conv11[0].weight.data != 0).type(torch.FloatTensor)
		mask[11] = (net.conv12[0].weight.data != 0).type(torch.FloatTensor)
		mask[12] = (net.conv13[0].weight.data != 0).type(torch.FloatTensor)
		mask[13] = (net.fc1[1].weight.data != 0).type(torch.FloatTensor)
		mask[14] = (net.fc2[1].weight.data != 0).type(torch.FloatTensor)
		mask[15] = (net.fc3[0].weight.data != 0).type(torch.FloatTensor)
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv2[0].parameters():
				mask[1] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv3[0].parameters():
				mask[2] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv4[0].parameters():
				mask[3] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv5[0].parameters():
				mask[4] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv6[0].parameters():
				mask[5] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv7[0].parameters():
				mask[6] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv8[0].parameters():
				mask[7] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv9[0].parameters():
				mask[8] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv10[0].parameters():
				mask[9] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv11[0].parameters():
				mask[10] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv12[0].parameters():
				mask[11] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.conv13[0].parameters():
				mask[12] = (param.data != 0).type(torch.FloatTensor)	

		for child in net.children():
			for param in child.fc1[1].parameters():
				mask[13] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.fc2[1].parameters():
				mask[14] = (param.data != 0).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.fc3[0].parameters():
				mask[15] = (param.data != 0).type(torch.FloatTensor)	
	return mask

def pruneNetwork(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[0].cuda())
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[1].cuda())
			param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[2].cuda())
			param.data = torch.mul(param.data,mask[2].cuda())
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[3].cuda())
			param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[4].cuda())
			param.data = torch.mul(param.data,mask[4].cuda())
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[5].cuda())
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[6].cuda())
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[7].cuda())
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[8].cuda())
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[9].cuda())
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[10].cuda())
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[11].cuda())
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[12].cuda())
			param.data = torch.mul(param.data,mask[12].cuda())

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[13].cuda())
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[14].cuda())
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[15].cuda())
			param.data = torch.mul(param.data,mask[15].cuda())
	return

def set_mask(mask, block, val):
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

def save_network(net):
	mask = torch.load('mask_null.dat')
	try:
		mask[0] = net.conv1[0].weight.data
		mask[1] = net.conv2[0].weight.data
		mask[2] = net.conv3[0].weight.data
		mask[3] = net.conv4[0].weight.data
		mask[4] = net.conv5[0].weight.data
		mask[5] = net.conv6[0].weight.data
		mask[6] = net.conv7[0].weight.data
		mask[7] = net.conv8[0].weight.data
		mask[8] = net.conv9[0].weight.data
		mask[9] = net.conv10[0].weight.data 
		mask[10] = net.conv11[0].weight.data
		mask[11] = net.conv12[0].weight.data
		mask[12] = net.conv13[0].weight.data
		mask[13] = net.fc1[1].weight.data
		mask[14] = net.fc2[1].weight.data
		mask[15] = net.fc3[0].weight.data
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = param.data
		for child in net.children():
			for param in child.conv2[0].parameters():
				mask[1] = param.data		
		for child in net.children():
			for param in child.conv3[0].parameters():
				mask[2] = param.data		
		for child in net.children():
			for param in child.conv4[0].parameters():
				mask[3] = param.data		
		for child in net.children():
			for param in child.conv5[0].parameters():
				mask[4] = param.data	
		for child in net.children():
			for param in child.conv6[0].parameters():
				mask[5] = param.data
		for child in net.children():
			for param in child.conv7[0].parameters():
				mask[6] = param.data
		for child in net.children():
			for param in child.conv8[0].parameters():
				mask[7] = param.data
		for child in net.children():
			for param in child.conv9[0].parameters():
				mask[8] = param.data
		for child in net.children():
			for param in child.conv10[0].parameters():
				mask[9] = param.data
		for child in net.children():
			for param in child.conv11[0].parameters():
				mask[10] = param.data
		for child in net.children():
			for param in child.conv12[0].parameters():
				mask[11] = param.data
		for child in net.children():
			for param in child.conv13[0].parameters():
				mask[12] = param.data

		for child in net.children():
			for param in child.fc1[1].parameters():
				mask[13] = param.data
		for child in net.children():
			for param in child.fc2[1].parameters():
				mask[14] = param.data
		for child in net.children():
			for param in child.fc3[0].parameters():
				mask[15] = param.data
	return mask

def add_network():
	layer = save_network(net2)
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

def net_mask_mul(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[0].cuda())
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.conv2[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[1].cuda())
			param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param in child.conv3[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[2].cuda())
			param.data = torch.mul(param.data,mask[2].cuda())
	for child in net.children():
		for param in child.conv4[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[3].cuda())
			param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param in child.conv5[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[4].cuda())
			param.data = torch.mul(param.data,mask[4].cuda())
	for child in net.children():
		for param in child.conv6[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[5].cuda())
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.conv7[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[6].cuda())
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.conv8[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[7].cuda())
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.conv9[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[8].cuda())
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.conv10[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[9].cuda())
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.conv11[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[10].cuda())
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.conv12[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[11].cuda())
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.conv13[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[12].cuda())
			param.data = torch.mul(param.data,mask[12].cuda())

	for child in net.children():
		for param in child.fc1[1].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[13].cuda())
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.fc2[1].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[14].cuda())
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.fc3[0].parameters():
			if args.mode != 0:
				param.grad.data = torch.mul(param.grad.data, mask[15].cuda())
			param.data = torch.mul(param.data,mask[15].cuda())

def concatMask(mask):
	params = mask[0].view(-1,)
	params = torch.cat((params, mask[1].view(-1,)),0)
	params = torch.cat((params, mask[2].view(-1,)),0)
	params = torch.cat((params, mask[3].view(-1,)),0)
	params = torch.cat((params, mask[4].view(-1,)),0)
	params = torch.cat((params, mask[5].view(-1,)),0)
	params = torch.cat((params, mask[6].view(-1,)),0)
	params = torch.cat((params, mask[7].view(-1,)),0)
	params = torch.cat((params, mask[8].view(-1,)),0)
	params = torch.cat((params, mask[9].view(-1,)),0)
	params = torch.cat((params, mask[10].view(-1,)),0)
	params = torch.cat((params, mask[11].view(-1,)),0)
	params = torch.cat((params, mask[12].view(-1,)),0)
	params = torch.cat((params, mask[13].view(-1,)),0)
	params = torch.cat((params, mask[14].view(-1,)),0)
	params = torch.cat((params, mask[15].view(-1,)),0)
	return params

# Load checkpoint.
if args.mode == 0:
	checkpoint = torch.load('./checkpoint/'+args.network)
	net = checkpoint['net']

'''
elif args.mode == 1:
	checkpoint = torch.load('./checkpoint/ckpt_20181130_half_clean.t0')
	ckpt = torch.load('./checkpoint/ckpt_20181130_half_clean.t0')
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0
'''

if args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		checkpoint = torch.load('./checkpoint/ckpt_20181130_half_clean.t0')
		net = checkpoint['net']
		best_acc = checkpoint['acc']
	else:
		net = CNN()
		best_acc = 0

if args.mode == 2:
	checkpoint = torch.load('./checkpoint/'+args.network)
	net = checkpoint['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0
	params = paramsget(net)
	thres = findThreshold(params)
	mask_prune = getPruningMask(thres)	

if args.mode == 3:
	checkpoint = torch.load('./checkpoint/'+args.network)
	ckpt = torch.load('./checkpoint/'+args.network2)
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0
	mask_nonzero = getNonzeroPoints(net2)  
	mask_zero = []
	for i in range(16):
		mask_zero.append((mask_nonzero[i] == 0).type(torch.FloatTensor))

if args.mode == 4:
	checkpoint = torch.load('./checkpoint/'+args.network)
	ckpt = torch.load('./checkpoint/'+args.network2)
	net = checkpoint['net']
	net2 = ckpt['net']
	if args.resume:
		print('==> Resuming from checkpoint..')
		best_acc = checkpoint['acc']
	else:
		best_acc = 0
	mask_net = save_network(net) 
	mask_net2 = save_network(net2) 
	'''
	print(mask_net[5][0])
	print(mask_net2[5][30])
	exit()
	'''
	mask_nonzero = getNonzeroPoints(net2)
	mask_zero = []
	for i in range(16):
		mask_zero.append((mask_nonzero[i] == 0).type(torch.FloatTensor))
	mask_diff = []
	for i in range(16):
		mask_diff.append(mask_net[i] - mask_net2[i])
	params = concatMask(mask_diff) 	
	thres = findThreshold(params)
	mask_prune = getPruningMask(thres)	

if args.mode == 7:
	checkpoint = torch.load('./checkpoint/'+args.network)
	net = checkpoint['net']
	params = paramsget(net)
	tmp = torch.sum(params.data != 0)
	print(tmp.item()/params.size()[0])
	exit()

if use_cuda:
	net.cuda()
	net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
	if args.mode > 2:
		net2.cuda()
		net2 = torch.nn.DataParallel(net2, device_ids=range(torch.cuda.device_count()))
	cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
#optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

start_epoch = args.se
num_epoch = args.ne

# Training
def train(epoch):
	global glob_gau
	global glob_blur
	glob_gau = 0
	glob_blur = 0
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	#mask_channel = set_mask(set_mask(mask_channel, 3, 1), 4, 0)
	mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		net_mask_mul(mask_channel)
		net_mask_mul(mask_group_reverse)
		add_network()

		optimizer.step()

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def pruning(epoch):
	global glob_gau
	global glob_blur
	glob_gau = 0
	glob_blur = 0
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		net_mask_mul(mask_channel)
		net_mask_mul(mask_prune)

		optimizer.step()

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def retrain(epoch):
	global glob_gau
	global glob_blur
	glob_gau = 0
	glob_blur = 0
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		net_mask_mul(mask_channel)
		net_mask_mul(mask_zero)
		add_network()	

		optimizer.step()

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def repruning(epoch):
	global glob_gau
	global glob_blur
	glob_gau = 0
	glob_blur = 0
	print('\nEpoch: %d' % epoch)
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		net_mask_mul(mask_channel)
		net_mask_mul(mask_prune)
		net_mask_mul(mask_zero)

		add_network()	

		optimizer.step()

		train_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test():
	global best_acc
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	mask_channel = torch.load('mask_null.dat')
	mask_channel = set_mask(mask_channel, 4, 1)
	net_mask_mul(mask_channel)
	if args.mode == 2:
		net_mask_mul(mask_channel)
		net_mask_mul(mask_prune)
	if args.mode == 3:
		net_mask_mul(mask_channel)
		net_mask_mul(mask_zero)
		add_network()
	elif args.mode == 4:
		net_mask_mul(mask_prune)
		net_mask_mul(mask_channel)
		net_mask_mul(mask_zero)
		add_network()
	for batch_idx, (inputs, targets) in enumerate(test_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)

		test_loss += loss.data.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += predicted.eq(targets.data).cpu().sum().item()

		progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
			% (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


	# Save checkpoint.
	acc = 100.*correct/total
	if acc > best_acc:

		state = {
			'net': net.module if use_cuda else net,
			'acc': acc,
		}
		if not os.path.isdir('checkpoint'):
			os.mkdir('checkpoint')
		if args.mode == 0:
			pass
		else:
			print('Saving..')
			#torch.save(state, './checkpoint/ckpt_20181130_half_clean.t0')
			torch.save(state, './checkpoint/'+args.outputfile)
		best_acc = acc

	return acc
	
# Retraining
# Truncate weight param
pprec = args.pprec
def quantize():
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv2[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv3[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv4[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv5[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv6[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv7[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv8[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv9[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv10[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv11[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv12[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.conv13[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))

	for child in net.children():
		for param in child.fc1[1].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.fc2[1].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))
	for child in net.children():
		for param in child.fc3[0].parameters():
			param.data = torch.round(param.data / (2 ** -(pprec))) * (2 ** -(pprec))


# Train+inference vs. Inference
if mode == 0: # only inference
	test()

elif mode == 1: # mode=1 is training & inference @ each epoch
	for epoch in range(start_epoch, start_epoch+num_epoch):
		train(epoch)

		test()
elif mode == 2: # retrain for quantization and pruning
	for epoch in range(0,num_epoch):
		pruning(epoch) 

		test()
elif mode == 3: # retrain for quantization and pruning
	for epoch in range(0,num_epoch):
		retrain(epoch) 

		test()
elif mode == 4: # retrain for quantization and pruning
	for epoch in range(0,num_epoch):
		repruning(epoch) 

		test()
else:
	pass

