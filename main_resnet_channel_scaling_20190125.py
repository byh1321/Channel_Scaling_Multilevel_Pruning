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

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--se', default=0, type=int, help='start epoch')
parser.add_argument('--ne', default=0, type=int, help='number of epoch')
parser.add_argument('--pr', default=0, type=int, help='pruning') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--ldpr', default=0, type=int, help='previously pruned network') # mode=1 is pruning, mode=0 is no pruning
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--mode', default=1, type=int, help='train or inference') #mode=1 is train, mode=0 is inference
parser.add_argument('--thres', default=0, type=float)
parser.add_argument('--cs', default=0, type=float)
parser.add_argument('--pprec', type=int, default=20, metavar='N',help='parameter precision for layer weight')
parser.add_argument('--aprec', type=int, default=20, metavar='N',help='Arithmetic precision for internal arithmetic')
parser.add_argument('--iwidth', type=int, default=10, metavar='N',help='integer bitwidth for internal part')
parser.add_argument('--fixed', type=int, default=0, metavar='N',help='fixed=0 - floating point arithmetic')
parser.add_argument('--network', default='ckpt_20190125.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--network2', default='ckpt_20190125_prune_95.t0', help='input network ckpt name', metavar="FILE")
parser.add_argument('--outputfile', default='ckpt_20190125.t0', help='output file name', metavar="FILE")
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

class ResNet18(nn.Module):
	def __init__(self):
		super(ResNet18,self).__init__()
		self.conv1 = nn.Sequential(
			nn.Conv2d(3, int(64*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(64*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
		)
		self.layer1_basic1 = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(64*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(64*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic2 = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(64*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(64*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer1_basic3 = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(64*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(64*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer1_basic4 = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(64*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(64*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer1_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer2_basic1 = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(128*args.cs), kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(int(128*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_downsample = nn.Sequential(
			nn.Conv2d(int(64*args.cs), int(128*args.cs), kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(int(128*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_basic2 = nn.Sequential(
			nn.Conv2d(int(128*args.cs), int(128*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(128*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer2_basic3 = nn.Sequential(
			nn.Conv2d(int(128*args.cs), int(128*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(128*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer2_basic4 = nn.Sequential(
			nn.Conv2d(int(128*args.cs), int(128*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(128*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer2_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic1 = nn.Sequential(
			nn.Conv2d(int(128*args.cs), int(256*args.cs), kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(int(256*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_downsample = nn.Sequential(
			nn.Conv2d(int(128*args.cs), int(256*args.cs), kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(int(256*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_basic2 = nn.Sequential(
			nn.Conv2d(int(256*args.cs), int(256*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(256*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer3_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer3_basic3 = nn.Sequential(
			nn.Conv2d(int(256*args.cs), int(256*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(256*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_basic4 = nn.Sequential(
			nn.Conv2d(int(256*args.cs), int(256*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(256*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer3_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)

		self.layer4_basic1 = nn.Sequential(
			nn.Conv2d(int(256*args.cs), int(512*args.cs), kernel_size=3, stride=2, padding=1, bias=False),
			nn.BatchNorm2d(int(512*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_downsample = nn.Sequential(
			nn.Conv2d(int(256*args.cs), int(512*args.cs), kernel_size=1, stride=2, bias=False),
			nn.BatchNorm2d(int(512*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
		)
		self.layer4_basic2 = nn.Sequential(
			nn.Conv2d(int(512*args.cs), int(512*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(512*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu1 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.layer4_basic3 = nn.Sequential(
			nn.Conv2d(int(512*args.cs), int(512*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(512*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_basic4 = nn.Sequential(
			nn.Conv2d(int(512*args.cs), int(512*args.cs), kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(int(512*args.cs), eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=False),
		)
		self.layer4_relu2 = nn.Sequential(
			nn.ReLU(inplace=False),
		)
		self.linear = nn.Sequential(
			nn.Linear(int(512*args.cs), 100, bias=False)
		)
		self._initialize_weights()

	def forward(self,x):
		if args.fixed:
			x = quant(x)
			x = roundmax(x)

		out = x.clone()
		out = self.conv1(out)
		

		residual = out

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic1(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic2(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu1(out)
		residual = out

		out = self.layer1_basic3(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer1_basic4(out)
		
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer1_relu2(out)
		residual = self.layer2_downsample(out)

		out = self.layer2_basic1(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic2(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual

		out = self.layer2_relu1(out)
		residual = out

		out = self.layer2_basic3(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer2_basic4(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer2_relu2(out)

		residual = self.layer3_downsample(out)

		out = self.layer3_basic1(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic2(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu1(out)

		residual = out

		out = self.layer3_basic3(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer3_basic4(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer3_relu2(out)

		residual = self.layer4_downsample(out)

		out = self.layer4_basic1(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic2(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)
		out += residual
		out = self.layer4_relu1(out)
		residual = out

		out = self.layer4_basic3(out)
		

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out = self.layer4_basic4(out)
		

		if args.fixed:
			residual = quant(residual)
			residual = roundmax(residual)

		if args.fixed:
			out = quant(out)
			out = roundmax(out)

		out += residual
		out = self.layer4_relu2(out)
		out = F.avg_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		
		out = self.linear(out)

		return out

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

def maskgen():
	mask = []
	mask_conv1 = torch.zeros(int(64*args.cs),3,3,3).cuda()
	mask_layer1_basic1_conv = torch.zeros(int(64*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer1_basic2_conv = torch.zeros(int(64*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer1_basic3_conv = torch.zeros(int(64*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer1_basic4_conv = torch.zeros(int(64*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer2_basic1_conv = torch.zeros(int(128*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer2_basic2_conv = torch.zeros(int(128*args.cs),int(128*args.cs),3,3).cuda()
	mask_layer2_basic3_conv = torch.zeros(int(128*args.cs),int(128*args.cs),3,3).cuda()
	mask_layer2_basic4_conv = torch.zeros(int(128*args.cs),int(128*args.cs),3,3).cuda()
	mask_layer3_basic1_conv = torch.zeros(int(256*args.cs),int(128*args.cs),3,3).cuda()
	mask_layer3_basic2_conv = torch.zeros(int(256*args.cs),int(256*args.cs),3,3).cuda()
	mask_layer3_basic3_conv = torch.zeros(int(256*args.cs),int(256*args.cs),3,3).cuda()
	mask_layer3_basic4_conv = torch.zeros(int(256*args.cs),int(256*args.cs),3,3).cuda()
	mask_layer4_basic1_conv = torch.zeros(int(512*args.cs),int(256*args.cs),3,3).cuda()
	mask_layer4_basic2_conv = torch.zeros(int(512*args.cs),int(512*args.cs),3,3).cuda()
	mask_layer4_basic3_conv = torch.zeros(int(512*args.cs),int(512*args.cs),3,3).cuda()
	mask_layer4_basic4_conv = torch.zeros(int(512*args.cs),int(512*args.cs),3,3).cuda()
	mask_layer2_downsample = torch.zeros(int(128*args.cs),int(64*args.cs),3,3).cuda()
	mask_layer3_downsample = torch.zeros(int(256*args.cs),int(128*args.cs),3,3).cuda()
	mask_layer4_downsample = torch.zeros(int(512*args.cs),int(256*args.cs),3,3).cuda()
	mask_linear = torch.zeros(100,int(512*args.cs)).cuda()

	mask.append(mask_conv1)
	mask.append(mask_layer1_basic1_conv)
	mask.append(mask_layer1_basic2_conv)
	mask.append(mask_layer1_basic3_conv)
	mask.append(mask_layer1_basic4_conv)
	mask.append(mask_layer2_basic1_conv)
	mask.append(mask_layer2_basic2_conv)
	mask.append(mask_layer2_basic3_conv)
	mask.append(mask_layer2_basic4_conv)
	mask.append(mask_layer3_basic1_conv)
	mask.append(mask_layer3_basic2_conv)
	mask.append(mask_layer3_basic3_conv)
	mask.append(mask_layer3_basic4_conv)
	mask.append(mask_layer4_basic1_conv)
	mask.append(mask_layer4_basic2_conv)
	mask.append(mask_layer4_basic3_conv)
	mask.append(mask_layer4_basic4_conv)
	mask.append(mask_layer2_downsample)
	mask.append(mask_layer3_downsample)
	mask.append(mask_layer4_downsample)
	mask.append(mask_linear)
	return mask
	
	
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
		params = torch.cat((params,net.layer1_basic1[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer1_basic2[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer1_basic3[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer1_basic4[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer2_basic1[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer2_basic2[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer2_basic3[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer2_basic4[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer3_basic1[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer3_basic2[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer3_basic3[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer3_basic4[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer4_basic1[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer4_basic2[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer4_basic3[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer4_basic4[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer2_downsample[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer3_downsample[0].weight.view(-1,)),0)
		params = torch.cat((params,net.layer4_downsample[0].weight.view(-1,)),0)
		params = torch.cat((params,net.linear[0].weight.view(-1,)),0)
	except:
		for child in net.children():
			for param in child.layer1_basic1[0].parameters():
				params = param.view(-1,)
		for child in net.children():
			for param in child.layer1_basic2[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer1_basic3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer1_basic4[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer2_basic1[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer2_basic2[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer2_basic3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer2_basic4[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer3_basic1[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer3_basic2[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer3_basic3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer3_basic4[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer4_basic1[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer4_basic2[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer4_basic3[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer4_basic4[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)

		for child in net.children():
			for param in child.layer2_downsample[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer3_downsample[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.layer4_downsample[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
		for child in net.children():
			for param in child.linear[0].parameters():
				params = torch.cat((params, param.view(-1,)),0)
	return params

def findThreshold(params):
	thres=0
	while 1:
		tmp = (torch.abs(params.data)>thres).type(torch.FloatTensor)
		result = torch.sum(tmp)/11210432
		#result = torch.sum(tmp)/params.size()[0] # for full size
		if ((100-args.pr)/100)>result:
			print("threshold : {}".format(thres))
			return thres
		else:
			thres += 0.0001

def getPruningMask(thres):
	mask = maskgen()
	try:
		mask[0] = (torch.abs(net.conv1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[1] = (torch.abs(net.layer1_basic1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[2] = (torch.abs(net.layer1_basic2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[3] = (torch.abs(net.layer1_basic3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[4] = (torch.abs(net.layer1_basic4[0].weight.data)>thres).type(torch.FloatTensor)
		mask[5] = (torch.abs(net.layer2_basic1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[6] = (torch.abs(net.layer2_basic2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[7] = (torch.abs(net.layer2_basic3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[8] = (torch.abs(net.layer2_basic4[0].weight.data)>thres).type(torch.FloatTensor)
		mask[9] = (torch.abs(net.layer3_basic1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[10] = (torch.abs(net.layer3_basic2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[11] = (torch.abs(net.layer3_basic3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[12] = (torch.abs(net.layer3_basic4[0].weight.data)>thres).type(torch.FloatTensor)
		mask[13] = (torch.abs(net.layer4_basic1[0].weight.data)>thres).type(torch.FloatTensor)
		mask[14] = (torch.abs(net.layer4_basic2[0].weight.data)>thres).type(torch.FloatTensor)
		mask[15] = (torch.abs(net.layer4_basic3[0].weight.data)>thres).type(torch.FloatTensor)
		mask[16] = (torch.abs(net.layer4_basic4[0].weight.data)>thres).type(torch.FloatTensor)

		mask[17] = (torch.abs(net.layer2_downsample[0].weight.data)>thres).type(torch.FloatTensor)
		mask[18] = (torch.abs(net.layer3_downsample[0].weight.data)>thres).type(torch.FloatTensor)
		mask[19] = (torch.abs(net.layer4_downsample[0].weight.data)>thres).type(torch.FloatTensor)

		mask[20] = (torch.abs(net.linear[0].weight.data)>thres).type(torch.FloatTensor)
	except:
		for child in net.children():
			for param in child.conv1[0].parameters():
				mask[0] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer1_basic1[0].parameters():
				mask[1] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer1_basic2[0].parameters():
				mask[2] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer1_basic3[0].parameters():
				mask[3] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer1_basic4[0].parameters():
				mask[4] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer2_basic1[0].parameters():
				mask[5] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer2_basic2[0].parameters():
				mask[6] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer2_basic3[0].parameters():
				mask[7] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer2_basic4[0].parameters():
				mask[8] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer3_basic1[0].parameters():
				mask[9] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer3_basic2[0].parameters():
				mask[10] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer3_basic3[0].parameters():
				mask[11] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer3_basic4[0].parameters():
				mask[12] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer4_basic1[0].parameters():
				mask[13] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer4_basic2[0].parameters():
				mask[14] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer4_basic3[0].parameters():
				mask[15] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer4_basic4[0].parameters():
				mask[16] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer2_downsample[0].parameters():
				mask[17] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer3_downsample[0].parameters():
				mask[18] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.layer4_downsample[0].parameters():
				mask[19] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
		for child in net.children():
			for param in child.linear[0].parameters():
				mask[20] = (torch.abs(param.data)>thres).type(torch.FloatTensor)	
	return mask

def pruneNetwork(mask):
	for child in net.children():
		for param in child.conv1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[0].cuda())
			param.data = torch.mul(param.data,mask[0].cuda())
	for child in net.children():
		for param in child.layer1_basic1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[1].cuda())
			param.data = torch.mul(param.data,mask[1].cuda())
	for child in net.children():
		for param in child.layer1_basic2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[2].cuda())
			param.data = torch.mul(param.data,mask[2].cuda())
	for child in net.children():
		for param in child.layer1_basic3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[3].cuda())
			param.data = torch.mul(param.data,mask[3].cuda())
	for child in net.children():
		for param in child.layer1_basic4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[4].cuda())
			param.data = torch.mul(param.data,mask[4].cuda())
	for child in net.children():
		for param in child.layer2_basic1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[5].cuda())
			param.data = torch.mul(param.data,mask[5].cuda())
	for child in net.children():
		for param in child.layer2_basic2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[6].cuda())
			param.data = torch.mul(param.data,mask[6].cuda())
	for child in net.children():
		for param in child.layer2_basic3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[7].cuda())
			param.data = torch.mul(param.data,mask[7].cuda())
	for child in net.children():
		for param in child.layer2_basic4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[8].cuda())
			param.data = torch.mul(param.data,mask[8].cuda())
	for child in net.children():
		for param in child.layer3_basic1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[9].cuda())
			param.data = torch.mul(param.data,mask[9].cuda())
	for child in net.children():
		for param in child.layer3_basic2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[10].cuda())
			param.data = torch.mul(param.data,mask[10].cuda())
	for child in net.children():
		for param in child.layer3_basic3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[11].cuda())
			param.data = torch.mul(param.data,mask[11].cuda())
	for child in net.children():
		for param in child.layer3_basic4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[12].cuda())
			param.data = torch.mul(param.data,mask[12].cuda())
	for child in net.children():
		for param in child.layer4_basic1[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[13].cuda())
			param.data = torch.mul(param.data,mask[13].cuda())
	for child in net.children():
		for param in child.layer4_basic2[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[14].cuda())
			param.data = torch.mul(param.data,mask[14].cuda())
	for child in net.children():
		for param in child.layer4_basic3[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[15].cuda())
			param.data = torch.mul(param.data,mask[15].cuda())
	for child in net.children():
		for param in child.layer4_basic4[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[16].cuda())
			param.data = torch.mul(param.data,mask[16].cuda())
	for child in net.children():
		for param in child.layer2_downsample[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[17].cuda())
			param.data = torch.mul(param.data,mask[17].cuda())
	for child in net.children():
		for param in child.layer3_downsample[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[18].cuda())
			param.data = torch.mul(param.data,mask[18].cuda())
	for child in net.children():
		for param in child.layer4_downsample[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[19].cuda())
			param.data = torch.mul(param.data,mask[19].cuda())
	for child in net.children():
		for param in child.linear[0].parameters():
			param.grad.data = torch.mul(param.grad.data,mask[20].cuda())
			param.data = torch.mul(param.data,mask[20].cuda())
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
	mask = maskgen()
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
	#checkpoint = torch.load('./checkpoint/ckpt_resnet18_cs'+str(args.cs*10000)+'_20190125.t0')
	net = checkpoint['net']

if args.mode == 1:
	if args.resume:
		print('==> Resuming from checkpoint..')
		checkpoint = torch.load('./checkpoint/ckpt_resnet18_cs_'+str(args.cs*10000)+'_20190214.t0')
		net = checkpoint['net']
		best_acc = checkpoint['acc']
	else:
		net = ResNet18()
		best_acc = 0

if args.mode == 2:
	checkpoint = torch.load('./checkpoint/ckpt_resnet18_cs_'+str(args.cs*10000)+'_20190214.t0')
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
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

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
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		optimizer.step()
		pruneNetwork(mask_prune)

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
	#mask_channel = set_mask(mask_channel, 4, 1)
	mask_channel = set_mask(set_mask(mask_channel, 2, 1), 4, 0)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		pruneNetwork(mask_channel)
		#net_mask_mul(mask_zero)
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
	mask_channel = set_mask(set_mask(mask_channel, 2, 1), 4, 0)
	#mask_channel = set_mask(mask_channel, 4, 1)
	for batch_idx, (inputs, targets) in enumerate(train_loader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda()
		optimizer.zero_grad()
		inputs, targets = Variable(inputs), Variable(targets)
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()

		pruneNetwork(mask_channel)
		pruneNetwork(mask_prune)
		#net_mask_mul(mask_zero)

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
	if args.mode == 2:
		pruneNetwork(mask_prune)
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
			if args.mode == 1:
				torch.save(state, './checkpoint/ckpt_resnet18_cs_'+str(args.cs*10000)+'_20190214.t0')
			if args.mode == 2:
				torch.save(state, './checkpoint/ckpt_resnet18_cs_'+str(args.cs*10000)+'_20190214_pr_'+str(args.pr)+'.t0')
			#torch.save(state, './checkpoint/'+args.outputfile)
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

