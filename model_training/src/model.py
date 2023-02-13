import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import os
sys.path.insert(0, '..') # Uncomment for bajcsy (Check?)
from src.resnet_dilated import Resnet34_8s, Resnet50_8s, Resnet34_Classifier, Resnet50_Classifier
import segmentation_models_pytorch as smp

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640, channels=3, resnet_type='34', pretrained=True):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.num_outputs = self.num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		if resnet_type == '34':
			self.resnet = Resnet34_8s(channels=channels, pretrained=pretrained) #Resnet50_8s(channels=channels, pretrained=True) #Resnet34_8s(channels=channels, pretrained=False)
		elif resnet_type == '50':
			self.resnet = Resnet50_8s(channels=channels, pretrained=pretrained)
		elif resnet_type == 'UNet18':
			self.resnet = smp.Unet(encoder_name='resnet18', encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=self.num_outputs)
		elif resnet_type == 'UNet34':
			self.resnet = smp.Unet(encoder_name='resnet34', encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=self.num_outputs)
		elif resnet_type == 'UNet50':
			self.resnet = smp.Unet(encoder_name='resnet50', encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=self.num_outputs)
		elif resnet_type == 'UNet101':
			self.resnet = smp.Unet(encoder_name='resnet101', encoder_weights='imagenet' if pretrained else None, in_channels=3, classes=self.num_outputs)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		output = self.resnet(x) 
		heatmaps = self.sigmoid(output[:,:self.num_keypoints, :, :])
		return heatmaps

class ClassificationModel(nn.Module):
	def __init__(self, num_classes, img_height=480, img_width=640, resnet_type = '34', channels=3):
		super(ClassificationModel, self).__init__()
		self.img_height = img_height
		self.img_width = img_width
		self.num_classes = num_classes
		self.resnet = None
		if resnet_type == '34':
			self.resnet = Resnet34_Classifier(channels=channels, num_classes=self.num_classes, pretrained=False) #Resnet50_8s(channels=channels, pretrained=True) #Resnet34_8s(channels=channels, pretrained=False)
		elif resnet_type == '50':
			self.resnet = Resnet50_Classifier(channels=channels, num_classes=self.num_classes, pretrained=False)
		self.sigmoid = torch.nn.Sigmoid()

	def forward(self, x):
		output = self.resnet(x)
		assert output.shape[1] == self.num_classes
		return self.sigmoid(output[:,:self.num_classes])

if __name__ == '__main__':
	model = KeypointsGauss(4).cuda()
	x = torch.rand((1,3,480,640)).cuda()
	result = model.forward(x)
	print(x.shape)
