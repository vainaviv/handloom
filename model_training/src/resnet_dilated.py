import torch.nn as nn
import torchvision.models as models
from model_training.src.resnet import resnet34, resnet50, resnet101, resnet152

class Resnet34_8s(nn.Module):
    def __init__(self, num_classes=1000, channels=4, pretrained=False):
        super(Resnet34_8s, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                       channels=channels,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)
        self.resnet34_8s = resnet34_8s
        self._normal_initialization(self.resnet34_8s.fc)
        
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet34_8s(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x

class Resnet34_Classifier(nn.Module):
    def __init__(self, num_classes=1, channels=3, pretrained=False):
        super(Resnet34_Classifier, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_classifier = resnet34(fully_conv=False,
                                       channels=channels,
                                       num_classes=num_classes,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)

        self.resnet34_classifier = resnet34_classifier
        self._normal_initialization(self.resnet34_classifier.fc)
        
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        x = self.resnet34_classifier(x)
        return x

class Resnet50_Classifier(nn.Module):
    def __init__(self, num_classes=1, channels=3, pretrained=False):
        super(Resnet50_Classifier, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_classifier = resnet50(fully_conv=False,
                                       channels=channels,
                                       num_classes=num_classes,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)

        self.resnet50_classifier = resnet50_classifier
        self._normal_initialization(self.resnet50_classifier.fc)
        
    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        x = self.resnet50_classifier(x)
        return x

class Resnet50_8s(nn.Module):
    def __init__(self, num_classes=1000, channels=4, pretrained=False):
        super(Resnet50_8s, self).__init__()
        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet50_8s = resnet50(fully_conv=True,
                                       pretrained=pretrained,
                                       output_stride=8,
                                       remove_avg_pool_layer=True)
        
        # Randomly initialize the 1x1 Conv scoring layer
        resnet50_8s.fc = nn.Conv2d(resnet50_8s.inplanes, num_classes, 1)
        self.resnet50_8s = resnet50_8s
        self._normal_initialization(self.resnet50_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()
        
    def forward(self, x):
        input_spatial_dim = x.size()[2:]
        x = self.resnet50_8s(x)
        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)
        return x