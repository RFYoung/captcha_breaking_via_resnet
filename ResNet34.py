from collections import OrderedDict

import torch
import torch.nn as nn

from torch import Tensor

class ResNet34Model(nn.Module):
    '''
    The ResNet-34-Like model
    '''
    def __init__(self, char_classes,str_len ,input_shape=(3, 64, 128)):
        super(ResNet34Model, self).__init__()

        features_modules = OrderedDict()
        self.str_len = str_len

        # channels in chunks
        channels = [32, 64, 128, 256]

        # the beginning part before basic blocks
        begin_out_channel = channels[0]
        self.begin = nn.Sequential(OrderedDict({
            "begin-conv": nn.Conv2d(input_shape[0], begin_out_channel, kernel_size=7, stride=2, padding=3),
            "begin-bn" : nn.BatchNorm2d(begin_out_channel),
            "begin-relu" : nn.ReLU(inplace=True),
            "begin-maxpool" : nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        }))
        
        # construct all the basic blocks
        in_channel = begin_out_channel
        basic_blocks_list = [3, 4, 6, 3]
        resnet_stride = [1,2,2,2]
        for i, (stride, net_channel, num_blocks) in enumerate((zip(resnet_stride,channels, basic_blocks_list))) :
            
            if stride != 1:
                shortcut_downsample = nn.Sequential(
                    nn.Conv2d(in_channel,net_channel,kernel_size=1,stride=stride),
                    nn.BatchNorm2d(net_channel)
                )
            else:
                shortcut_downsample = None
            features_modules[f'basic-{i}-0'] = FeatureBasicBlock(in_channel,net_channel,stride,shortcut_downsample)
            
            # basic blocks without downsamples
            for j in range(1,num_blocks):
                features_modules[f'basic-{i}-{j}'] = FeatureBasicBlock(net_channel,net_channel,1,shortcut_downsample=None)
                
                
            in_channel = net_channel

        self.features = nn.Sequential(features_modules)
        
        
        # 4 parallel classifier at the bottom
        for i in range(str_len):
            setattr(self, "linear_final-%d" % i, nn.Linear(channels[-1]*2*6,char_classes))

    def forward(self, x:Tensor):
        x = self.begin(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return torch.stack([getattr(self,"linear_final-%d" %i)(x) for i in range(self.str_len)],dim=1)


# The basic block of ResNet32
class FeatureBasicBlock(nn.Module):
    def __init__(self,in_channel,n_channel, stride=1, shortcut_downsample:nn.modules=None):
        super(FeatureBasicBlock, self).__init__()
        self.shortcut_downsample = shortcut_downsample

        self.feature_conv1= nn.Conv2d(in_channel, n_channel, kernel_size=3, padding=1, stride=stride)
        self.feature_bn1 = nn.BatchNorm2d(n_channel)
        
        self.feature_conv2 = nn.Conv2d(n_channel, n_channel, kernel_size=3, padding=1)
        self.feature_bn2 = nn.BatchNorm2d(n_channel)
        
        self.feature_relu = nn.ReLU(inplace=True)

        
    def forward(self,x:Tensor) -> Tensor:
        if self.shortcut_downsample is None:
            identity = x
        else:
            identity = self.shortcut_downsample(x)
        out = self.feature_conv1(x)
        out = self.feature_bn1(out)
        out = self.feature_relu(out)
        out = self.feature_conv2(out)
        out = self.feature_bn2(out)
        out += identity
        out = self.feature_relu(out)

        return out