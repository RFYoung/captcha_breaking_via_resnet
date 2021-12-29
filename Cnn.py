from collections import OrderedDict

import torch
import torch.nn as nn

class CnnModel(nn.Module):
    def __init__(self, char_classes,str_len ,input_shape=(3, 64, 128)):
        super(CnnModel, self).__init__()
        channels = [32, 64, 128, 256, 256]
        cnn_kernel = 3
        cnn_pool = 2
        features_modules = OrderedDict()
        self.str_len = str_len

        in_channel = input_shape[0]
        for block, (n_channel) in enumerate(channels):
            features_modules[f'conv1-{block}'] = nn.Conv2d(in_channel, n_channel, cnn_kernel, padding=1)
            features_modules[f'bn1-{block}'] = nn.BatchNorm2d(n_channel)
            features_modules[f'relu1-{block}'] = nn.ReLU(inplace=True)
            features_modules[f'conv2-{block}'] = nn.Conv2d(n_channel, n_channel, cnn_kernel, padding=1)
            features_modules[f'bn2-{block}'] = nn.BatchNorm2d(n_channel)
            features_modules[f'relu2-{block}'] = nn.ReLU(inplace=True)
            features_modules[f'pool-{block}'] = nn.MaxPool2d(cnn_pool, stride=2)
            in_channel = n_channel
        features_modules[f'dropout'] = nn.Dropout(0.25, inplace=True)

        self.features = nn.Sequential(features_modules)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=channels[-1]*2*6, out_features=2048),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=2048, out_features=2048),
            nn.ReLU(inplace=True),
        )
        for i in range(str_len):
            setattr(self, "linear_final-%d" % i, nn.Linear(2048,char_classes))

    def forward(self, x:torch.Tensor):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return torch.stack([getattr(self,"linear_final-%d" %i)(x) for i in range(self.str_len)],dim=1)

