import torchvision.models as models
import torch.nn as nn
import numpy as np



class MobileNet(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.model = models.MobileNetV2()
        self.model.classifier = nn.Sequential()
        self.linear = nn.Linear(self.model.last_channel, num_features)

    def forward(self, input_mn):
        #print(np.shape(input_mn))
        x = self.model(input_mn)
        #print(np.shape(x))
        y = self.linear(x)
        #print(np.shape(y))
        return y
