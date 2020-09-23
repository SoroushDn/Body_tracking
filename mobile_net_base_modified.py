import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2()
        self.model._modules['features'][18] = nn.Sequential(nn.ConvTranspose2d(320, 150, (3, 3), (2, 2), padding=(1, 1), output_padding=(1, 1),bias=False),
                                                            nn.BatchNorm2d(150,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                            nn.ReLU6(inplace=True),
                                                            nn.ConvTranspose2d(150, 150, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1), output_padding=(1, 1),bias=False),
                                                            nn.BatchNorm2d(150,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                            nn.ReLU6(inplace=True),
                                                            nn.ConvTranspose2d(150, 150, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1), output_padding=(1, 1),bias=False),
                                                            nn.BatchNorm2d(150,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                                            nn.ReLU6(inplace=True),
                                                            nn.Conv2d(150, 14, kernel_size=(1, 1), stride=(1, 1),bias=False))

        #self.model.classifier = nn.Sequential()


    def forward(self, input_mn):
        #print(np.shape(input_mn))
        x = self.model.features(input_mn)
        #print(np.shape(x))
        # y = self.linear(x)
        #print(np.shape(y))
        return x
