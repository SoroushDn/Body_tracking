import torchvision.models as models
import torch.nn as nn
import numpy as np



class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.MobileNetV2()
        # self.model._modules['features'][1] = nn.Sequential()
        # self.model._modules['features'][2] = nn.Sequential()
        # self.model._modules['features'][3] = nn.Sequential()
        # self.model._modules['features'][4] = nn.Sequential()
        # self.model._modules['features'][5] = nn.Sequential()
        # self.model._modules['features'][6] = nn.Sequential()
        # self.model._modules['features'][7] = nn.Sequential()
        # self.model._modules['features'][8] = nn.Sequential()
        # self.model._modules['features'][9] = nn.Sequential()
        # self.model._modules['features'][10] = nn.Sequential()
        # self.model._modules['features'][11] = nn.Sequential()
        # self.model._modules['features'][12] = nn.Sequential()
        # self.model._modules['features'][13] = nn.Sequential()
        # self.model._modules['features'][14] = nn.Sequential()
        # self.model._modules['features'][15] = nn.Sequential()
        # self.model._modules['features'][16] = nn.Sequential()
        # self.model._modules['features'][17] = nn.Sequential()
        # self.model._modules['features'][18] = nn.Sequential()
        # self.model = nn.Sequential(nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
        #                            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        #                            nn.ReLU6(inplace=True))
        self.model._modules['features'][18] = nn.Sequential(nn.ConvTranspose2d(320, 150, (3, 3), (2, 2), padding=(1, 1), output_padding=(1, 1)),
                                                            nn.BatchNorm2d(150),
                                                            nn.ReLU(),
                                                            nn.ConvTranspose2d(150, 150, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1), output_padding=(1, 1)),
                                                            nn.BatchNorm2d(150),
                                                            nn.ReLU(),
                                                            nn.ConvTranspose2d(150, 150, kernel_size=(3, 3), stride=(2, 2),padding=(1, 1), output_padding=(1, 1)),
                                                            nn.BatchNorm2d(150),
                                                            nn.ReLU(),
                                                            nn.Conv2d(150, 14, kernel_size=(1, 1), stride=(1, 1)))
        # self.model._modules['features'][18] = nn.Sequential(nn.UpsamplingNearest2d(scale_factor=2),
        #                                                     nn.Conv2d(in_channels=320, out_channels=150, kernel_size=3, padding=1),
        #                                                     nn.BatchNorm2d(150),
        #                                                     nn.ReLU(),
        #                                                     nn.UpsamplingNearest2d(scale_factor=2),
        #                                                     nn.Conv2d(in_channels=150, out_channels=150, kernel_size=3, padding=1),
        #                                                     nn.BatchNorm2d(150),
        #                                                     nn.ReLU(),
        #                                                     nn.UpsamplingNearest2d(scale_factor=2),
        #                                                     nn.Conv2d(in_channels=150, out_channels=150, kernel_size=3,padding=1),
        #                                                     nn.BatchNorm2d(150),
        #                                                     nn.ReLU(),
        #                                                     nn.Conv2d(150, 14, kernel_size=(1, 1), stride=(1, 1)))
        #self.model.classifier = nn.Sequential()


    def forward(self, input_mn):
        #print(np.shape(input_mn))
        x = self.model.features(input_mn)
        #print(np.shape(x))
        # y = self.linear(x)
        #print(np.shape(y))
        return x
