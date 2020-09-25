import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.cnn_en1 = nn.Sequential(nn.Conv2d(3, 64, (3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())
        self.cnn_en2 = nn.Sequential(nn.Conv2d(64, 128, (3,3), stride=(1,1), padding=(1,1)),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())
        self.maxpooling_en1 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        self.cnn_en3 = nn.Sequential(nn.Conv2d(128, 256, (3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU())
        self.maxpooling_en2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))
        self.cnn_en4 = nn.Sequential(nn.Conv2d(256, 512, (3, 3), stride=(1, 1), padding=(1, 1)),
                                     nn.BatchNorm2d(512),
                                     nn.ReLU())
        self.maxpooling_en3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1,1))

        self.cnn_dec1 = nn.Sequential(nn.Conv2d(512, 256, (3,3), stride=(1,1), padding=(1,1)),
                                      nn.ReLU())
        self.upsampling_dec1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cnn_dec2 = nn.Sequential(nn.Conv2d(512, 128, (3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU())
        self.upsampling_dec2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cnn_dec3 = nn.Sequential(nn.Conv2d(256, 64, (3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU())
        self.upsampling_dec3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.cnn_dec4 = nn.Sequential(nn.Conv2d(128, 14, (3, 3), stride=(1, 1), padding=(1, 1)),
                                      nn.ReLU())

    def forward(self, input_AE):
        #print(np.shape(input_mn))
        c1 = self.cnn_en1(input_AE)
        # print("c1")
        # print(np.shape(c1))
        c2 = self.cnn_en2(c1)
        # print("c2")
        # print(np.shape(c2))
        m1 = self.maxpooling_en1(c2)
        # print("m1")
        # print(np.shape(m1))
        c3 = self.cnn_en3(m1)
        # print("c3")
        # print(np.shape(c3))
        m2 = self.maxpooling_en2(c3)
        # print("m2")
        # print(np.shape(m2))
        c4 = self.cnn_en4(m2)
        # print("c4")
        # print(np.shape(c4))
        m3 = self.maxpooling_en3(c4)
        # print("m3")
        # print(np.shape(m3))
        c21 = self.cnn_dec1(m3)
        # print("c21")
        # print(np.shape(c21))
        u1 = self.upsampling_dec1(c21)
        # print("u1")
        # print(np.shape(u1))
        c22 = self.cnn_dec2(torch.cat((m2,u1), dim=1))
        # print("c22")
        # print(np.shape(c22))
        u2 = self.upsampling_dec2(c22)
        # print("u2")
        # print(np.shape(u2))
        c23 = self.cnn_dec3(torch.cat((m1,u2), dim=1))
        # print("c23")
        # print(np.shape(c23))
        u3 = self.upsampling_dec3(c23)
        # print("u3")
        # print(np.shape(u3))
        c24 = self.cnn_dec4(torch.cat((c1,u3), dim=1))
        # print("c24")
        # print(np.shape(c24))
        #print(np.shape(x))
        # y = self.linear(x)
        #print(np.shape(y))
        return c24
