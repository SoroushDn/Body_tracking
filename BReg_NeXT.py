import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

def complex_map(x, alpha, beta):
    output = torch.div(torch.atan(torch.div(torch.mul(x, alpha), torch.sqrt(torch.add(torch.pow(beta, 2),1)))), torch.mul(alpha, torch.sqrt(torch.add(torch.pow(beta, 2), 1))))
    return output

class residual_block(nn.Module):
    def __init__(self, n_blocks, in_ch, out_ch, downsample=False, downsample_strides=2, batch_norm=True, activation=1):
        super(residual_block, self).__init__()
        if not downsample:
            downsample_strides = 1
        self.downsample_strides = downsample_strides
        block = []
        for i in range(n_blocks):
            block.append(nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), stride=downsample_strides, padding=(1,1)))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ELU())
            block.append(nn.Conv2d(out_ch, out_ch, kernel_size=(3, 3), stride=(1,1), padding=(1, 1)))
            if batch_norm:
                block.append(nn.BatchNorm2d(out_ch))
            block.append(nn.ELU())

        self.conv = nn.Sequential(*block)
        if self.downsample_strides > 1:
            self.avg_pooling = nn.AvgPool2d(kernel_size=self.downsample_strides, stride=self.downsample_strides)

    def forward(self, x, alpha, beta):
        output = complex_map(x, alpha, beta)
        bs, n_ch, m, n = np.shape(self.conv(x))
        padding = nn.ZeroPad2d(int((np.shape(self.conv(x))[1]-np.shape(output)[1])/2))
        zero_mat = torch.zeros([bs, int(np.shape(self.conv(x))[1]-np.shape(output)[1]), m, n])
        if torch.cuda.is_available():
            zero_mat = zero_mat.cuda()
        # print("conv")
        # print(np.shape(self.conv(x)))

        if self.downsample_strides > 1:
            # print("avg")
            # print(np.shape(self.avg_pooling(output)))
            return torch.cat((self.avg_pooling(output), zero_mat), dim=1) + self.conv(x)
        return output + self.conv(x)




class BREg_NeXT(nn.Module):
    def __init__(self):
        super(BREg_NeXT, self).__init__()
        self.layer_first = nn.Conv2d(3, 32, 3, padding=(1,1))
        self.block_1 = residual_block(7, 32, 32)
        self.block_2 = residual_block(1, 32, 64,downsample=True)
        self.block_3 = residual_block(8, 64, 64)
        self.block_4 = residual_block(1, 64, 128, downsample=True)
        self.block_5 = residual_block(7, 128, 128)
        self.layer_last = nn.Conv2d(128, 14, 3,padding=(1,1))
            # models.mobilenet_v2()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_mn, alpha, beta):
        #print(np.shape(input_mn))
        x = self.layer_first(input_mn)
        #print(np.shape(x))
        x = self.block_1(x, alpha, beta)
        x = self.block_2(x, alpha, beta)
        x = self.block_3(x, alpha, beta)
        x = self.block_4(x, alpha, beta)
        x = self.block_5(x, alpha, beta)
        x = self.layer_last(x)
        #print(np.shape(x))
        # y = self.linear(x)
        #print(np.shape(y))
        return x
