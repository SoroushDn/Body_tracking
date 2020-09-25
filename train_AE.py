from torch.utils.data.dataloader import DataLoader
from dataset import Dataset
from autoencoder_model import Autoencoder
from torch.nn import MSELoss
import torch
from config import BodyTrackingConfig
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.transforms import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

current_path = os.path.dirname(os.path.realpath(__file__))

config = BodyTrackingConfig.from_json_file(os.path.join(current_path, "config.json"))

train_transformer = transforms.Compose([
    transforms.Resize(56),
    transforms.ToTensor()
])
image_dataset = Dataset(config.dataset_image_dir, config.dataset_lbl_dir, train_transformer)
saved_model_path = config.model_path
data_loader = DataLoader(image_dataset, batch_size=80, shuffle=True)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_AE = MobileNet().to(device)

model_AE = Autoencoder()
if torch.cuda.is_available():
    model_AE.cuda()

if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()

optim = torch.optim.Adam(model_AE.parameters(), lr=0.0001, weight_decay=1e-5)
#optim = torch.optim.SGD(model_AE.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
#loss_function = MSELoss().to(device)
epoch = 150
cnt = 0
loss_mean_list = []
print(model_AE)
model_AE.train()
for k in range(epoch):
    print("epoch: " + str(k))
    #model_AE.train()
    loss_mean = 0
    cnt = 0

    for batch_data, batch_label in data_loader:
        #print(i)
        #cnt += 1
        # print(np.shape(batch_data))
        if torch.cuda.is_available():
            batch_data, batch_label = batch_data.cuda(), batch_label.cuda()
            # print(np.shape(batch_data))

    #     # image = image.data.numpy()
    #     # image = np.sum(image[0], axis=0)
    #     # plt.imshow(image)
    #     # plt.show()
    #     # #print(np.shape(image))
    #     # #print(np.shape(heat_map))

        #predict = model_AE.forward(Variable(image.type(torch.FloatTensor)).to(device))
        #predict = model_AE.forward(Variable(image.type(torch.FloatTensor).to(device)))
        optim.zero_grad()
        predict = model_AE.forward(batch_data)
    #     # #print(predict)
        output = criterion(predict, batch_label)

    #     optim.zero_grad()
        output.backward()
        optim.step()
        loss_mean += output/(data_loader.__len__())
    loss_mean_list.append(loss_mean)
    print(loss_mean)
    torch.save(model_AE, saved_model_path + "body_tracking_model.pth")

    # train
    print(cnt)
# print(loss_mean_list)
# f = open(saved_model_path + "loss_mean_list.txt", "w+")
# f.write(str(loss_mean_list))
# f.close()
