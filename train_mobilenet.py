from torch.utils.data.dataloader import DataLoader
from dataset import Dataset
from mobile_net import MobileNet
from torch.nn import MSELoss
import torch
from config import BodyTrackingConfig

config = BodyTrackingConfig.from_json_file("config.json")

image_dataset = Dataset(config.dataset_dir)

data_loader = DataLoader(image_dataset, batch_size=5)
model_mn = MobileNet(28)
optim = torch.optim.Adam(model_mn.parameters(), lr=0.0001)
loss_function = MSELoss()
epoch = 5

loss_mean_list = []

for k in range(epoch):
    print("epoch: " + str(k))
    model_mn.train()
    loss_mean = 0
    for i, data_tensor in enumerate(data_loader):
        [image, points] = data_tensor
        predict = model_mn.forward(image.type(torch.FloatTensor))
        output = loss_function(predict, points)
        optim.zero_grad()
        output.backward()
        optim.step()
        loss_mean += output/(data_loader.__len__())
    loss_mean_list.append(loss_mean)
    print(loss_mean)
    # train
