from torch.utils.data.dataloader import DataLoader
from dataset import Dataset
from mobile_net import MobileNet
from torch.nn import MSELoss
import torch
from config import BodyTrackingConfig
from torch.autograd import Variable
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

current_path = os.path.dirname(os.path.realpath(__file__))

config = BodyTrackingConfig.from_json_file(os.path.join(current_path, "config.json"))

image_dataset = Dataset(config.dataset_image_dir)
saved_model_path = config.model_path
data_loader = DataLoader(image_dataset, batch_size=80)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_mn = MobileNet().to(device)
optim = torch.optim.Adam(model_mn.parameters(), lr=0.0001)
#optim = torch.optim.SGD(model_mn.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
loss_function = MSELoss()
epoch = 150

loss_mean_list = []
print(model_mn)
for k in range(epoch):
    print("epoch: " + str(k))
    model_mn.train()
    loss_mean = 0
    for i, data_tensor in enumerate(data_loader):
        [image, heat_map] = data_tensor
        # image = image.data.numpy()
        # image = np.sum(image[0], axis=0)
        # plt.imshow(image)
        # plt.show()
        # #print(np.shape(image))
        # #print(np.shape(heat_map))
        predict = model_mn.forward(Variable(image.type(torch.FloatTensor)).to(device))
        # #print(predict)
        output = loss_function(predict, (Variable(heat_map).to(device)))
        optim.zero_grad()
        output.backward()
        optim.step()
        loss_mean += output/(data_loader.__len__())
    loss_mean_list.append(loss_mean)
    print(loss_mean)
    torch.save(model_mn, saved_model_path + "body_tracking_model.pth")

    # train
print(loss_mean_list)
f = open(saved_model_path + "loss_mean_list.txt", "w+")
f.write(str(loss_mean_list))
f.close()