import torchvision.models as models
import torch.nn as nn
from config import BodyTrackingConfig
import glob
import sys
import os
import cv2

class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.MobileNetV2()

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
        self.model.classifier = nn.Sequential()

    def forward(self, input_mn):
        x = self.model.features(input_mn)
        return x


config = BodyTrackingConfig.from_json_file(os.path.join(current_path, "config.json"))
images_path = config.dataset_image_dir
labels_path = config.points_dir
saved_labels_path = labels_path
saved_images_path = images_path


images = sorted(glob.glob(saved_images_path + "*.png"),key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
labels = sorted(glob.glob(saved_labels_path + "*.nyp"),key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

train_loader = [(images[i], labels[i]) for i in range(0,516,1)]
test_loader = [(images[i], labels[i]) for i in range(516,len(images),1)]

batch_size = 1
n_iters = 3000 # how many times weights are updated
num_epochs = n_iters/(len(images)/batch_size)
num_epochs = int(num_epochs)

image_dataset = Dataset(config.dataset_image_dir)
saved_model_path = config.model_path
data_loader = DataLoader(image_dataset, batch_size=80)
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_AE = MobileNet().to(device)

model_mn = MobileNet()
if torch.cuda.is_available():
    model_mn.cuda()

if torch.cuda.is_available():
    criterion = nn.MSELoss().cuda()
else:
    criterion = nn.MSELoss()

optim = torch.optim.Adam(model_mn.parameters(), lr=0.01, weight_decay=1e-5)
#optim = torch.optim.SGD(model_AE.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-5)
#loss_function = MSELoss().to(device)
epoch = 150
cnt = 0
loss_mean_list = []
print(model_mn)
for k in range(epoch):
    print("epoch: " + str(k))
    #model_AE.train()
    loss_mean = 0
    cnt = 0
    for i, data_tensor in enumerate(data_loader):
        #print(i)
        #cnt += 1
        [image, heat_map] = data_tensor
        if torch.cuda.is_available():
            image = Variable(image.type(torch.FloatTensor).cuda())
            heat_map = Variable(heat_map.cuda())
        else:
            image = Variable(image.type(torch.FloatTensor))
            heat_map = Variable(heat_map)

    #     # image = image.data.numpy()
    #     # image = np.sum(image[0], axis=0)
    #     # plt.imshow(image)
    #     # plt.show()
    #     # #print(np.shape(image))
    #     # #print(np.shape(heat_map))
        optim.zero_grad()
        #predict = model_AE.forward(Variable(image.type(torch.FloatTensor)).to(device))
        #predict = model_AE.forward(Variable(image.type(torch.FloatTensor).to(device)))
        predict = model_mn.forward(image)
    #     # #print(predict)
        output = criterion(torch.flatten(predict.to(dtype=torch.float32)), torch.flatten(heat_map.to(dtype=torch.float32)))
    #     optim.zero_grad()
        output.backward()
        optim.step()
        loss_mean += output/(data_loader.__len__())
    loss_mean_list.append(loss_mean)
    print(loss_mean)
    torch.save(model_mn, saved_model_path + "body_tracking_model.pth")

    # train
    print(cnt)
# print(loss_mean_list)
