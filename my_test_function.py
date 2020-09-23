import numpy as np
import torch
from PIL import Image
from skimage import io, transform
import matplotlib.pyplot as plt
from image_utility import ImageUtility
import cv2
from graphviz import Digraph
import re
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import torchvision.models as models
from torchviz import make_dot
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

#
#
# writer = SummaryWriter('/home/soroush/PycharmProjects/Bodytracking/body_tracking/runs/fashion_mnist_experiment_1')
#
def test_result_per_image(k, img):
    image_utility = ImageUtility()
    pose_predicted = []
    #image = np.expand_dims(img, axis=0)

    #predict = model.forward(image)

    heatmap_main = img
    for i in range(14):
        image_utility.print_image_arr_heat(k*100 + i + 1, heatmap_main[i])

model_mn = torch.load("/home/soroush/PycharmProjects/Bodytracking/body_tracking/breg_model/body_tracking_model.pth", map_location=torch.device('cpu'))
#
#
#filename = "4400003_0"
#k = 0
filename = "9370003_0"
k = 1
# # filename = "4400003_0"
# # k = 0
image = Image.open("/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/" + filename + ".jpg")
#image = image.astype(np.float32)
#image = transform.resize(image, (224, 224))
#image = image.swapaxes(2, 3).swapaxes(1, 2)
train_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor()
])
# image = np.expand_dims(image, axis=0)
imimage = train_transformer(image)
# predictedim = model_mn.forward(imimage.unsqueeze(0))
predictedim = model_mn.forward(imimage.unsqueeze(0),torch.tensor([1], dtype=torch.float),torch.tensor([0], dtype=torch.float))
prednp = predictedim.data.numpy()
print(prednp)
prednp = np.squeeze(prednp, axis=0)
prednp_sum = np.sum(prednp, axis=0)
print(predictedim)
plt.imshow(prednp_sum)
plt.show()
test_result_per_image(k, prednp)
# #
heatmap = np.load("/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/" + filename + ".npy")
print(heatmap)
heatmap = np.sum(heatmap, axis=2)
print(heatmap)
plt.imshow(heatmap, vmax=1, vmin=0)
plt.show()
#
#
#
#
# # def make_dot(var):
# #     node_attr = dict(style='filled',
# #                      shape='box',
# #                      align='left',
# #                      fontsize='12',
# #                      ranksep='0.1',
# #                      height='0.2')
# #     dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
# #     seen = set()
# #
# #     def add_nodes(var):
# #         if var not in seen:
# #             if isinstance(var, Variable):
# #                 value = '('+(', ').join(['%d'% v for v in var.size()])+')'
# #                 dot.node(str(id(var)), str(value), fillcolor='lightblue')
# #             else:
# #                 dot.node(str(id(var)), str(type(var).__name__))
# #             seen.add(var)
# #             if hasattr(var, 'previous_functions'):
# #                 for u in var.previous_functions:
# #                     dot.edge(str(id(u[0])), str(id(var)))
# #                     add_nodes(u[0])
# #     add_nodes(var.creator)
# #     return dot
# #
# writer.add_image('one_image', imimage[0])
#
#
# writer.add_graph(model_mn, imimage.type(torch.FloatTensor))
# writer.close()