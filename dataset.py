import os
import numpy as np
import torch
from PIL import Image
from skimage import io, transform, color
from torch.utils.data import DataLoader, IterableDataset
import scipy.io
import random

def generate_hm(self, height, width, landmarks, s=1.0, upsample=True):
    """ Generate a full Heat Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        landmarks    : [(x1,y1),(x2,y2)...] containing landmarks
        s : Standard deviation of gaussian function
    """


    Nlandmarks = len(landmarks)
    hm = np.zeros((height, width, Nlandmarks // 2), dtype=np.float32)

    j = 0
    for i in range(0, Nlandmarks, 2):

        if upsample:
            x = (112 - float(landmarks[i]) * 224)
            y = (112 - float(landmarks[i + 1]) * 224)
        else:
            x = landmarks[i]
            y = landmarks[i + 1]

        x = int(x // 4)
        y = int(y // 4)

        hm[:, :, j] = self.__gaussian_k(x, y, s, height, width)
        j += 1
    return hm


class Dataset(IterableDataset):
    def __init__(self, dataset_image_dir):
        self.dataset_image_dir = dataset_image_dir
        self.list_files = sorted(os.listdir(self.dataset_image_dir))
        self.len_data = np.shape(self.list_files)[0]

    def __iter__(self):
        i = 0
        for image_path in self.list_files:
            if image_path.endswith(".jpg"):
                image = io.imread(self.dataset_image_dir + image_path)
                width = np.shape(image)[1]
                height = np.shape(image)[0]
                x_center = width / 2
                y_center = height / 2
                image = transform.resize(image, (224, 224))
                file_name = self.dataset_image_dir + image_path[0:-4] + ".npy"
                points = np.load(file_name)

            yield [torch.tensor(image.swapaxes(1, 2).swapaxes(0, 1)), torch.tensor(points.swapaxes(1, 2).swapaxes(0, 1))]

    def __len__(self):
        return self.len_data



# if __name__ == "__main__":
#
#     dataset = Dataset("/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/images/", "/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/points/")
#     j = 1
#
#     data_loader = DataLoader(dataset, batch_size=5)
#     for i, data in enumerate(data_loader):
#         [image, points] = data
#         print(j)
#         print(np.shape(image))
#         print(points)
