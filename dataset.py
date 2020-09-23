import os
import numpy as np
import torch
from PIL import Image
from skimage import io, transform, color
import cv2
from torch.utils.data import DataLoader, IterableDataset, Dataset
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


class Dataset(Dataset):
    def __init__(self, dataset_image_dir, dataset_lbl_dir, transform):
        self.dataset_image_dir = dataset_image_dir
        self.dataset_lbl_dir = dataset_lbl_dir
        self.list_files = [file_name for file_name in os.listdir(self.dataset_image_dir) if file_name.endswith(".jpg")]
        self.labels = [file_name[0:-4] for file_name in self.list_files]
        self.transform = transform

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):

        image = Image.open(self.dataset_image_dir + self.list_files[idx])
        image = self.transform(image)

        points = np.load(self.dataset_lbl_dir + self.labels[idx] + ".npy")
        points = points.swapaxes(1, 2)
        points = points.swapaxes(0, 1)

        return image, points




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
