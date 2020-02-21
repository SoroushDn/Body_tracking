import os
import numpy as np
import torch
from PIL import Image
from skimage import io, transform, color
from torch.utils.data import DataLoader, IterableDataset
import scipy.io

class Dataset(IterableDataset):
    def __init__(self, dataset_image_dir, points_dir):
        self.dataset_image_dir = dataset_image_dir
        self.points_dir = points_dir
        self.list_files = sorted(os.listdir(self.dataset_image_dir))
        self.len_data = np.shape(self.list_files)[0]

    def __iter__(self):
        i = 0
        for image_path in self.list_files:
            image = io.imread(self.dataset_image_dir + image_path)
            image = transform.resize(image, (224, 224))
            points_arr = []
            file_name = self.points_dir + image_path[:-3] + "pts"
            with open(file_name) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 18:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1

            yield [torch.tensor(image.swapaxes(1,2).swapaxes(0,1)), torch.tensor(points_arr)]

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
