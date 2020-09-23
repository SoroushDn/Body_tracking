import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage import transform
from PIL import Image
from image_utility import ImageUtility

def __gaussian_k(x0, y0, sigma, width, height):
    """ Make a square gaussian kernel centered at (x0, y0) with sigma as SD.
    """
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_hm(height, width, landmarks, s=1.0, upsample=True):
    """ Generate a full Heap Map for every landmarks in an array
    Args:
        height    : The height of Heat Map (the height of target output)
        width     : The width  of Heat Map (the width of target output)
        joints    : [(x1,y1),(x2,y2)...] containing landmarks
        maxlenght : Lenght of the Bounding Box
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

        hm[:, :, j] = __gaussian_k(x, y, s, height, width)
        j += 1
    return hm


#dir_heatmap = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/'
#dir_heatmap = '/media/data/Soroush_data/body_tracking/heatmap/'
dir_heatmap = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/test_images_crop/'
keypoints_start = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 2, 3]
keypoints_end = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 13, 8, 9]
# skeleton_color = [[255, 255, 255],[255, 0, 0],[0, 255, 0],
#                   [0, 0,255],[255, 255, 0], [0, 255, 255],
#                   [255, 0, 255],[192, 192, 192],[128, 128, 0],
#                   [128, 0, 0],[128, 128, 0],[0, 128,0],
#                   [128, 0, 128]]

skeleton_color = [[100, 100, 100],[101, 101, 101],[102, 102, 102],
                  [103, 103, 103],[104, 104, 104], [105, 105, 105],
                  [106, 106, 106],[107, 107, 107], [108, 108, 108],
                  [109, 109, 109],[110, 110, 110], [111, 111, 111],
                  [112, 112, 112]]


image_utility = ImageUtility()

for file_ in os.listdir(dir_heatmap):
    if file_.endswith(".jpg"):
        npy_file = file_[0:-4] + ".npy"
        points_file = file_[0:-4] + ".pts"


        points_arr = []
        points_x_arr = []
        points_y_arr = []
        with open(dir_heatmap + points_file) as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if 3 < cnt < 18:
                    x_y_pnt = line.strip()
                    x = float(x_y_pnt.split(" ")[0])
                    y = float(x_y_pnt.split(" ")[1])
                    points_arr.append(x)
                    points_arr.append(y)
                    points_x_arr.append(x)
                    points_y_arr.append(y)
                line = fp.readline()
                cnt += 1
        # generate hm
        heatmap_landmark = generate_hm(56, 56, points_arr, s=1.0)
        file_name_save = npy_file
        hm_f = dir_heatmap + file_name_save
        np.save(hm_f, heatmap_landmark)

        points_x_arr = 112 - 224*np.array(points_x_arr)
        #points_x_arr = points_x_arr / 4
        points_y_arr = 112 - 224*np.array(points_y_arr)
        #points_y_arr = points_y_arr / 4
        points_arr = 112 - 224*np.array(points_arr)
        #points_arr = points_arr / 4
        img = cv2.imread(dir_heatmap + file_)
        #img = Image.fromarray((img).astype(np.uint8))
        #img = img.resize((56, 56))
        #img = np.array(img)
        img = 0 * img
        for i in range(len(keypoints_start)):
            point_start = tuple(np.array([points_x_arr[keypoints_start[i]], points_y_arr[keypoints_start[i]]]).astype(int))
            point_end = tuple(np.array([points_x_arr[keypoints_end[i]], points_y_arr[keypoints_end[i]]]).astype(int))
            img = cv2.line(img, point_start, point_end, color=skeleton_color[i], thickness=5)
        #img = cv2.dilate(img, np.ones([2,2]))
        img = cv2.resize(img, (56,56))
        img = img[:, :, 0]
        heatmap = np.load(dir_heatmap + npy_file)
        mask_joints = heatmap == 0
        mask_joints = mask_joints.astype(int)
        body_shape = np.zeros([56, 56, 14])
        for j in range(np.shape(heatmap)[2]):
            body_shape[:,:,j] = -1*img[:, :]

        heatmap = heatmap + body_shape * mask_joints
        heatmap_sum = np.sum(heatmap, axis=2)
        np.save(dir_heatmap + npy_file, heatmap)
        #plt.imshow(heatmap_sum)
        #plt.show()
        # image_utility.print_image_arr(1,img, points_x_arr//4, points_y_arr//4)
        # maxx = np.max(img)
        # im = Image.fromarray((img).astype(np.uint8))
        # file_name = "test_my_skelton"
        # print(os.getcwd())
        # im.save(str(file_name) + '.jpg')



