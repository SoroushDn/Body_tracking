from image_utility import ImageUtility
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

def print_image_arr(k, image, landmarks_x, landmarks_y):
    plt.figure()
    plt.imshow(image)
    implot = plt.imshow(image)

    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='black', s=20)
    plt.scatter(x=landmarks_x[:], y=landmarks_y[:], c='white', s=15)
    plt.axis('off')
    plt.savefig('sss' + str(k) + '.png', bbox_inches='tight')
    # plt.show()
    plt.clf()

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

def generate_hm_and_save():
    images_dir = IbugConf.images_dir
    npy_dir = IbugConf.lbls_dir

    for file in os.listdir(images_dir):
        if file.endswith(".pts"):
            points_arr = []
            file_name = os.path.join(images_dir, file)
            file_name_save = str(file)[:-3] + "npy"
            hm_f = npy_dir + file_name_save
            # imgpr.print_image_arr_heat(1, hm, print_single=False)

            np.save(hm_f, hm)
            with open(file_name) as fp:
                line = fp.readline()
                cnt = 1
                while line:
                    if 3 < cnt < 72:
                        x_y_pnt = line.strip()
                        x = float(x_y_pnt.split(" ")[0])
                        y = float(x_y_pnt.split(" ")[1])
                        points_arr.append(x)
                        points_arr.append(y)
                    line = fp.readline()
                    cnt += 1
            hm = generate_hm(56, 56, np.array(points_arr), 1.5, False)


class InputDataSize:
    image_input_size = 224
    landmark_len = 28
    landmark_face_len = 54
    landmark_nose_len = 18
    landmark_eys_len = 24
    landmark_mouth_len = 40
    pose_len = 3

class IbugConf:

    images_dir = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_before_heatmap/'
    lbls_dir = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train_before_heatmap_npy/'


    tf_train_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/train.tfrecords'
    tf_test_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/test.tfrecords'
    tf_evaluation_path = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/evaluation.tfrecords'

    tf_train_path_heatmap = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/'
    tf_test_path_heatmap = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/test_heatmap.tfrecords'
    tf_evaluation_path_heatmap = '/media/ali/extradata/facial_landmark_ds/from_ibug/train_set/evaluation_heatmap.tfrecords'

    # origin_number_of_all_sample = 3148  # afw, train_helen, train_lfpw
    # origin_number_of_train_sample = 2834  # 95 % for train
    # origin_number_of_evaluation_sample = 314  # 5% for evaluation

    origin_number_of_all_sample = 1000  # afw, train_helen, train_lfpw
    origin_number_of_train_sample = 1000  # 95 % for train
    origin_number_of_evaluation_sample = 0  # 5% for evaluation

    augmentation_factor = 3  # create 100 image from 1
    augmentation_factor_rotate = 20  # create 100 image from 1

    sum_of_train_samples = origin_number_of_train_sample * augmentation_factor
    sum_of_validation_samples = origin_number_of_evaluation_sample * augmentation_factor

    img_path_prefix = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/images/'
    pts_path_prefix = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/points/'

    rotated_img_path_prefix = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/images_rotated/'
    rotated_pts_path_prefix = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/points_rotated/'

    before_heatmap_img_path_prefix = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/'



image_utility = ImageUtility()

"""
import random
png_file_arr = []
png_file_name = []
for file in sorted(os.listdir(IbugConf.img_path_prefix)):
    if file.endswith(".jpg") or file.endswith(".png"):
        png_file_arr.append(os.path.join(IbugConf.img_path_prefix, file))
        png_file_name.append(file)

number_of_samples = IbugConf.origin_number_of_all_sample
# number_of_samples = 1000



# rotate
for i in range(number_of_samples):
    img_file = png_file_arr[i]
    pts_file = os.path.join(IbugConf.pts_path_prefix, png_file_name[i])[:-3] + "pts"
    print( 'image:  ' + img_file)
    points_arr = []
    points_x_arr = []
    points_y_arr = []
    with open(pts_file) as fp:
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

    img = Image.open(img_file)
    img = np.array(img)

    resized_img = img
    landmark_arr_xy = points_arr
    # print_image_arr(10000 * (i + 1) + 1, resized_img, points_x_arr, points_y_arr)

    # heatmap_lbl_img = np.zeros(shape=[resized_img.shape[0], resized_img.shape[1]])  # 2d is ok
    # for j in range(0, len(landmark_arr_xy), 2):
    #     heatmap_lbl_img[int(landmark_arr_xy[j + 1]), int(landmark_arr_xy[j])] = 255

    for j in range(IbugConf.augmentation_factor_rotate):
        image_utility.random_rotate(resized_img, landmark_arr_xy,
                                    IbugConf.rotated_img_path_prefix + str(10000 * (i + 1) + j),
                                    IbugConf.rotated_pts_path_prefix + str(10000 * (i + 1) + j), str(10000 * (i + 1) + j))




        number_of_samples = IbugConf.origin_number_of_all_sample
        number_of_train = IbugConf.origin_number_of_train_sample
        number_of_evaluation = IbugConf.origin_number_of_evaluation_sample

"""
png_file_arr = []
png_file_name = []

for file in sorted(os.listdir(IbugConf.rotated_img_path_prefix)):
    if file.endswith(".jpg") or file.endswith(".png"):
        png_file_arr.append(os.path.join(IbugConf.rotated_img_path_prefix, file))
        png_file_name.append(file)

number_of_samples = IbugConf.origin_number_of_all_sample * IbugConf.augmentation_factor_rotate
# number_of_samples = 1000


npy_dir = IbugConf.before_heatmap_img_path_prefix

for i in range(len(os.listdir(IbugConf.rotated_img_path_prefix))):
    print(i)
    img_file = png_file_arr[i]
    pts_file = os.path.join(IbugConf.rotated_pts_path_prefix, png_file_name[i])[:-3] + "pts"

    points_arr = []
    points_x_arr = []
    points_y_arr = []
    with open(pts_file) as fp:
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

    img = Image.open(img_file)

    '''normalize image'''
    resized_img = np.array(img) / 255.0

    #resized_img = transform.resize(resized_img, (224, 224))
    '''crop data: we add a small margin to the images'''
    landmark_arr_xy, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=points_arr,
                                                                                    scale_factor_x=1,
                                                                                    scale_factor_y=1)

    '''augment the images, then normalize the landmarks based on the hyperface method'''
    for k in range(IbugConf.augmentation_factor):
        '''save the origin image as well'''
        #print(k)
        if k == 0:
            landmark_arr_flat_aug = landmark_arr_xy
            img_aug = resized_img

        else:
            '''save the augmented images'''
            if k % 2 == 0:
                #print(np.shape(resized_img))
                landmark_arr_flat_aug, img_aug = image_utility.random_augmentation(landmark_arr_xy, resized_img)
            else:
                landmark_arr_flat_aug, img_aug = image_utility.augment(resized_img, landmark_arr_xy)

        '''test '''
        #print_image_arr(k, img_aug, [], [])

        '''again resize image to 224*224 after augmentation'''
        resized_img_new = transform.resize(img_aug,
                                          (InputDataSize.image_input_size, InputDataSize.image_input_size, 3)
                                          , anti_aliasing=True)

        #print_image_arr(k, resized_img_new, [], [])

        dims = resized_img.shape
        height = dims[0]
        width = dims[1]
        scale_factor_y = InputDataSize.image_input_size / height
        scale_factor_x = InputDataSize.image_input_size / width

        '''retrieve and rescale landmarks in after augmentation'''
        landmark_arr_flat, landmark_arr_x, landmark_arr_y = image_utility.create_landmarks(landmarks=landmark_arr_flat_aug,
                                       scale_factor_x=scale_factor_x,
                                       scale_factor_y=scale_factor_y)

        #print_image_arr(k, resized_img_new, landmark_arr_x, landmark_arr_y)

        '''calculate pose'''
        resized_img_new_cp = np.array(resized_img_new)
        # yaw_predicted, pitch_predicted, roll_predicted = detect.detect(resized_img_new_cp, isFile=False,show=False)
        '''normalize pose -1 -> +1 '''
        # min_degree = -65
        # max_degree = 65
        # yaw_normalized = 2 * ((yaw_predicted - min_degree) / (max_degree - min_degree)) - 1
        # pitch_normalized = 2 * ((pitch_predicted - min_degree) / (max_degree - min_degree)) - 1
        # roll_normalized = 2 * ((roll_predicted - min_degree) / (max_degree - min_degree)) - 1
        # pose_array = np.array([yaw_normalized, pitch_normalized, roll_normalized])
        pose_array = np.array([1, 1, 1])

        '''normalize landmarks based on hyperface method'''
        width = len(resized_img_new[0])
        height = len(resized_img_new[1])
        x_center = width / 2
        y_center = height / 2
        landmark_arr_flat_normalized = []
        for p in range(0, len(landmark_arr_flat), 2):
            landmark_arr_flat_normalized.append((x_center - landmark_arr_flat[p]) / width)
            landmark_arr_flat_normalized.append((y_center - landmark_arr_flat[p + 1]) / height)

        '''test print after augmentation'''
        landmark_arr_flat_n, landmark_arr_x_n, landmark_arr_y_n = image_utility.create_landmarks_from_normalized(
        landmark_arr_flat_normalized, 224, 224, 112, 112)
        #print_image_arr((i*100)+(k+1), resized_img_new, landmark_arr_x_n, landmark_arr_y_n)


        heatmap_landmark = generate_hm(56, 56, landmark_arr_flat_normalized, s=1.0)
        heatmap_landmark_all = np.sum(heatmap_landmark, axis=2)
        #print_image_arr(2*k, heatmap_landmark_all, [], [])
        """save heatmap"""

        file_name_save = png_file_name[i][0:-4] + "_" + str(k) + ".npy"
        hm_f = npy_dir + file_name_save
        # imgpr.print_image_arr_heat(1, hm, print_single=False)

        np.save(hm_f, heatmap_landmark)

        landmark_arr_flat_normalized = np.array(landmark_arr_flat_normalized)


        '''save image'''
        im = Image.fromarray((resized_img_new * 255).astype(np.uint8))
        file_name = IbugConf.before_heatmap_img_path_prefix + png_file_name[i][0:-4] + "_" + str(k)
        im.save(str(file_name) + '.jpg')

        pnt_file = open(str(file_name) + ".pts", "w")
        pre_txt = ["version: 1 \n", "n_points: 14 \n", "{ \n"]
        pnt_file.writelines(pre_txt)
        points_txt = ""
        for l in range(0, len(landmark_arr_flat_normalized), 2):
            points_txt += str(landmark_arr_flat_normalized[l]) + " " + str(landmark_arr_flat_normalized[l + 1]) + "\n"

        pnt_file.writelines(points_txt)
        pnt_file.write("} \n")
        pnt_file.close()

