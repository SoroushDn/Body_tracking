import scipy.io
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

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


"""
# WFLW
file_dataset = open('/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt')
os.mkdir('/imagesAnnotaion/')

for line in file_dataset:

    print(line)
    adata = line.split(" ")

    a1 = adata[0:196:2]
    a2 = adata[1:196:2]
    a11 = [float(i) for i in a1]
    a12 = [float(i) for i in a2]
    a = np.zeros([98, 2])
    a[:, 0] = np.array(a11).T
    a[:, 1] = np.array(a12).T
    np.savetxt("/home/soroush/PycharmProjects/FacialLandmark/data_temp.pts", a, fmt = "%s")
    file_name_path = adata[-1]
    #print(file_name)
    # file_name = file_name.split("--")
    file_name = file_name_path.split("/")[1]
    file_path = file_name_path.split("/")[0]
    file_name_path_pts = file_name_path[0:-5]
    print(file_path)
    #file_name = file_name.split("/")[-1]
    file_train = open('//data_temp.pts', "r+")
    if not os.path.isdir('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotaion/' + file_path + '/'):
        os.mkdir('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotaion/' + file_path + '/')
    cnt = 0
    while os.path.exists('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotaion/' + file_name_path_pts + '.pts'):
        cnt += 1
        file_name_path_pts = file_name_path_pts.split("#")[0]
        file_name_path_pts = file_name_path_pts + "#" + str(cnt)

    final_train = open('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotaion/' + file_name_path_pts + '.pts', "w+")
    bxymin = a1[196:198]
    bxymax = a1[198:200]
    final_train.write('version: 1' + '\n' +'n_points:' + str(np.shape(a)[0]) + '\n' + '{' + '\n' + file_train.read() + '}' + '\n' + str(adata[196]) + ' ' + str(adata[197]) + '\n' + str(adata[198]) + ' ' + str(adata[199]))

    final_train.close()
    os.remove('//data_temp.pts')


print(cnt)

"""

# COFW

image = cv2.imread('/home/soroush/myGray.png')
file_dataset = open('/home/soroush/annotation.txt')
# file_dataset = open('/home/soroush/PycharmProjects/FacialLandmark/COFW/annotation.txt')
#os.mkdir('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotationCOFW/')
img_num = 0
for line in file_dataset:
    img_num += 1
    print(line)
    adata = line.split("\t")

    a1 = adata[0:29]
    a2 = adata[29:58]
    a11 = [float(i) for i in a1]
    a12 = [float(i) for i in a2]
    print_image_arr(1, image, a11, a12)
    a = np.zeros([29, 2])
    a[:, 0] = np.array(a11).T
    a[:, 1] = np.array(a12).T
    np.savetxt("/home/soroush/PycharmProjects/FacialLandmark/data_temp.pts", a, fmt = "%s")
    #file_name_path = adata[-1]
    #print(file_name)
    # file_name = file_name.split("--")
    file_name = str(img_num) + ".pts"
    file_path = "/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotationCOFW/"

    print(file_path)
    #file_name = file_name.split("/")[-1]
    file_train = open('/home/soroush/PycharmProjects/FacialLandmark/data_temp.pts', "r")

    # cnt = 0
    # while os.path.exists('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotationCOFW/' + file_name):
    #     cnt += 1
    #     file_name_new = file_name[0:-4].split("#")[0]
    #     file_name = file_name_new + "#" + str(cnt)

    final_train = open('/home/soroush/PycharmProjects/FacialLandmark/imagesAnnotationCOFW/' + file_name, "w+")
    final_train.write('version: 1' + '\n' +'n_points:' + str(np.shape(a)[0]) + '\n' + '{' + '\n' + file_train.read() + '}')

    final_train.close()
    os.remove('/home/soroush/PycharmProjects/FacialLandmark/data_temp.pts')


#print(cnt)