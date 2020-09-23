import os
import numpy as np
from scipy.spatial import distance


def _generate_distance_matrix(xy_arr):
	x_arr = xy_arr[[slice(None, None, 2) for _ in range(xy_arr.ndim)]]
	y_arr = xy_arr[[slice(1, None, 2) for _ in range(xy_arr.ndim)]]

	d_matrix = np.zeros(shape=[len(x_arr), len(y_arr)])
	for i in range(0, x_arr.shape[0], 1):
		for j in range(i + 1, x_arr.shape[0], 1):
			p1 = [x_arr[i], y_arr[i]]
			p2 = [x_arr[j], y_arr[j]]
			d_matrix[i, j] = distance.euclidean(p1, p2)
			d_matrix[j, i] = distance.euclidean(p1, p2)
	return d_matrix


dataset_dir = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/'
npy_dir = '/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/'

for file_ in os.listdir(dataset_dir):
	if file_.endswith(".pts"):
		points_arr = []
		points_x_arr = []
		points_y_arr = []
		with open(dataset_dir + file_) as fp:
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
		d_matrix = _generate_distance_matrix(np.array(points_arr))
		file_name_save = file_[0:-4] + "_dis_mat" + ".npy"
		d_f = npy_dir + file_name_save
		np.save(d_f, d_matrix)







 
