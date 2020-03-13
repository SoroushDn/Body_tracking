import json
import numpy as np

def import_point(point_file):
    with open(point_file) as fr:
        points_info = json.load(fr)
    points_info['points'] = np.array(points_info['points'])

    return point_file

if __name__ == "__main__":
    datapoint_file = "/home/soroush/PycharmProjects/Bodytracking/dataloader/LSP/lsp_dataset_original/heatmap/10004.pts"
    p_info = import_point(datapoint_file)
    print(p_info)