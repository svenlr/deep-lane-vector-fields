import numpy as np

from nn_utils.geometry_utils import extrapolate_or_shrink_points, points_to_tensor
from dataset.data_sample_utils import dict_to_list


def extract_local_map_json_from_predict_dict(predict_dict, sample_dict, batch_idx, stats=None):
    predict_dict["local_map"] = {
        "right_lane": {"left_marking": predict_dict["local_map_rl"]},
    }
    predict_dict["gt_local_map"] = sample_dict["local_map"]
    if "local_map_rr" in predict_dict:
        predict_dict["local_map"]["right_lane"]["right_marking"] = predict_dict["local_map_rr"]
    if "visibility_mask" in predict_dict:
        predict_dict["local_map"]["visibility_mask"] = predict_dict["visibility_mask"][batch_idx]
    predict_dict = dict_to_list(predict_dict["local_map"])
    predict_dict["gt"] = {"right_lane": {}, "left_lane": {}}
    if stats is not None:
        predict_dict["metrics"] = stats.get(sample_dict["identifier"][batch_idx])
    for key in ["left_marking", "right_marking"]:
        if key in predict_dict["right_lane"]:
            predict_dict["right_lane"][key] = np.swapaxes(predict_dict["right_lane"][key][batch_idx], -1, -2).tolist()
        if key in sample_dict["local_map"]["right_lane"]:
            predict_dict["gt"]["right_lane"][key] = np.swapaxes(sample_dict["local_map"]["right_lane"][key][batch_idx].cpu().numpy(), -1, -2).tolist()
    if "visibility_mask" in sample_dict["local_map"]:
        predict_dict["gt"]["visibility_mask"] = sample_dict["local_map"]["visibility_mask"][batch_idx].cpu().numpy().tolist()
    predict_dict["transform"] = {
        "pixels_per_meter": float(sample_dict["local_map"]["transform"]["pixels_per_meter"].cpu().numpy().tolist()[0]),
        "car_to_image_offset": float(sample_dict["local_map"]["transform"]["car_to_image_offset"].cpu().numpy().tolist()[0]),
    }
    # print(predict_dict)
    return predict_dict


def cluster_prototypes_to_same_length(cluster_prototypes, target_num=0):
    if target_num == 0:
        target_num = 0
        for cluster_prototype in cluster_prototypes:
            target_num = max(target_num, len(cluster_prototype))
    return [extrapolate_or_shrink_points(p, target_num) for p in cluster_prototypes]


def make_coordinate_image(img_width, img_height, limits):
    top_row = np.linspace(limits[1], [limits[1][0], limits[0][1]], num=img_width, dtype=np.float32)
    bottom_row = np.linspace([limits[0][0], limits[1][1]], limits[0], num=img_width, dtype=np.float32)
    coordinates = np.linspace(top_row, bottom_row, num=img_height, dtype=np.float32)
    return coordinates


if __name__ == '__main__':
    print(points_to_tensor([[1, 0], [2, 0], [3, 0]]))
