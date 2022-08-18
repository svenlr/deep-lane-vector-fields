import cv2
import numpy as np
import torch


def world_to_ipm(coords, img_width, img_height, pixels_per_world_unit, car_to_image_offset=0.0):
    ret_coords = coords.copy()
    ret_coords[..., 0] = -coords[..., 1] * pixels_per_world_unit + img_width / 2.0
    ret_coords[..., 1] = (-coords[..., 0] + car_to_image_offset) * pixels_per_world_unit + img_height
    return ret_coords


def ipm_to_world(coords, img_width, img_height, pixels_per_world_unit, car_to_image_offset=0):
    ret_coords = coords.copy()
    ret_coords[..., 0] = car_to_image_offset + (img_height - coords[..., 1]) / pixels_per_world_unit
    ret_coords[..., 1] = (img_width / 2 - coords[..., 0]) / pixels_per_world_unit
    return ret_coords


def world_to_ipm_vector(coords, pixels_per_world_unit):
    ret_coords = coords.copy()
    ret_coords[..., 0] = -coords[..., 1] * pixels_per_world_unit
    ret_coords[..., 1] = -coords[..., 0] * pixels_per_world_unit
    return ret_coords


def indices_to_world(tns_points, coordinate_limits):
    index_per_x_meter = 2 / (coordinate_limits[1, 0] - coordinate_limits[0, 0])
    index_per_y_meter = 2 / (coordinate_limits[1, 1] - coordinate_limits[0, 1])
    ret = torch.zeros_like(tns_points)
    ret[:, 0, ...] = coordinate_limits[0, 0] + (1 - tns_points[:, 1, ...]) / index_per_x_meter
    ret[:, 1, ...] = (- tns_points[:, 0, ...]) / index_per_y_meter
    return ret


def world_to_indices(world_points, coordinate_limits):
    index_per_x_meter = 2 / (coordinate_limits[1, 0] - coordinate_limits[0, 0])
    index_per_y_meter = 2 / (coordinate_limits[1, 1] - coordinate_limits[0, 1])
    ret = world_points.clone()
    ret[:, 0, ...] = -world_points[:, 1, ...] * index_per_y_meter
    ret[:, 1, ...] = (-world_points[:, 0, ...] + coordinate_limits[0, 0]) * index_per_x_meter + 1
    return ret


def world_vectors_to_indices(world_vectors, coordinate_limits):
    index_per_x_meter = 2 / (coordinate_limits[1, 0] - coordinate_limits[0, 0])
    index_per_y_meter = 2 / (coordinate_limits[1, 1] - coordinate_limits[0, 1])
    ret = torch.zeros_like(world_vectors)
    ret[:, 0, ...] = -world_vectors[:, 1, ...] * index_per_y_meter
    ret[:, 1, ...] = -world_vectors[:, 0, ...] * index_per_x_meter
    return ret


def transform_to_ipm(points, img_width, img_height, json_data):
    return world_to_ipm(np.array(points, dtype=np.float), img_width, img_height,
                        json_data["transform"]["pixels_per_meter"],
                        json_data["transform"].get("car_to_image_offset", 0))


def indices_to_ipm(indices_points, img_shape):
    ret = torch.zeros_like(indices_points)
    ret[:, 0, ...] = (indices_points[:, 0, ...] + 1) / 2 * img_shape[1]
    ret[:, 1, ...] = (indices_points[:, 1, ...] + 1) / 2 * img_shape[0]
    return ret


def indices_to_ipm_vec(indices_vectors, img_shape):
    ret = torch.zeros_like(indices_vectors)
    ret[:, 0, ...] = (indices_vectors[:, 0, ...]) / 2 * img_shape[1]
    ret[:, 1, ...] = (indices_vectors[:, 1, ...]) / 2 * img_shape[0]
    return ret


def transform_to_world(points, img_width, img_height, json_data):
    return ipm_to_world(np.array(points, dtype=np.float), img_width, img_height,
                        json_data["transform"]["pixels_per_meter"],
                        json_data["transform"].get("car_to_image_offset", 0))


def draw_line(img, points_tensor, color):
    last_p = None
    points_tensor = torch.tensor(points_tensor).cpu().numpy()
    for i in range(points_tensor.shape[2]):
        p = world_to_ipm(points_tensor[0, :, i], img.shape[1], img.shape[0], 50, 0.1)
        if last_p is not None:
            img = cv2.line(img, tuple(last_p), tuple(p), color, thickness=1)
        last_p = p


def normalize_vectors(vectors_tensor, detach_normalization=False):
    inv_lengths = (vectors_tensor ** 2).sum(dim=1, keepdim=True).add_(1e-3).rsqrt_()
    if detach_normalization:
        return vectors_tensor * inv_lengths.detach()
    else:
        return vectors_tensor * inv_lengths


def create_coordinate_maps(coordinate_limits, input_height, input_width):
    x_min = coordinate_limits[0][0]
    y_min = coordinate_limits[0][1]
    x_max = coordinate_limits[1][0]
    y_max = coordinate_limits[1][1]
    corner_points = np.array([[[x_max, y_max], [x_max, y_min]],
                              [[x_min, y_max], [x_min, y_min]]], dtype=np.float32)
    # to (1, 2, H=2, W=2) tensor
    coordinate_maps = torch.from_numpy(corner_points).permute([2, 0, 1]).unsqueeze(0)
    # align corners=True because the corner_points exactly represent the corners of the coordinate maps
    coordinate_maps = torch.nn.functional.interpolate(coordinate_maps, size=(input_height, input_width), align_corners=True, mode="bilinear")
    return coordinate_maps


def get_tangents(lane_data):
    if "tangents" not in lane_data:
        left_tensor = lane_data["left_marking"]
        tangents = left_tensor[..., 2:] - left_tensor[..., :-2]
        tangents *= ((tangents ** 2).sum(dim=-2, keepdim=True) + 1e-3).rsqrt_()
        tangents = torch.cat([tangents[..., :1], tangents, tangents[..., -1:]], dim=-1)
        lane_data["tangents"] = tangents
    return lane_data["tangents"]


def get_normals(lane_data):
    if "normals" not in lane_data:
        tangents = get_tangents(lane_data)
        normals = tangents.flip(1)
        normals[:, 0, :] *= -1
        lane_data["normals"] = normals
    return lane_data["normals"]


def get_sines(lane_data):
    if "sines" not in lane_data:
        tangents = get_tangents(lane_data)
        normals = get_normals(lane_data)
        sines = (normals[..., :-1] * tangents[..., 1:]).sum(dim=1, keepdim=True)  # positive in left curve, negative in right curve
        sines = torch.avg_pool1d(sines, kernel_size=5, stride=1, padding=2)
        sines[torch.abs(sines) < 0.001] = 0
        lane_data["sines"] = torch.cat([sines[:, :, :1], sines], dim=2)
    return lane_data["sines"]


def extract_visibility_per_point(visibility_grid, points):
    points = points.detach()
    sample_at = points.permute(0, 2, 1).view(points.shape[0], points.shape[2], 1, 2)
    visibility = torch.nn.functional.grid_sample(visibility_grid, sample_at, mode="bilinear", padding_mode="zeros", align_corners=False)
    visibility = visibility.squeeze(1).squeeze(-1)
    return visibility


def decode_viz_mask(pred_viz_mask):
    pred_viz_mask_b = (pred_viz_mask > 0.5).to(torch.float32)
    # skip single invisible points
    pred_viz_mask_b = torch.avg_pool1d(pred_viz_mask_b.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
    pred_viz_mask_b[..., 5:] = pred_viz_mask_b[..., 5:].cumprod(dim=1)  # when visibility becomes zero, it stays zero
    pred_viz_mask_b = (pred_viz_mask_b > 0.5)
    return pred_viz_mask_b


def broadcast_for_pairwise_comparison_np(array1, array2):
    """
    Given two numpy arrays of data points, build two matrices of same shape by repeating each array by the length of the other array.
    The first matrix is the first array repeated along the second axis len(array2) times.
    The second matrix is the second array repeated along the first axis len(array1) times.
    If you do vectors = array2 - array1 afterwards, you get a matrix of pairwise vectors that point from each point in array1 to each point in array2.
    For instance, the vector from array1[0] to array2[1] can be accessed as vectors[0, 1].
    :param array1: array of data points
    :param array2: array of data points
    :return: comparison matrices
    """
    assert len(array1) > 0
    assert len(array2) > 0
    target_shape = tuple([array1.shape[0], array2.shape[0]] + list(array1.shape[1:]))
    array1 = np.expand_dims(array1, 1)
    array2 = np.expand_dims(array2, 0)
    array2 = np.broadcast_to(array2, shape=target_shape)
    array1 = np.broadcast_to(array1, shape=target_shape)
    return array1, array2


def calc_straightness_score(points):
    angles = calc_internal_angles(points)
    if len(angles) >= 2:
        angles_diff = angles[1:] - angles[:-1]
        return np.mean(np.abs(angles_diff))
    else:
        return 0


def calc_internal_angles(points):
    tangents = calc_normalized_tangents(points)
    angles = np.arccos((tangents[1:] * tangents[:-1]).sum(axis=1).clip(-1, 1))
    angles = np.concatenate([angles[:1], angles], axis=0)
    return angles


def calc_normalized_tangents(points):
    tangents = points[1:] - points[:-1]
    tangents = tangents / np.sqrt((tangents ** 2).sum(axis=-1, keepdims=True))
    tangents = np.concatenate([tangents, tangents[-1:]], axis=0)
    return tangents


def calc_alignment_indices(points1, points2):
    distances = calc_pairwise_distances(points1, points2)
    alignment1 = np.argmin(distances, axis=1)
    alignment2 = np.argmin(distances, axis=0)
    return alignment1, alignment2


def calc_max_aligned_distance(base_points, points):
    distances = calc_pairwise_distances(points, base_points)
    return np.max(np.min(distances, axis=1))


def calc_pairwise_distances(points1, points2):
    """
    :type points1: np.ndarray
    :type points2: np.ndarray
    :return:
    """
    points1, points2 = broadcast_for_pairwise_comparison_np(points1, points2)
    vectors = points2 - points1
    distances = np.sqrt((vectors ** 2).sum(axis=-1))
    return distances


def calc_similarity(base_sample_points, sample_points, max_distance, max_angle_diff):
    """ similarity of the cluster prototype and the given sample, lower is better """
    max_aligned_dist = calc_max_bidirectional_distance(base_sample_points, sample_points)
    if max_aligned_dist > max_distance:
        return np.inf
    # pw_distances = calc_pairwise_distances(sample_points, base_sample_points)
    # mean_distance = np.mean(np.min(pw_distances, axis=1))
    # return max_aligned_dist
    rel_angles = calc_relative_angles(base_sample_points, sample_points)
    rel_angles = np.abs(rel_angles)
    if np.max(rel_angles) > max_angle_diff:
        return np.inf
    return np.mean(rel_angles)  # mean_distance + np.mean(rel_angles)  # max_aligned_dist  # np.mean(rel_angles)  # mean_distance


def calc_similarities(points_pairs, max_distance, max_angle_diff):
    ret = []
    for i, b, p in points_pairs:
        ret.append((i, calc_similarity(b, p, max_distance, max_angle_diff)))
    return ret


def calc_max_bidirectional_distance(base_points, points):
    distances = calc_pairwise_distances(points, base_points)
    return max(np.max(np.min(distances, axis=1)), np.max(np.min(distances, axis=0)))


def calc_relative_angles(base_points, points):
    """
    For each point in points, calculate the angles relative to the closest points in the base_points.
    :param base_points: the base points that are used as a comparison basis
    :param points: the points to compare
    :return: angles between each point and its closest point from base_points
    """
    distances = calc_pairwise_distances(points, base_points)
    # indices of the closest base point for each point
    closest_indices = np.argmin(distances, axis=1)
    tangents = calc_normalized_tangents(points)
    base_tangents = calc_normalized_tangents(base_points)

    dot_products = (tangents * base_tangents[closest_indices]).sum(axis=1)
    # angles of all points respective to the closest base point for each point
    angles = np.arccos(dot_products.clip(-1, 1))
    return angles


def determine_first_visible_in_limits(points, limits):
    mask = np.logical_and(points >= limits[0], points <= limits[1])
    mask = np.logical_and(mask[:, 0], mask[:, 1])
    if mask[0]:
        return 0
    elif np.any(mask):
        return np.argmax(mask)
    else:
        return None


def determine_last_visible_in_limits(points, limits, first_all_visible_idx):
    mask = np.logical_and(points[first_all_visible_idx:] >= limits[0], points[first_all_visible_idx:] <= limits[1])
    mask = np.logical_and(mask[:, 0], mask[:, 1])
    if np.all(mask):
        return first_all_visible_idx + points.shape[0] - 1
    elif np.any(mask):
        return first_all_visible_idx + np.argmin(mask)
    else:
        return None


def find_first_not_visible_point_idx(points, img):
    found_first_visible = False
    for i, p in enumerate(points):
        if img.shape[1] > p[0] >= 0 and img.shape[0] > p[1] >= 0:
            found_first_visible = True
        elif found_first_visible:
            return i


def street_length(points):
    # type: (np.ndarray) -> float
    if len(points) > 1:
        return float(np.sum(np.sqrt(((points[1:] - points[:-1]) ** 2).sum(axis=1)), axis=0))
    else:
        return 0


def avg_spacing(points):
    # type: (np.ndarray) -> float
    if len(points) > 1:
        return street_length(points) / max(len(points) - 1, 1)
    else:
        return 0.1


def _resample_points(points, step_size=None, new_num_points=None, aligned_data=None):
    vectors = points[1:] - points[:-1]
    lengths = np.sqrt((vectors ** 2).sum(axis=1))
    cum_distances = np.append([0], np.cumsum(lengths))
    if new_num_points is not None:
        steps = np.linspace(0, cum_distances[-1], num=new_num_points)
    elif step_size is not None:
        steps = np.linspace(0, cum_distances[-1], num=int(np.round(cum_distances[-1] / step_size)) + 1)
    else:
        raise RuntimeError("you must specify either step_size or new_num_points")
    ret = np.zeros(shape=(steps.shape[0], points.shape[1]), dtype=points.dtype)
    for i in range(points.shape[1]):  # usually 2 or 3 for either 2D or 3D points
        ret[:, i] = np.interp(steps, cum_distances, points[:, i])
    if aligned_data is not None:
        ret_aligned_data = np.zeros(shape=(steps.shape[0], aligned_data.shape[1]), dtype=points.dtype)
        for j in range(aligned_data.shape[1]):
            ret_aligned_data[:, j] = np.interp(steps, cum_distances, aligned_data[:, j])
        return ret, ret_aligned_data
    else:
        return ret


def make_equidistant(points, step_size=0.1, aligned_data=None):
    return _resample_points(points, step_size=step_size, aligned_data=aligned_data)


def resample_to_size(points, new_num_points, other_arrays=None):
    return _resample_points(points, new_num_points=new_num_points, aligned_data=other_arrays)


def left_orthogonal(vector):
    # type: (np.ndarray) -> np.ndarray
    o = vector.copy()
    o[..., 0], o[..., 1] = -o[..., 1], o[..., 0].copy()
    return o


def right_orthogonal(vector):
    # type: (np.ndarray) -> np.ndarray
    o = vector.copy()
    o[..., 0], o[..., 1] = o[..., 1], -o[..., 0].copy()
    return o


def closest_point_index(points, point):
    return np.argmin(distance_squared_multiple(points, point))


def distance_squared_multiple(points, point):
    # type: (np.ndarray, np.ndarray) -> np.ndarray
    return ((point - points) ** 2).sum(axis=1)


def iterative_make_continuous(points):
    return iteratively_apply_filter(points, make_continuous_filter)


def iteratively_apply_filter(points, filter_function):
    for i in range(len(points)):
        f = filter_function(points)
        if np.any(f):
            points = points[f]
        else:
            break
    return points


def make_continuous_filter(points):
    if len(points) < 3:
        return np.repeat(True, len(points))
    else:
        vectors = points[1:] - points[:-1]
        inner_products = (vectors[1:] * vectors[:-1]).sum(axis=1)
        continuous = inner_products > 0
        return np.append([True], np.append(continuous, [True], axis=0), axis=0)


def vector_of(angle):
    # type: (float) -> np.ndarray
    ret = np.zeros(shape=(2,), dtype=np.float32)
    ret[0], ret[1] = np.cos(angle), np.sin(angle)
    return ret


def extrapolate_or_shrink_points(points, target_num):
    if len(points) == 1:
        return np.array([points[0]] * target_num)
    if len(points) < target_num:
        points = np.array(points)
        extra_points = []
        while len(extra_points) < target_num - len(points):
            extra_points.append(points[-1] + (points[-1] - points[-2]) * (len(extra_points) + 1))
        points = np.concatenate([points, extra_points], axis=0)
    elif len(points) > target_num:
        points = points[:target_num]
    return points


def points_to_tensor(points):
    points = np.array(points, dtype=np.float32)
    return torch.from_numpy(points).permute(1, 0)


if __name__ == '__main__':
    coord_limits = np.array([[0, -4], [4, 4]], dtype=np.float32)
    indices = torch.tensor([[
        [1, 0.5, 1],
        [1, 1, 1],
    ]], dtype=torch.float32)
    print(indices.shape)
    world_coords = indices_to_world(indices, coord_limits)
    print(world_coords)
    indices = world_to_indices(world_coords, coord_limits)
    print(indices)
