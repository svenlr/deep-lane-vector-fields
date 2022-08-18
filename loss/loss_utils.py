import numpy as np
import torch

from nn_utils.local_map_utils import make_coordinate_image


def ohem_loss(error_tensor, threshold):
    # type: (torch.Tensor, float) -> torch.Tensor
    if error_tensor.numel() == 0:
        return torch.tensor([0], dtype=error_tensor.dtype, device=error_tensor.device)
    error_tensor = error_tensor.flatten()
    ohem_mask = error_tensor > threshold
    if torch.any(ohem_mask):
        error_tensor = error_tensor[ohem_mask]
    else:
        error_tensor = error_tensor.topk(error_tensor.numel() // 16)[0]
    return error_tensor.mean()


def distance_decaying_loss_weights(start_decay_dist=0.2, img=None, img_width=None, img_height=None, v_decay=1.0, h_decay=1.0, min_weight=0.1,
                                   mode="euclid", end_decay_dist=0.8, car_pos=None, limits=None):
    """
    weights for loss in x,y dimensions

    :param start_decay_dist: when to start decay, by default in [0, 1]. If limits are given, according to limits.
    :param img: the image used to determine the size of the loss masks
    :param img_width: instead of specifying img, you can use img_width and img_height
    :param img_height: instead of specifying img, you can use img_width and img_height
    :param v_decay: multiplier to control the shape of the ellipse by slowing or increasing loss decay in vertical direction
    :param h_decay: multiplier to control the shape of the ellipse by slowing or increasing loss decay in horizontal direction
    :param min_weight: minimum weight (no weight will be smaller than this). Default 0.1
    :param mode: decay mode. use "euclid" for linear decay. "square" for (first slow than fast) or "sqrt" for (first fast than slow)
    :param end_decay_dist: where to stop the decay transition
    :param car_pos: center of the decay ellipse. default: bottom center of image (usually car position). Same coordinate system as limits, if given, else px.
    :param limits: meter coordinate range limits for the image. [[bottom_left0, bottom_left1], [top_right0, top_right1]]
    :return: the mask to multiply to the weights
    """
    if img is not None:
        img_width = img.shape[1]
        img_height = img.shape[0]
    if limits is None:
        limits = np.array([[1, 1], [0, 0]])
        coordinates = np.indices((img_height, img_width), dtype=np.float32)
        coordinates = np.swapaxes(coordinates, 0, 2)
        coordinates = np.swapaxes(coordinates, 0, 1)
        coordinates[..., 0] /= img_height
        coordinates[..., 1] /= img_width
    else:
        limits = np.array(limits)
        coordinates = None
    if car_pos is None:
        limits_width = abs(limits[1][0] - limits[0][0])
        limits_height = abs(limits[1][1] - limits[0][1])
        car_pos = np.array([limits_height, limits_width / 2.0])
    else:
        car_pos[0] /= img_width
        car_pos[1] /= img_height
    if coordinates is None:
        coordinates = make_coordinate_image(img_width, img_height, limits)
    vectors = coordinates - car_pos
    vectors[:, :, 0] *= v_decay
    vectors[:, :, 1] *= h_decay
    start_decay_dist = start_decay_dist
    if mode == "square":
        distances = (vectors ** 2).sum(axis=2)
        start_decay_dist = start_decay_dist ** 2
        end_decay_dist = end_decay_dist ** 2
    elif mode == "sqrt":
        distances = np.sqrt(np.sqrt((vectors ** 2).sum(axis=2)))
        start_decay_dist = np.sqrt(start_decay_dist)
        end_decay_dist = np.sqrt(end_decay_dist)
    elif mode == "l1":
        distances = np.abs(vectors).sum(axis=2)
        start_decay_dist = start_decay_dist
    else:
        distances = np.sqrt((vectors ** 2).sum(axis=2))
    offset_distances = np.clip(distances - start_decay_dist, 0, np.max(distances))
    if end_decay_dist == start_decay_dist:
        weights = np.ones(shape=img.shape[:2], dtype=img.dtype)
    else:
        weights = np.clip(1 - offset_distances / abs(end_decay_dist - start_decay_dist), 0, 1)
    if min_weight > 0:
        weights = weights * (1 - min_weight) + min_weight
    return weights
