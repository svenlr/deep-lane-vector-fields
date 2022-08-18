import torch
import torch.nn as nn
import torch.nn.functional

from loss.loss_utils import ohem_loss
from nn_utils.geometry_utils import extract_visibility_per_point, world_to_indices


class MSEAndIndexOrderLoss(nn.Module):
    """
    Loss for lane detection heads that compares point arrays.
    Approximates distance between arrays as MSE error and also penalizes index order when aligning the arrays (i.e. penalizes incorrect ordering of points).
    Tensor dimensions: (N, C, W) = (batch_size, point_dim=|{x,y}|=2, num_points)
    """

    def __init__(self, comparison_up_factor=8, point_order_weight=0.1, ohem_thresh=None, use_visibility_mask=True,
                 prediction_suffix=""):
        super(MSEAndIndexOrderLoss, self).__init__()
        self.comparison_up_factor = comparison_up_factor
        self.point_order_weight = point_order_weight
        self.ohem_thresh = ohem_thresh
        self.use_visibiltiy_mask = use_visibility_mask
        self.prediction_suffix = prediction_suffix

    def forward(self, prediction, label):
        loss = 0
        if "local_map" in label:
            label_map = label["local_map"]
            visibility_mask = label_map.get("visibility_mask", None) if self.use_visibiltiy_mask else None
            if ("local_map_rl" + self.prediction_suffix) in prediction:
                loss += mse_and_index_order_loss(label_map["right_lane"]["left_marking"], prediction["local_map_rl" + self.prediction_suffix], visibility_mask,
                                                 up_factor=self.comparison_up_factor, point_order_weight=self.point_order_weight, ohem_thresh=self.ohem_thresh)
            if ("local_map_rr" + self.prediction_suffix) in prediction:
                loss += mse_and_index_order_loss(label_map["right_lane"]["right_marking"], prediction["local_map_rr" + self.prediction_suffix], visibility_mask,
                                                 up_factor=self.comparison_up_factor, point_order_weight=self.point_order_weight, ohem_thresh=self.ohem_thresh)
        if loss == 0:
            print("warning: ineffective MSEAndIndexOrderLoss: label.keys()={} prediction.keys()={} prediction_suffix={}".format(
                label.keys(), prediction.keys(), self.prediction_suffix))
        return loss


def broadcast_for_pairwise_comparison(array1, array2):
    """
    Given two arrays of data points, build two matrices of same shape by repeating each array by the length of the other array.
    (first dimension is batch size!)
    The first matrix is the first array repeated along the second dim len(array2) times.
    The second matrix is the second array repeated along the first dim len(array1) times.
    If you do vectors = array2 - array1 afterwards, you get a matrix of pairwise vectors that point from each point in array1 to each point in array2.
    For instance, the vector from array1[0] to array2[1] can be accessed as vectors[0, 1]. (prepend 0 for batch dimension)
    :param array1: array of data points
    :type array1: torch.FloatTensor
    :param array2: array of data points
    :type array2: torch.FloatTensor
    :return: comparison matrices
    """
    assert array1.shape[0] == array2.shape[0] or array1.shape[0] == 1 or array2.shape[0] == 1
    target_shape = (max(array1.shape[0], array2.shape[0]), array1.shape[1], array1.shape[2], array2.shape[2])
    array1 = torch.unsqueeze(array1, -1)
    array2 = torch.unsqueeze(array2, -2)
    array2 = array2.expand(target_shape)
    array1 = array1.expand(target_shape)
    return array1, array2


def calc_pairwise_distances(points1, points2):
    """
    :type points1: np.ndarray
    :type points2: np.ndarray
    :return:
    """
    return torch.sqrt(calc_pairwise_sq_error(points1, points2))


def calc_pairwise_sq_error(points1, points2):
    """
    :type points1: np.ndarray
    :type points2: np.ndarray
    :return:
    """
    points1, points2 = broadcast_for_pairwise_comparison(points1, points2)
    vectors = points2 - points1
    distances = (vectors ** 2).sum(dim=1)  # sum along channel dimension (which is x,y[,z])
    return distances


def mse_and_index_order_loss(target_points, predictions, visibility_mask=None, up_factor=8, point_order_weight=0.1, ohem_thresh=None):
    """
    Loss for lane detection heads that compares point arrays.
    Approximates distance between arrays as MSE error and also penalizes index order when aligning the arrays (i.e. penalizes incorrect ordering of points).
    points dimensions: (N, C, W) = (batch_size, point_dim=|{x,y}|=2, num_points)
    visibility_mask:   (N, 1, W) = (batch_size, 1, num_points)
    """
    if len(target_points.shape) == 2:
        target_points = target_points.unsqueeze(0)
    compared_size = target_points.shape[2] * up_factor
    target_points = torch.nn.functional.interpolate(target_points, size=compared_size, mode="linear", align_corners=True)
    predictions = torch.nn.functional.interpolate(predictions, size=compared_size, mode="linear", align_corners=True)
    error_direct = torch.square(target_points - predictions).sum(dim=1)
    if visibility_mask is not None:
        visibility_mask = visibility_mask.unsqueeze(1)
        visibility_mask = torch.nn.functional.interpolate(visibility_mask, size=compared_size, mode="linear", align_corners=True)
        visibility_mask = visibility_mask.squeeze(1)
        visibility_mask[visibility_mask < 0.5] = 0
        error_direct *= visibility_mask
    error_direct = error_direct.mean(dim=1)
    error_matched = 0.5 * _directed_mse_and_index_order_loss(point_order_weight, predictions, target_points, visibility_mask, ohem_thresh=ohem_thresh)
    error_matched += 0.5 * _directed_mse_and_index_order_loss(point_order_weight, target_points, predictions, visibility_mask, ohem_thresh=ohem_thresh)
    if ohem_thresh is not None:
        error_direct = ohem_loss(error_direct, ohem_thresh)
    return torch.cat([error_matched.mean().unsqueeze(-1), error_direct.mean().unsqueeze(-1)], dim=0).min(dim=0)[0].mean()


def _directed_mse_and_index_order_loss(point_order_weight, points1, points2, visibility_mask, ohem_thresh=None):
    pairwise_errors = calc_pairwise_sq_error(points2, points1)
    mse_error, indices = torch.min(pairwise_errors, dim=2)
    neg_index_diffs = -(indices[:, 1:] - indices[:, :-1])
    neg_index_diffs = torch.square(torch.relu(neg_index_diffs.to(torch.float32)))
    if visibility_mask is not None:
        mse_error *= visibility_mask
        neg_index_diffs *= visibility_mask[:, :-1]
    if ohem_thresh is not None:
        mse_error = ohem_loss(mse_error, ohem_thresh)
    return mse_error.mean() + (point_order_weight * neg_index_diffs).mean()


def _visibility_mask_loss(pred_viz_mask, label):
    target_viz_mask = label["local_map"]["visibility_mask"]
    error = torch.square(pred_viz_mask - target_viz_mask)
    error_ohem = error[error > 0.2 ** 2]
    if error_ohem.numel() == 0:
        error_ohem = error.topk(max(error.numel() // 16, 1))
    visibility_loss = torch.mean(error_ohem)
    # visibility should be decreasing
    visibility_gradients = pred_viz_mask[..., 1:] - pred_viz_mask[..., :-1]
    # check if there are any faulty increasing gradients which should be decreasing
    faulty_visibility_gradients_mask = visibility_gradients > 0.01
    if torch.any(faulty_visibility_gradients_mask):
        error_viz_grad = visibility_gradients[faulty_visibility_gradients_mask].mean()
        visibility_loss = visibility_loss + error_viz_grad
    return visibility_loss


def visibility_grid_loss(prediction, label, coordinate_limits, lane="right_lane", marking="left_marking"):
    points = world_to_indices(label["local_map"][lane][marking], coordinate_limits)
    visibility_mask = extract_visibility_per_point(prediction["visibility_grid"], points)
    loss = _visibility_mask_loss(visibility_mask, label)  # type: torch.Tensor
    return loss


if __name__ == '__main__':
    a = torch.FloatTensor([[[0.0, 0], [1, 0], [2, 0], [3, 0]]])
    b = torch.FloatTensor([[[0.0, 0], [1.3, 0], [2.3, 0], [3.0, 0]],
                           [[0.0, 0], [1.1, 0], [2.1, 0], [3.0, 0]]])
    a = a.permute([0, 2, 1])
    b = b.permute([0, 2, 1])
    # b = b.unsqueeze(1)
    print(b.shape)
    print(b.shape)
    # b = b.squeeze(1)
    loss_value = mse_and_index_order_loss(a, b, 8)
    print(loss_value)
