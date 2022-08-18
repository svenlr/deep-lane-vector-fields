import torch
import torch.nn as nn
import torch.nn.functional

from loss.lane_mse_loss import calc_pairwise_sq_error
from nn_utils.geometry_utils import decode_viz_mask


def lane_mse(pred_points, gt_points, gt_visibility_mask, res_factor=8):
    pred_points_i = nn.functional.interpolate(pred_points, size=res_factor * pred_points.shape[2], mode="linear", align_corners=True)
    gt_points = gt_points[..., 1:]
    gt_visibility_mask = gt_visibility_mask[..., 1:]
    pairwise_errors = calc_pairwise_sq_error(gt_points, pred_points_i)
    mse_error, indices = torch.min(pairwise_errors, dim=2)
    mse_error = mse_error * gt_visibility_mask  # zero out invisible points
    num_pred = gt_visibility_mask.sum(dim=1)
    mse_error = mse_error.sum(dim=1) / torch.max(num_pred, torch.ones_like(num_pred))
    return mse_error


def lane_precision(pred_points, pred_viz_mask, target, res_factor=8, lane="right_lane", marking="left_marking", threshold=0.08):
    all_gt_points = nn.functional.interpolate(target["local_map"][lane][marking + "_all"], scale_factor=res_factor, mode="linear", align_corners=True)
    errors_sq = calc_pairwise_sq_error(pred_points, all_gt_points)
    pred_viz_mask_b = decode_viz_mask(pred_viz_mask)
    errors_sq = errors_sq.min(dim=2)[0]
    errors_sq[pred_viz_mask < 0.5] = 1e6
    threshold_sq = threshold ** 2
    num_match = (errors_sq < threshold_sq).to(torch.float32).sum(dim=1)
    num_pred = pred_viz_mask_b.to(torch.float32).sum(dim=1)
    precision = num_match / torch.max(num_pred, torch.ones_like(num_pred))
    precision[num_pred == 0] = 1  # nothing predicted -> all predicted correct -> precision = 1
    return precision


def lane_recall(pred_points, pred_viz_mask, target, res_factor=8, lane="right_lane", marking="left_marking", threshold=0.08):
    threshold_sq = threshold ** 2
    gt_points = target["local_map"][lane][marking]
    gt_viz_mask = target["local_map"]["visibility_mask"]
    gt_points = gt_points[..., 1:]
    gt_viz_mask = gt_viz_mask[..., 1:]
    pred_points_i = nn.functional.interpolate(pred_points, size=res_factor * pred_points.shape[2], mode="linear", align_corners=True)
    pred_viz_mask = decode_viz_mask(pred_viz_mask).to(torch.float32)
    pred_mask_i = nn.functional.interpolate(pred_viz_mask.unsqueeze(1), size=res_factor * pred_points.shape[2], mode="nearest").squeeze(1)
    pairwise_errors = calc_pairwise_sq_error(gt_points, pred_points_i)
    # before min(): ignore columns corresponding to non-visible predicted points by setting "infinity" as distance
    pairwise_errors[pred_mask_i.unsqueeze(1).expand_as(pairwise_errors) < 0.5] = 1e6
    se, _ = torch.min(pairwise_errors, dim=2)
    # do not count invisible gt points, either
    se[gt_viz_mask < 0.5] = 1e6
    num_match = (se < threshold_sq).to(torch.float32).sum(dim=1)
    num_gt = gt_viz_mask.sum(dim=1)
    return num_match / torch.max(num_gt, torch.ones_like(num_gt))


def lane_f1(pred_points, pred_viz_mask, target, res_factor=8, lane="right_lane", marking="left_marking", threshold=0.08):
    recall = lane_recall(pred_points, pred_viz_mask, target, res_factor=res_factor, lane=lane, marking=marking, threshold=threshold)
    precision = lane_precision(pred_points, pred_viz_mask, target, res_factor=res_factor, lane=lane, marking=marking, threshold=threshold)
    return 2 * precision * recall / (precision + recall + 1e-5), precision, recall
