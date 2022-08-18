import numpy as np
import torch
import torch.nn.functional
from torch import nn

from loss.loss_utils import ohem_loss
from model.vector_fields.lane_detection_vf import integrate_xy_attractor
from nn_utils.geometry_utils import world_to_indices, get_normals, indices_to_world, get_sines, create_coordinate_maps
from nn_utils.grid_sample_utils import create_index_grid, create_index_maps


def indirect_main_flow_loss(pred, target, coordinate_limits, lookahead_meters=0.8):
    main_flow = pred["main_flow"]
    gt_points = target["local_map"]["right_lane"]["left_marking"]
    samples, _ = _gen_random_noisy_samples_around_gt(target["local_map"]["right_lane"], target["local_map"]["step_size"].mean(),
                                                     max_noise_outer_curve=0.3, max_noise_inner_curve=0.2)
    gt_points_indices = world_to_indices(gt_points, coordinate_limits)
    samples_indices = world_to_indices(samples, coordinate_limits)
    lookahead = int(np.round(lookahead_meters / target["local_map"]["step_size"].mean().cpu().numpy()))
    tangents_indices = gt_points_indices[..., lookahead:] - samples_indices[..., :-lookahead]
    tangents_indices *= ((tangents_indices ** 2).sum(dim=1, keepdim=True) + 1e-3).rsqrt_()
    tangents_indices = torch.cat([tangents_indices, tangents_indices[..., -1:].repeat_interleave(lookahead, dim=2)], dim=2)
    similarities = []
    for i in range(gt_points.shape[2]):
        sample = samples_indices[:, :, i].unsqueeze(1).unsqueeze(1)
        sampled_flow = torch.nn.functional.grid_sample(main_flow, sample, align_corners=False, padding_mode="border", mode="bilinear")
        sampled_flow = sampled_flow.mean(dim=[2, 3])
        similarities.append((tangents_indices[..., i] * sampled_flow).sum(dim=1))
    similarities = torch.stack(similarities, dim=1)[target["local_map"]["visibility_mask"] > 0]
    return 1 - similarities.mean()


def indirect_xy_attractor_integration_loss(pred, target, coordinate_limits, max_noise_tan=1.0, max_noise_pos=0.4, ohem_thresh=0.02 ** 2,
                                           res_factor=8, iterations=1, normalize_momentum=False, max_forward_step_length=0.1,
                                           min_forward_step_length=0.0, only_train_representable=False):
    lane_data = target["local_map"]["right_lane"]
    gt_points = target["local_map"]["right_lane"]["left_marking"]
    target_visibility = target["local_map"]["visibility_mask"]
    attractor = pred["lane_attractor"]
    gt_step_size = target["local_map"]["step_size"].mean()
    min_forward_step = int(min_forward_step_length / gt_step_size * res_factor)
    max_forward_step = int(max_forward_step_length / gt_step_size * res_factor)
    samples_momentum, _ = _gen_random_noisy_samples_around_gt(lane_data, max_noise_tan * gt_step_size)
    samples_pos, _ = _gen_random_noisy_samples_around_gt(lane_data, max_noise_pos)
    samples = world_to_indices(samples_pos, coordinate_limits)
    if only_train_representable:
        # because of align_corners during affinity field integration, lane shape in top border side borders of the image can not be fully represented
        # in order to avoid degradation during training, we ignore those positions during training
        representable_y = (attractor.shape[-2] - 1) / attractor.shape[-2]
        representable_x = (attractor.shape[-1] - 1) / attractor.shape[-1]
        representable_mask = torch.abs(samples[:, 0, :]) < representable_x
        representable_mask = torch.logical_and(samples[:, 1, :] > -representable_y, representable_mask)
    else:
        representable_mask = torch.ones_like(target_visibility > 0)
    gt_points_indices = world_to_indices(gt_points, coordinate_limits)
    # vectors that point from gt_point to the next randomized gt point
    momentum_vectors = world_to_indices(samples_momentum, coordinate_limits)[..., 1:] - gt_points_indices[..., :-1]
    # always starting with "up" vector
    momentum_vectors[:, 0, 0], momentum_vectors[:, 1, 0] = 0.0, -0.08
    attracted_points = []
    offset = iterations + max_forward_step // res_factor + 1
    for i in range(1, samples.shape[-1] - offset):
        attracted_point = integrate_xy_attractor(attractor, samples[..., i], momentum_vectors[..., i - 1], iterations,
                                                 normalize_momentum=normalize_momentum).squeeze(-1)
        attracted_points.append(attracted_point if attracted_point.ndim == 2 else attracted_point[..., -1])
    attracted_points = torch.stack(attracted_points, dim=2)
    attracted_points = indices_to_world(attracted_points, coordinate_limits)
    num_gt = 1 + res_factor * (gt_points.shape[2] - 1)
    gt_points_interp = nn.functional.interpolate(gt_points, size=num_gt, mode="linear", align_corners=True)
    errors = []
    start_compare_offset = iterations * res_factor + min_forward_step
    end_compare_offset = iterations * res_factor + max_forward_step
    for i in range(start_compare_offset, end_compare_offset):
        # We want the attraction to point forwards along driving direction or at least 90 degrees, but most certainly not backwards.
        # Compare attracted points to their corresponding ground truth and the following points, therein take the minimum.
        end_compare_idx = i + attracted_points.shape[-1] * res_factor
        errors.append(((attracted_points - gt_points_interp[..., i:end_compare_idx:res_factor]) ** 2).sum(dim=1))
    error_tensor = torch.stack(errors, dim=2)
    error_tensor = error_tensor.min(dim=2)[0]
    representable_mask = representable_mask[..., 1:attracted_points.shape[-1] + 1]
    error_mask = torch.logical_and(target_visibility[..., 1:attracted_points.shape[-1] + 1] > 0, representable_mask)
    error_tensor = error_tensor[error_mask]
    # initial_pos = torch.zeros_like(samples[..., 0:1], requires_grad=False)
    # initial_pos[:, 0, 0] = 0  # torch.normal(-0.1, 0.05, size=(initial_pos.shape[0],), dtype=initial_pos.dtype, device=initial_pos.device)
    # initial_pos[:, 1, 0] = 0.95
    # first_point = integrate_xy_attractor(attractor, initial_pos, momentum_vectors[..., 0], 1, normalize_momentum=False).squeeze(-1)
    # first_pred_point = indices_to_world(first_point, coordinate_limits).unsqueeze(-1)
    # first_point_error = ((first_pred_point - gt_points_interp[..., 0:res_factor * 5]) ** 2).sum(dim=1, keepdim=True).min(dim=2)[0]
    return ohem_loss(error_tensor, ohem_thresh)  # + first_point_error.mean() / gt_points.shape[2]


def direct_sq_attractor_training_loss(pred, target, coordinate_limits):
    gt_points = target["local_map"]["right_lane"]["left_marking"]
    basin = pred["lane_attractor"]
    lengths_sq = gt_to_world_sq_distances(gt_points, coordinate_limits, basin.shape)
    close_to_lane_mask = lengths_sq < 0.4 ** 2
    return ((basin - lengths_sq) ** 2)[close_to_lane_mask].mean()


def direct_xy_attractor_training_loss(pred, target, coordinate_limits, mean=True, ohem_thresh=0.05 ** 2, scaling=8):
    with torch.no_grad():
        target_shape = list(pred["lane_attractor"].shape)
        target_shape[2] *= scaling
        target_shape[3] *= scaling
        target_shape = tuple(target_shape)
        gt_attractor = gt_to_index_xy_attractor(target["local_map"]["right_lane"]["left_marking_all"], coordinate_limits, target_shape)
        distances_sq = gt_to_world_sq_distances(target["local_map"]["right_lane"]["left_marking_all"], coordinate_limits, target_shape)
        close_to_lane_mask = distances_sq < 0.3 ** 2
        all_indexes = create_index_grid(target_shape, pred["lane_attractor"].device)
    pred_attractor_scaled = torch.nn.functional.grid_sample(pred["lane_attractor"], all_indexes, mode="bilinear", align_corners=False)
    errors = (pred_attractor_scaled - gt_attractor) ** 2
    error_tensor = errors[close_to_lane_mask.expand_as(errors)]
    if mean:
        error_tensor = ohem_loss(error_tensor, ohem_thresh)
        error_tensor = error_tensor.mean()  # .sum(dim=[1, 2, 3]).mean()
    return error_tensor


def _gen_random_noisy_samples_around_gt(lane_data, gt_step_size, max_noise_outer_curve=0.4, max_noise_inner_curve=None, min_noise=None,
                                        points_key="left_marking", min_radius_to_expect=0.4):
    if max_noise_inner_curve is None:
        max_noise_inner_curve = min(0.15, max_noise_outer_curve)
    max_sine = gt_step_size / min_radius_to_expect
    gt_samples = lane_data[points_key]
    sines = get_sines(lane_data)
    sines = torch.max_pool1d(sines, 9, stride=1, padding=4)
    sines = torch.avg_pool1d(sines, 5, stride=1, padding=2)
    normals = get_normals(lane_data)

    # random_scales = torch.normal(0, max_noise_outer_curve ** 2, (gt_samples.shape[0], 1, gt_samples.shape[-1]), dtype=torch.float32, device=gt_samples.device)
    random_scales = torch.rand((gt_samples.shape[0], 1, gt_samples.shape[-1]), dtype=torch.float32, device=gt_samples.device) * 2 - 1
    random_scales *= max_noise_outer_curve
    # sines and noise go in different directions (product negative) => then apply outer curve noise
    # random_scales[sines != 0] *= (max_noise_outer_curve + max_noise_inner_curve) / (2 * max_noise_outer_curve)
    opposite_noise_mask = (sines * random_scales > 0)
    random_scales[opposite_noise_mask] -= ((sines[opposite_noise_mask] / max_sine).clip(-1, 1) * (max_noise_outer_curve - max_noise_inner_curve))
    if min_noise is not None:
        random_scales *= (1 - min_noise / max_noise_outer_curve)
        sign_scales = torch.sign(random_scales)
        sign_scales[sign_scales == 0] = 1
        random_scales += sign_scales * min_noise
    samples = gt_samples + normals * random_scales
    return samples.to(torch.float32), random_scales


def _gt_to_attractor_vectors(gt_points, coordinate_limits, target_shape):
    gt_points_i = world_to_indices(gt_points, coordinate_limits)
    coord_maps = create_index_maps(target_shape, gt_points_i.device)
    gt_points_v = gt_points_i.view(gt_points_i.shape[0], 2, 1, 1, gt_points_i.shape[-1])
    return gt_points_v - coord_maps.unsqueeze(-1).to(gt_points_v.device)


def gt_to_index_xy_attractor(gt_points, coordinate_limits, target_shape):
    vectors = _gt_to_attractor_vectors(gt_points, coordinate_limits, target_shape)
    corresponding_indices = (vectors ** 2).sum(dim=1, keepdim=True).min(dim=-1, keepdim=True)[1]
    corresponding_indices = corresponding_indices.expand((vectors.shape[0], 2, vectors.shape[2], vectors.shape[3], 1))
    return vectors.gather(-1, corresponding_indices).squeeze(-1)


def gt_to_world_sq_distances(gt_points, coordinate_limits, target_shape):
    coord_maps = create_coordinate_maps(coordinate_limits, target_shape[-2], target_shape[-1])
    gt_points_v = gt_points.view(gt_points.shape[0], 2, 1, 1, gt_points.shape[-1])
    vectors = coord_maps.unsqueeze(-1).to(gt_points_v.device) - gt_points_v
    distances_sq = (vectors ** 2).sum(dim=1, keepdim=True).min(dim=-1, keepdim=False)[0]
    return distances_sq
