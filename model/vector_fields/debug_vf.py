import math

import cv2

from model.building_blocks import *
from nn_utils.geometry_utils import indices_to_world, world_to_ipm, get_tangents, \
    indices_to_ipm, indices_to_ipm_vec


def _integrate_attractor_debug(attractor, extract_correction_fn, pos, momentum, num_steps, only_attractor=False, momentum_weight=0.7, normalize_momentum=True,
                               local_grid=None, per_point_back_prop=False):
    points = []
    points_guess = []
    corrections = []
    magnitude_sq = (momentum ** 2).sum(dim=1, keepdim=True)
    for i in range(num_steps):
        if local_grid is not None:
            grid = pos.view(attractor.shape[0], 1, 1, 2) + local_grid
        else:
            # Default: this is not really a grid, but just a single interpolated lookup in the attractor
            grid = pos.view(attractor.shape[0], 1, 1, 2)
        # lookup index corrections for the approximate position in the attractor tensor
        # align_corners=False is important, so that each cell in the tensor covers a receptive field of the same size
        attractor_lookup = torch.nn.functional.grid_sample(attractor, grid, align_corners=False, padding_mode="border", mode="bilinear")
        correction = extract_correction_fn(attractor_lookup, momentum)
        points_guess.append(pos)
        corrections.append(correction)
        pos = pos + correction
        points.append(pos)
        if i >= 1:
            new_momentum = momentum * momentum_weight + correction * (1 - momentum_weight)
            if normalize_momentum:
                # normalize integration speed so that we use approximately equidistant steps
                momentum = new_momentum * torch.sqrt(magnitude_sq / (new_momentum ** 2).sum(dim=1, keepdim=True))
        if not only_attractor:
            pos = pos + momentum
        if per_point_back_prop:
            pos = pos.detach()
    points = torch.stack(points, dim=2)
    points_guess = torch.stack(points_guess, dim=2)
    corrections = torch.stack(corrections, dim=2)
    return points, points_guess, corrections


def _draw_vector_field(img, vector_field, color, normalize=False, step=15):
    x_field = cv2.resize(vector_field[0, ...], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    y_field = cv2.resize(vector_field[1, ...], (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    for y in range(0, img.shape[0], step):
        for x in range(0, img.shape[1], step):
            v = [x_field[y, x], y_field[y, x]]
            if normalize:
                v_len = math.sqrt(v[0] ** 2 + v[1] ** 2) / step / 0.5
                v[0] /= v_len
                v[1] /= v_len
            else:
                v[0] *= img.shape[1] / 2
                v[1] *= img.shape[0] / 2
            t = int(x + v[0]), int(y + v[1])
            cv2.arrowedLine(img, (x, y), t, color, thickness=1, tipLength=0.1)


def _attractor_example():
    import argparse
    import numpy as np
    from dataset.io_data_utils import smart_parse_args, init_data_loaders, get_coordinate_limits_from_dataset
    from nn_utils.lane_metrics import lane_mse
    from loss.vector_field_loss import indirect_xy_attractor_integration_loss, gt_to_index_xy_attractor
    import random
    random.seed(24)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='path of training data')
    args = smart_parse_args(parser)
    args.no_normalize = True
    dataloader_train, dataloader_val, dataloader_test = init_data_loaders(args, shuffle=False)
    sample = dataloader_train.dataset[2]
    transform = sample["local_map"]["transform"]

    coordinate_limits = get_coordinate_limits_from_dataset(dataloader_train.dataset)
    gt_points = sample["local_map"]["right_lane"]["left_marking"].unsqueeze(0)
    gt_viz_mask = sample["local_map"]["visibility_mask"] = torch.from_numpy(sample["local_map"]["visibility_mask"]).unsqueeze(0)
    sample["local_map"]["right_lane"]["left_marking"] = gt_points
    sample["local_map"]["step_size"] = torch.tensor(sample["local_map"]["step_size"])
    tangents = get_tangents(sample["local_map"]["right_lane"])
    attractor = gt_to_index_xy_attractor(gt_points, coordinate_limits, (12, 20))
    attractor = (torch.rand_like(attractor) - 0.5) * 0.1
    attractor = torch.zeros_like(attractor)
    # attractor = torch.cat([attractor] * 8, dim=1)
    attractor = nn.Parameter(attractor)

    optimizer = torch.optim.SGD([attractor], lr=0.5, weight_decay=0)
    for i in range(200):
        pred = {"lane_attractor": torch.tanh(attractor), "local_map_rl": gt_points, "main_flow": attractor}
        # loss = indirect_biased_xy_attractor_integration_loss(pred, sample, coordinate_limits, bias_directions)
        loss = indirect_xy_attractor_integration_loss(pred, sample, coordinate_limits, iterations=1, ohem_thresh=0.02 ** 2, min_forward_step_length=0.05,
                                                      max_forward_step_length=0.2, only_train_representable=True)
        # loss = indirect_attractor_step_loss(pred, sample, coordinate_limits)
        # loss += indirect_xy_direction_loss(pred, sample, coordinate_limits)
        # loss = indirect_main_flow_loss(pred, sample, coordinate_limits, lookahead_meters=0.8)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # attractor = torch.tanh(attractor) * 0.2
    attractor = attractor.detach().cpu()

    # print(attractor.shape)
    index_per_x_meter = 2.0 / (coordinate_limits[1, 0] - coordinate_limits[0, 0])
    initial_pos = torch.tensor([[0.0, 0.95]], dtype=torch.float32)
    initial_momentum = torch.tensor([0.0, -index_per_x_meter * 0.2], dtype=torch.float32).view(-1, 2)
    initial_momentum = torch.tensor([0.0, -0.03], dtype=torch.float32).view(-1, 2)

    def extract_correction(attractor_lookup, curr_momentum):
        # remove spatial dimensions, which are 1-sized because of the 1x1 grid_sample() lookup
        return attractor_lookup.mean(dim=[-2, -1])

    points, points_n, corrections = _integrate_attractor_debug(attractor, extract_correction, initial_pos, initial_momentum, 60, momentum_weight=0.7)
    # print(points)
    # print("gt", gt_points)
    # print("gt int", indices_to_world(points, self.coordinate_limits))
    pred_points = indices_to_world(points, coordinate_limits)
    # pred_points = _gen_random_noisy_samples_around_gt(sample["local_map"]["right_lane"], max_noise_inner_curve=0.2)[0]
    img = cv2.cvtColor(sample["img"][0].cpu().numpy(), cv2.COLOR_GRAY2BGR)
    up_scaling = 5
    img = cv2.resize(img, dsize=None, fx=up_scaling, fy=up_scaling)
    # _, pred_points, gt_points = indirect_attractor_step_loss(pred, sample, coordinate_limits)
    ipm_points = indices_to_ipm(points, img.shape)
    ipm_points_n = indices_to_ipm(points_n, img.shape)
    ipm_corrections = indices_to_ipm_vec(corrections, img.shape)
    ipm_points_gt = world_to_ipm(gt_points.squeeze().permute(1, 0).numpy(), img.shape[1], img.shape[0],
                                 pixels_per_world_unit=transform["pixels_per_meter"] * up_scaling, car_to_image_offset=transform["car_to_image_offset"])
    attractor = attractor[0, :, ...].cpu().numpy()
    attractor_viz = cv2.resize(attractor[0, ...] + 0.5, sample["img"].shape[-2:][::-1], interpolation=cv2.INTER_NEAREST)
    _draw_vector_field(img, attractor, (0, 0, 100), step=up_scaling * 5, normalize=False)
    attractor_viz = cv2.cvtColor(attractor_viz, cv2.COLOR_GRAY2BGR)
    # print(ipm_points.shape)
    # print(np.sqrt(((ipm_points[1:] - ipm_points[:-1]) ** 2).sum(axis=-1)))
    ipm_points = ipm_points.squeeze().permute(1, 0).numpy().astype(np.int)
    ipm_points_n = ipm_points_n.squeeze().permute(1, 0).numpy().astype(np.int)
    ipm_points_gt = ipm_points_gt.astype(np.int)
    # uncomment to visualize how gen_random_noisy_samples_around_gt works
    # for r in range(100):
    #     random_points = _gen_random_noisy_samples_around_gt(sample["local_map"]["right_lane"], sample["local_map"]["step_size"],
    #                                                         max_noise_inner_curve=0.2)[0]
    #     random_points = world_to_indices(random_points, coordinate_limits)
    #     random_points = indices_to_ipm(random_points, img.shape).squeeze().permute(1, 0).numpy().astype(np.int)
    #     for i in range(len(ipm_points_gt) - 1):
    #         cv2.line(img, tuple(random_points[i]), tuple(random_points[i]), (255, 255, 0), thickness=6, lineType=cv2.LINE_4)
    for i in range(len(ipm_points_gt) - 1):
        cv2.line(img, tuple(ipm_points_gt[i]), tuple(ipm_points_gt[i]), (255, 0, 0), thickness=6, lineType=cv2.LINE_4)
    for i in range(len(ipm_points) - 1):
        cv2.line(img, tuple(ipm_points[i]), tuple(ipm_points[i + 1]), (0, 255, 0), thickness=2, lineType=cv2.LINE_4)
        cv2.line(attractor_viz, tuple(ipm_points[i]), tuple(ipm_points[i]), (255, 0, 255), thickness=2, lineType=cv2.LINE_4)
        cv2.arrowedLine(img, tuple(ipm_points_n[i]), tuple(ipm_points[i]), (255, 0, 255), thickness=2)
        cv2.arrowedLine(img, tuple(ipm_points[i]), tuple(ipm_points_n[i + 1]), (255, 255, 255), thickness=2)
    # cv2.imshow("attractor", cv2.resize(attractor_viz, dsize=None, fx=3, fy=3))
    cv2.imshow("test", img[:, :, ...])
    # cv2.imshow("test", img)
    while cv2.waitKey(0) != 27:
        continue

    print("Model's MSE on image: ", lane_mse(pred_points, gt_points, (sample["local_map"]["visibility_mask"])))


if __name__ == '__main__':
    _attractor_example()
