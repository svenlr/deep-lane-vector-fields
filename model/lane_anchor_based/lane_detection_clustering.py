import math
import os

import cv2
import numpy as np
from torch.nn.modules.flatten import Flatten

from dataset.data_sample_utils import tensor_to_np_uint8
from model.BiSeNetV2 import *
from model.building_blocks import *
from model.building_blocks import GlobalAvgPool2d
from nn_utils.geometry_utils import draw_line, create_coordinate_maps, extract_visibility_per_point, world_to_indices
from nn_utils.grid_sample_utils import *
from nn_utils.train_utils import freeze_module


class StreetFittingHead(nn.Module):
    def __init__(self, input_width, input_height, coordinate_limits, heatmap_sharpness_eps_sq: float = 6., tensor_debug=False):
        super(StreetFittingHead, self).__init__()
        self.tensor_debug = tensor_debug
        self.input_width = input_width
        self.input_height = input_height
        working_height = (input_height + input_height % 2) * 4 - 4
        working_width = (input_width + input_width % 2) * 4 - 4
        coordinate_maps = create_coordinate_maps(coordinate_limits, working_height, working_width)
        self.register_buffer("coordinate_maps", coordinate_maps)

        chn_part1 = 32
        chn_part2 = 1 + chn_part1
        move_chn1 = 32
        move_chn2_cut1 = 24
        move_chn3_cut2 = 16
        move_chn4_down = 32
        move_chn_final = 16
        self.calculate_move = nn.Sequential(
            # ResBlock(chn_part2, out_channels=move_chn1, reduction="cut_bottom"),
            # ResBlock(move_chn1, out_channels=move_chn2_cut1, reduction="cut_bottom"),
            ResBlock(chn_part2, out_channels=move_chn2_cut1, res_projection="cut_bottom"),
            ResBlock(move_chn2_cut1, out_channels=move_chn3_cut2, res_projection="cut_bottom"),
            ResBlock(move_chn3_cut2),
            # ResBlock(move_chn4_down, out_channels=move_chn_final, reduction="cut_bottom"),
            ResBlock(move_chn_final),
            ConvBNReLU(move_chn_final, move_chn_final, ks=3, bias=True)
        )
        self.x_move = nn.Sequential(
            nn.ConvTranspose2d(in_channels=move_chn_final, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self.y_move = nn.Sequential(
            nn.ConvTranspose2d(in_channels=move_chn_final, out_channels=4, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.Tanh(),
        )
        self.heatmap_eps_sq: float = heatmap_sharpness_eps_sq

    def forward(self, approx_points, feat):
        heatmaps = points_to_heatmaps(approx_points, self.coordinate_maps, eps_sq=self.heatmap_eps_sq)
        if self.tensor_debug:
            for i in range(heatmaps.shape[1]):
                cv2.imwrite(os.path.join("tensor_debug", "heatmap{:02d}.png".format(i)), tensor_to_np_uint8(heatmaps[0, i, ...]))
        heatmaps_small = torch.nn.functional.interpolate(heatmaps, size=(self.input_height, self.input_width), mode="bilinear", align_corners=False)
        heatmap_combined = heatmaps_small.max(dim=1, keepdim=True)[0]
        if self.tensor_debug:
            cv2.imwrite(os.path.join("tensor_debug", "heatmaps_combined.png"), tensor_to_np_uint8(heatmap_combined[0, 0, ...]))
        feat = torch.cat([heatmap_combined, feat], dim=1)
        feat = self.calculate_move(feat)
        x_move = self.x_move(feat)
        y_move = self.y_move(feat)
        if self.tensor_debug:
            cv2.imwrite(os.path.join("tensor_debug", "x_move.png"), tensor_to_np_uint8(x_move[0, 0, ...]))
            cv2.imwrite(os.path.join("tensor_debug", "y_move.png"), tensor_to_np_uint8(y_move[0, 0, ...]))
        move = torch.cat([x_move, y_move], dim=1)
        move_coords = (move.unsqueeze(2) * heatmaps.unsqueeze(1)).sum(dim=[-2, -1])
        if self.tensor_debug:
            cv2.imwrite(os.path.join("tensor_debug", "moves.png"), tensor_to_np_uint8(move_coords[0, ...]))
        if self.tensor_debug:
            img = np.zeros(shape=(192, 320, 3), dtype=np.uint8)
            draw_line(img, approx_points + move_coords, (0, 255, 0))
            cv2.imwrite(os.path.join("tensor_debug", "result.png"), img)
        return approx_points + move_coords


class FittingHeadGridSample(nn.Module):
    def __init__(self, coordinate_limits):
        super().__init__()
        self.register_buffer("coordinate_limits", torch.tensor(coordinate_limits))
        self.reduce_channels = nn.Sequential(
            ConvBNReLU(128, 32, 1, stride=1, bias=True, padding=0),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
        )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
        )
        self.predict_attractor = nn.Sequential(
            ConvBN(8, 2, bias=True),
            nn.Tanh(),
        )

    def forward(self, approx_points, feat):
        attractor = self.predict_attractor(self.up(self.res_blocks(self.reduce_channels(feat))))
        approx_points_i = world_to_indices(approx_points, self.coordinate_limits).permute([0, 2, 1]).unsqueeze(-2)
        corrections = torch.nn.functional.grid_sample(attractor, approx_points_i, align_corners=False, padding_mode="border", mode="bilinear")
        points = approx_points + corrections.squeeze(-1)
        return points


class StreetShapeClassifierHead(nn.Module):
    def __init__(self, num_clusters, input_width, input_height, fully_conv=True):
        assert input_width < 20 and input_height < 20  # would get too complex otherwise
        super(StreetShapeClassifierHead, self).__init__()
        if fully_conv:
            self.classifier = nn.Sequential(
                ResBlock(32),
                ConvBNReLU(32, num_clusters, ks=3, padding=0, bias=True),
                GlobalAvgPool2d(),
            )
        else:
            w = math.ceil(input_width / 2)
            h = math.ceil(input_height / 2)
            self.classifier = nn.Sequential(
                ResBlockDown(32, 32),
                Flatten(start_dim=1),
                FullyConnected2LayerHead(w * h * 32, num_clusters),
            )

    def forward(self, feat):
        cluster_weights = self.classifier(feat)
        cluster_weights = torch.softmax(cluster_weights, dim=1)  # alternative to the weighted summation as below
        # cluster_weights = torch.sigmoid(cluster_weights)
        # cluster_weights = cluster_weights / (cluster_weights.sum(dim=-1, keepdim=True) + 1e-5)
        return cluster_weights


class SegToLaneDetectionFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce_channels = nn.Sequential(
            ConvBNReLU(128, 64, 1, stride=1, bias=True, padding=0),
            ConvBNReLU(64, 32, 1, stride=1, bias=True, padding=0),
        )
        self.res_blocks = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlockDown(32, 64),
            ConvBNReLU(64, 48, 1, padding=0, bias=True),
            ResBlock(48),
            ConvBNReLU(48, 32, 1, padding=0, bias=True),
            ResBlock(32),
        )

    def forward(self, seg_features):
        return self.res_blocks(self.reduce_channels(seg_features))


class StreetShapeClassifierNet(nn.Module):
    def __init__(self, input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=True):
        super(StreetShapeClassifierNet, self).__init__()
        self.segment = SegmentBranch()
        if freeze_segment_branch:
            freeze_module(self.segment)
        self.seg_to_lane = SegToLaneDetectionFeatures()
        self.shape_classifier_head = StreetShapeClassifierHead(len(cluster_prototypes), input_shape[1] // 32, input_shape[0] // 32)
        self.register_buffer("coordinate_limits", torch.tensor(coordinate_limits))
        self.register_buffer("clustering", clustering_to_tensor(cluster_prototypes))
        self.predict_visibility = nn.Sequential(
            ResBlock(32),
            ConvBNReLU(32, 1, bias=True),
        )

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_lane = self.seg_to_lane(feat_s)
        cluster_weights = self.shape_classifier_head(feat_lane)
        # convert to (bs, 1, 1, num_clusters), then multiply
        # (bs, 1, 1, num_clusters) * (1, 2, num_points, num_clusters)
        weighted_cluster_shapes = cluster_weights.unsqueeze(1).unsqueeze(1) * self.clustering
        # sum weighted clustering to obtain prediction
        street_points = weighted_cluster_shapes.sum(-1)
        visibility_grid = self.predict_visibility(feat_lane)
        return {"local_map_rl": street_points,
                "visibility_mask": extract_visibility_per_point(visibility_grid, world_to_indices(street_points, self.coordinate_limits)),
                "visibility_grid": visibility_grid,
                }


class StreetFittingNet(nn.Module):
    def __init__(self, input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=True, freeze_shape_classifier=True, fully_conv=True,
                 tensor_debug=False, smooth_fitting_head=True):
        super().__init__()
        self.tensor_debug = tensor_debug
        self.segment = SegmentBranch()
        self.seg_to_lane = SegToLaneDetectionFeatures()
        self.shape_classifier_head = StreetShapeClassifierHead(len(cluster_prototypes), input_shape[1] // 64, input_shape[0] // 64,
                                                               fully_conv=fully_conv)
        self.smooth_fitting_head = smooth_fitting_head
        if smooth_fitting_head:
            self.fitting_head = StreetFittingHead(input_shape[1] // 64, input_shape[0] // 64, coordinate_limits, tensor_debug=tensor_debug)
        else:
            self.fitting_head = FittingHeadGridSample(coordinate_limits)
        self.predict_visibility = nn.Sequential(
            ResBlock(32),
            ConvBNReLU(32, 1, bias=True),
        )

        self.register_buffer("coordinate_limits", torch.tensor(coordinate_limits))
        self.register_buffer("clustering", clustering_to_tensor(cluster_prototypes))

        if freeze_shape_classifier:
            freeze_module(self.shape_classifier_head)
        if freeze_segment_branch:
            freeze_module(self.segment)

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat_seg = self.segment(x)
        feat_lane = self.seg_to_lane(feat_seg)
        cluster_weights = self.shape_classifier_head(feat_lane)
        # convert to (bs, 1, 1, num_clusters), then multiply
        # (bs, 1, 1, num_clusters) * (1, 2, num_points, num_clusters)
        weighted_cluster_shapes = cluster_weights.unsqueeze(1).unsqueeze(1) * self.clustering
        # sum weighted clustering to obtain prediction
        approx_street_points = weighted_cluster_shapes.sum(-1)
        if self.smooth_fitting_head:
            street_points = self.fitting_head(approx_street_points, feat_lane)
        else:
            street_points = self.fitting_head(approx_street_points, feat_seg)
        visibility_grid = self.predict_visibility(feat_lane)
        if self.tensor_debug:
            img = np.ones(shape=(192, 320, 3), dtype=np.uint8) * 255
            # img[...] = np.array((inverse_normalize(x, (0.406, 0.456, 0.485), (0.229, 0.224, 0.225)).cpu()
            #                      .squeeze().numpy().swapaxes(0, 1).swapaxes(1, 2) * 255).clip(0, 255).astype(np.uint8), dtype=np.uint8)
            for i in range(cluster_weights.shape[-1]):
                # draw_line(img, self.clustering[:, :, :, i],
                #           (int(255 * cluster_weights[0, i]), int(100 * cluster_weights[0, i]), int(50 * (1 - cluster_weights[0, i]))))
                c = int(255 * (1 - cluster_weights[0, i]) ** 2)
                draw_line(img, self.clustering[:, :, :, i], (c, c, c))
            draw_line(img, approx_street_points, (255, 100, 0))
            # draw_line(img, street_points, (100, 255, 100))
            cv2.imwrite(os.path.join("tensor_debug", "result.png"), img)
            grid = create_index_maps((192, 320), visibility_grid.device).permute(0, 2, 3, 1)
            abc = torch.nn.functional.grid_sample(visibility_grid, grid, align_corners=False, padding_mode="zeros", mode="bilinear")
            cv2.imwrite(os.path.join("tensor_debug", "visibility_grid.png"), (abc[0, 0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        return {"local_map_rl": street_points,
                "local_map_rl_cluster": approx_street_points,
                "visibility_mask": extract_visibility_per_point(visibility_grid, world_to_indices(street_points, self.coordinate_limits)),
                "visibility_grid": visibility_grid,
                }


class BiseNetV2WithLaneScript(StreetFittingNet):
    def __init__(self, input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=True, freeze_shape_classifier=True, segmentation_classes=19,
                 tensor_debug=False, input_extra_height=0, fully_conv=True):
        super().__init__(input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=freeze_segment_branch,
                         freeze_shape_classifier=freeze_shape_classifier, tensor_debug=tensor_debug, fully_conv=fully_conv)
        self.detail = DetailBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, segmentation_classes, up_factor=8, aux=False)
        self.output_cut_off = -input_extra_height if input_extra_height > 0 else None
        self.output_cut_off_32 = -input_extra_height // 32 if input_extra_height > 0 else None

    def bisenet_part2(self, feat_d, feat_s):
        feat_head = self.bga(feat_d, feat_s[..., :self.output_cut_off_32, :])
        logits = self.head(feat_head)
        return logits

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_l = self.seg_to_lane(feat_s)
        cluster_weights = self.shape_classifier_head(feat_l)
        feat_d = self.detail(x[..., :self.output_cut_off, :])
        # convert to (bs, 1, 1, num_clusters), then multiply
        # (bs, 1, 1, num_clusters) * (1, 2, num_points, num_clusters)
        weighted_cluster_shapes = cluster_weights.unsqueeze(1).unsqueeze(1) * self.clustering
        # sum weighted clustering to obtain prediction
        approx_street_points = weighted_cluster_shapes.sum(-1)
        street_points = self.fitting_head(approx_street_points, feat_l)
        return self.bisenet_part2(feat_d, feat_s), street_points


class BiseNetV2WithLane(BiseNetV2WithLaneScript):
    def __init__(self, input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=True, freeze_shape_classifier=True, segmentation_classes=19,
                 tensor_debug=False):
        super().__init__(input_shape, cluster_prototypes, coordinate_limits, freeze_segment_branch=freeze_segment_branch,
                         freeze_shape_classifier=freeze_shape_classifier, segmentation_classes=segmentation_classes, tensor_debug=tensor_debug)

    def forward(self, x):
        ret = super(BiseNetV2WithLane, self).forward(x)
        return {"seg": ret[0], "local_map_rl": ret[1]}


def clustering_to_tensor(cluster_prototypes):
    """
    Convert an array of point arrays which represent clusters for common street shapes to a pytorch tensor.
    Input dims: (num_clusters, num_points, 2)
    Output tensor: (1, 2, num_points, num_clusters)
    """
    cluster_prototypes = torch.tensor(cluster_prototypes, dtype=torch.float32)
    cluster_prototypes = cluster_prototypes.permute(2, 1, 0)
    return cluster_prototypes.unsqueeze(0)  # add batch dimension


def points_to_heatmaps(points_tensor, coordinates_tensor, eps_sq: float = 6.):
    """
    points_tensor: (N, 2, num_points)
    coordinates_tensor: (1, 2, H, W)
    """
    # points_tensor -> (N, 2, num_points, H, W)
    points_tensor = points_tensor.unsqueeze(-1).unsqueeze(-1)
    # Add 3rd dimension that will be expanded to num_points: coordinates (1, 2, H, W) -> (1, 2, 1, H, W)
    coordinates_tensor = coordinates_tensor.unsqueeze(2)
    # (N, 2, num_points, H, W) --- sum ---> (N, num_points, H, W)
    sq_distances = torch.square(coordinates_tensor - points_tensor).sum(dim=1)  # sum squares along coordinates dimension
    # gaussian activation to create heatmap from distances
    heatmaps = torch.exp(-eps_sq * sq_distances)
    return heatmaps / heatmaps.sum(dim=[-2, -1], keepdim=True)


def _main():
    import numpy as np
    import cv2

    points = torch.from_numpy(np.array([[[0.0, 1.0, 2.0], [0.5, 1, 1.5]]]))
    heatmaps = points_to_heatmaps(points, create_coordinate_maps([[0, -3], [4, 3]], 6, 10), eps_sq=3).squeeze().numpy()
    viz = np.concatenate(heatmaps, axis=1)
    cv2.imshow("test", viz / viz.max())
    cv2.waitKey(0)


if __name__ == '__main__':
    _main()
