import torch
import torch.nn as nn

from model.BiSeNetV2 import SegmentBranch, ConvBNReLU
from model.building_blocks import FullyConnected2LayerHead, ResBlock, ScaledTanh
from model.lane_anchor_based.lane_detection_clustering import create_coordinate_maps
from nn_utils.train_utils import freeze_module


class IterativeStreetMatchingNet(nn.Module):
    def __init__(self, input_shape, coordinate_limits, freeze_segment_branch=True):
        super(IterativeStreetMatchingNet, self).__init__()
        self.segment = SegmentBranch()
        if freeze_segment_branch:
            freeze_module(self.segment)
        self.matching_head = IterativeStreetMatchingHead(input_shape[1] // 32, input_shape[0] // 32, coordinate_limits)

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        points = self.matching_head(feat_s)
        return {"local_map_rl": points}


class IterativeStreetMatchingModule(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.conv = nn.Sequential(
            ResBlock(10),
            FullyConnected2LayerHead(patch_size * patch_size * 10, patch_size * patch_size * 2 + 2 + 2 + 1)
        )
        self.tanh_fit = nn.Tanh()
        self.tanh_next_attend = ScaledTanh(3)
        self.cache_shape = (-1, 2, self.patch_size, self.patch_size)
        self.cache_numel = 2 * self.patch_size * self.patch_size
        self.relu = nn.PReLU()

    def forward(self, feat, cache):
        feat = torch.cat([feat, cache], dim=1)
        feat = self.conv(feat)
        next_cache = self.relu(feat[:, :self.cache_numel].view(self.cache_shape))
        next_patch_diff = self.tanh_next_attend(feat[:, self.cache_numel:self.cache_numel + 2])
        this_patch_fit = self.tanh_fit(feat[:, self.cache_numel + 2:self.cache_numel + 4])
        visibility = feat[: self.cache_numel + 4:self.cache_numel + 5]
        return next_cache, next_patch_diff.to(torch.int32), this_patch_fit, visibility


class IterativeStreetMatchingHead(nn.Module):
    def __init__(self, input_width, input_height, coordinate_limits, num_points=8):
        super().__init__()
        self.num_points = num_points
        self.extract_relevant = ConvBNReLU(128, 32, 1, stride=1, padding=0, bias=True)
        self.res_blocks = nn.Sequential(
            ResBlock(32),
            nn.PixelShuffle(2),
            ResBlock(8),
            # nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=2, stride=2, padding=0, output_padding=0, bias=True),
            # nn.BatchNorm2d(8), nn.PReLU(),
            # ResBlock(32),
            # ResBlock(32),
            # ResBlockDown(32, 32),
            # ResBlock(16),
            # ResBlock(16),
            # ResBlock(16),
            # nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
            # nn.BatchNorm2d(4), nn.PReLU(),
        )
        # trainable local-bias-like parameter for the first iteration
        self.register_parameter("init_cache_bias", torch.nn.Parameter(torch.zeros((1, 4, input_height * 2, input_width * 2))))

        self.init_cache = nn.Sequential(
            ResBlock(8),
            ResBlock(8),
            ConvBNReLU(8, 4, bias=True),
        )

        self.patch_size = 5

        self.select_start = FullyConnected2LayerHead(input_width * 2 * 3 * 8, 2 + 2 * self.patch_size * self.patch_size)

        self.ipm_to_world = FullyConnected2LayerHead(2, 2, 4)

        self.zero_pad = nn.ZeroPad2d(self.patch_size - 1)
        self.tracking_module = IterativeStreetMatchingModule(self.patch_size)
        self.relu = nn.PReLU()

        coordinate_maps = create_coordinate_maps(coordinate_limits, input_height * 2, input_width * 2)
        self.register_buffer("coordinate_maps", coordinate_maps)

    def create_patches(self, feat_padded, patch_center):
        patch_center[:, 0] = patch_center[:, 0].clamp(self.patch_size // 2, feat_padded.shape[2] - self.patch_size // 2 - 1)
        patch_center[:, 1] = patch_center[:, 1].clamp(self.patch_size // 2, feat_padded.shape[3] - self.patch_size // 2 - 1)
        start_positions = patch_center - self.patch_size // 2
        end_positions = patch_center + self.patch_size // 2 + 1
        patches = None
        for i in range(feat_padded.shape[0]):
            patch = feat_padded[i, :, start_positions[i, 0]:end_positions[i, 0], start_positions[i, 1]:end_positions[i, 1]].unsqueeze(0)
            if patches is None:
                patches = patch
            else:
                patches = torch.cat([patches, patch], dim=0)
        return patches

    def forward(self, x):
        feat = self.extract_relevant(x)
        feat = self.res_blocks(feat)
        init_ = self.select_start(feat[:, :, -3:, :])
        patch_center = init_[:, :2].to(torch.int32)
        cache = init_[:, 2:].view(-1, 2, self.patch_size, self.patch_size)
        feat_padded = self.zero_pad(feat)
        patch = self.create_patches(feat_padded, patch_center)
        points = None
        for i in range(self.num_points):
            cache, offset, fit, viz = self.tracking_module(patch, cache)
            world_pos = self.ipm_to_world(patch_center.to(torch.float32)) + fit
            patch_center += offset
            patch = self.create_patches(feat_padded, patch_center)
            points = torch.unsqueeze(world_pos, -1) if points is None else torch.cat([points, torch.unsqueeze(world_pos, -1)], dim=2)
        return points
