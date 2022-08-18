import torch.nn.functional

from model.BiSeNetV2 import ConvBNReLU, SegmentBranch, DetailBranch, BGALayer, SegmentHead
from model.building_blocks import *
from nn_utils.geometry_utils import indices_to_world, extract_visibility_per_point
from nn_utils.train_utils import freeze_module


class ProjectedAttractorNet(nn.Module):
    def __init__(self, coordinate_limits, freeze_segment_branch=True, num_iter=32, context_embedding=False, vf_out_activation=nn.Tanh,
                 direction_field_prediction=True, direction_field_projection=True, vf_integration=True, num_seg_classes=None):
        super(ProjectedAttractorNet, self).__init__()
        assert (not direction_field_projection or direction_field_prediction)  # projection implies prediction
        self.num_iter = num_iter
        self.direction_field_prediction = direction_field_prediction
        self.direction_field_projection = direction_field_projection
        self.vf_integration = vf_integration
        self.include_seg = num_seg_classes is not None
        self.context_embedding = context_embedding
        self.segment = SegmentBranch()
        if freeze_segment_branch:
            freeze_module(self.segment)
        if self.include_seg:
            self.detail = DetailBranch()
            self.bga = BGALayer()
            self.head = SegmentHead(128, 1024, num_seg_classes, up_factor=8, aux=False)
        self.register_buffer("coordinate_limits", torch.tensor(coordinate_limits))
        self.register_buffer("initial_pos", torch.tensor([0, 0.95], dtype=torch.float32).view(-1, 2))
        self.register_buffer("initial_momentum", torch.tensor([0, -0.05 * 48 / num_iter], dtype=torch.float32).view(-1, 2))
        self.extract_relevant = ConvBNReLU(128, 32, 1, stride=1, bias=True, padding=0)
        self.res_blocks = nn.Sequential(
            ResBlock(32),
            ResBlock(32),
            ResBlock(32),
        )
        self.down = nn.Sequential(
            ResBlockDown(32, 32),
            ConvBNReLU(32, 16, ks=1, padding=0, bias=True),
            ResBlock(16),
        )
        self.up = nn.Sequential(
            nn.PixelShuffle(2),
        )
        if self.direction_field_prediction:
            self.predict_main_flow = nn.Sequential(
                ConvBN(16, 2, bias=True),
                vf_out_activation(),
            )
        self.predict_attractor = nn.Sequential(
            ConvBN(8, 2, bias=True),
            vf_out_activation(),
        )
        self.predict_visibility = nn.Sequential(
            ConvBN(16, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        if self.context_embedding:
            feat = self.res_blocks(self.extract_relevant(feat_s))
        else:
            feat = self.res_blocks(self.extract_relevant(feat5_4))
        feat_down = self.down(feat)
        feat_up = self.up(feat)
        afn_field = self.predict_attractor(feat_up)
        visibility_grid = self.predict_visibility(feat_down)
        ret = {
            "lane_attractor": afn_field,
            "visibility_grid": visibility_grid,
        }
        if self.include_seg:
            seg_detail = self.detail(x)
            seg_feat_head = self.bga(seg_detail, feat_s)
            ret["seg"] = self.head(seg_feat_head)
        if self.direction_field_prediction:
            dir_field = self.predict_main_flow(feat_down)
            dir_field = torch.nn.functional.interpolate(dir_field, size=(afn_field.shape[-2], afn_field.shape[-1]), mode="bilinear", align_corners=False)
            dir_field = dir_field * ((dir_field ** 2).sum(dim=1, keepdim=True) + 1e-3).rsqrt_()
            ret["main_flow"] = dir_field
            if self.direction_field_projection:
                # extract part of affinity field that goes against direction field
                attractor_against_main_flow = torch.relu((afn_field * -dir_field).sum(dim=1, keepdim=True))
                afn_field = afn_field + attractor_against_main_flow * dir_field
                ret["lane_attractor"] = afn_field
        if self.vf_integration and not self.training:
            initial_pos = self.initial_pos.expand(afn_field.shape[0], 2)
            points = integrate_xy_attractor(afn_field, initial_pos, self.initial_momentum, self.num_iter, use_momentum_first_time=False)
            ret["local_map_rl"] = indices_to_world(points, self.coordinate_limits)
            ret["visibility_mask"] = extract_visibility_per_point(visibility_grid, points)
        return ret


def attract_lateral_initial_pos(attractor, initial_pos, num_steps, lateral_direction=None):
    if lateral_direction is None:
        lateral_direction = torch.tensor([1, 0], dtype=initial_pos.dtype, device=initial_pos.device).view(1, 2).expand(attractor.shape[0], 2)
    pos = initial_pos.clone()
    for i in range(num_steps):
        grid = pos.view(attractor.shape[0], 1, 1, 2)
        attractor_lookup = torch.nn.functional.grid_sample(attractor, grid, align_corners=False, padding_mode="border", mode="bilinear")
        attractor_lookup = attractor_lookup.squeeze(-1).squeeze(-1)
        projected_correction = (lateral_direction * attractor_lookup).sum(dim=1, keepdim=True)
        pos = pos + projected_correction * lateral_direction
    return pos


def integrate_xy_attractor(attractor, pos, momentum, num_steps, momentum_weight=0.7, normalize_momentum=True,
                           local_grid=None, detached_back_prop=False, use_momentum_first_time=True):
    points = []
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
        correction = attractor_lookup.mean(dim=[-2, -1])
        pos = pos + correction
        points.append(pos)
        if i >= 1 or use_momentum_first_time:
            new_momentum = momentum * momentum_weight + correction * (1 - momentum_weight)
            if normalize_momentum:
                # normalize integration speed so that we use approximately equidistant steps
                momentum = new_momentum * torch.sqrt(magnitude_sq / (new_momentum ** 2).sum(dim=1, keepdim=True))
        pos = pos + momentum
        if detached_back_prop:
            pos = pos.detach()
    points = torch.stack(points, dim=2)
    return points
