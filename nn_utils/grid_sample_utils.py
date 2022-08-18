import torch
import torch.nn
import torch.nn.functional


def make_local_grid(flow, size, index_span=None):
    if index_span is None:
        index_span = size
    half_span = index_span / 2
    h, w = flow.shape[-2:]
    grid = torch.tensor([[
        [[-half_span / w, -half_span / h], [+half_span / w, -half_span / h]],
        [[-half_span / w, +half_span / h], [+half_span / w, +half_span / h]]
    ]], dtype=torch.float32, device=flow.device) * 2
    grid = grid.permute([0, 3, 1, 2])
    # align corners because we want the spans to be the corner points
    grid = torch.nn.functional.interpolate(grid, size=(size, size), align_corners=True, mode="bilinear")
    grid = grid.permute([0, 2, 3, 1])
    return grid


def create_index_maps(target_shape, target_device):
    maps = torch.tensor([[
        [[-1, -1], [+1, -1]],
        [[-1, +1], [+1, +1]]
    ]], dtype=torch.float32, device=target_device)
    maps = maps.permute([0, 3, 1, 2])
    # align corners because we want the index limits to be the corner points
    maps = torch.nn.functional.interpolate(maps, size=target_shape[-2:], align_corners=True, mode="bilinear")
    return maps


def create_index_grid(target_shape, target_device):
    maps = create_index_maps(target_shape, target_device)
    maps = maps.permute(0, 2, 3, 1)
    return maps.expand(target_shape[0], *target_shape[-2:], 2)
