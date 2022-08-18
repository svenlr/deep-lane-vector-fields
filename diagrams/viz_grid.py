import cv2
import numpy as np
import torch

from loss.vector_field_loss import create_index_maps

if __name__ == '__main__':
    vis_grid = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524.png_visibility_grid.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    vis_grid = torch.from_numpy(vis_grid).unsqueeze(0).unsqueeze(0)
    maps = create_index_maps((192, 320), vis_grid.device).permute(0, 2, 3, 1)
    vis_grid = torch.nn.functional.grid_sample(vis_grid, maps, mode="bilinear", padding_mode="zeros", align_corners=False).squeeze(0).squeeze(0).numpy()
    cv2.imwrite("/home/sven/Downloads/vector_fields/vis_grid.png", vis_grid)
