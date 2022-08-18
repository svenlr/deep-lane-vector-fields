from collections import namedtuple

import torch
# from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from ptflops import get_model_complexity_info

from model.build_model import build_model
from model.lane_anchor_based.lane_detection_clustering import StreetFittingHead


class StreetFittingNetBenchmark(nn.Module):
    def __init__(self, input_shape, num_points=48):
        super(StreetFittingNetBenchmark, self).__init__()
        self.fitting_head = StreetFittingHead(input_shape[2] // 32, input_shape[1] // 32, num_points, [[0, 0], [1, 1]])
        self.approx_points = torch.ones((1, 2, num_points), dtype=torch.float32)
        self.segment_output = torch.ones((1, 128, input_shape[1] // 32, input_shape[2] // 32), dtype=torch.float32)

    def forward(self, x):
        self.segment_output = self.segment_output.expand((x.shape[0], -1, -1, -1))
        self.approx_points = self.approx_points.expand((x.shape[0], -1, -1))
        return self.fitting_head(self.approx_points, self.segment_output)


class Trace(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        x = self.net(x)
        if isinstance(x, dict):
            li = list()
            for k in x.keys():
                li.append(x[k])
            return tuple(li)
        else:
            return x


if __name__ == '__main__':
    args = namedtuple(typename="args", field_names=["s", "num_classes", "network"])
    args.network = "bisenetv2"
    # args.network = "proj_attractor_xy"
    # args.network = "deep_attractor_xy"
    args.tensor_debug = False
    args.multi_img_num = 1
    args.unfreeze_all = False
    args.crop_height = 192
    args.crop_width = 320
    args.street_length = 4.8
    args.street_step_size = 0.1
    args.no_normalize = False
    args.viz_aug_probability = 0
    args.batch_size = 1
    args.num_workers = 1
    args.eval_batch_size = 1
    args.use_gpu = False
    args.cuda = '0'
    args.data = "/home/sven/isfl_generated_train_data/prepared/obstacle_mode_2d_19_ipm_camvid_mono"
    input_shape = (3, args.crop_height, args.crop_width)
    net = build_model(args, only_inference=True)
    # net = SegmentBranch()
    # net = BiSeNetV2(19)
    # net = StreetFittingNetBenchmark(input_shape)
    # net = StreetShapeClassifierHead(40, input_shape[2], input_shape[1])
    # net = ESPNetV2(args(s=2, num_classes=19), 19, dataset="city")
    verbose = False
    macs, params = get_model_complexity_info(net, input_shape,
                                             as_strings=True,
                                             print_per_layer_stat=True,
                                             verbose=verbose)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    example_input = torch.zeros(tuple([1] + list(input_shape)), dtype=torch.float32)
    writer = SummaryWriter()
    writer.add_graph(Trace(net), example_input, verbose=False)
    writer.close()
