import time
import argparse

import torch

from model.InferenceWrapper import InferenceWrapper
from model.build_model import build_model, add_architecture_args
from dataset.io_data_utils import smart_parse_args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_ckpt', type=str, default=None, help="Path to the jit model to be tested")
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu for testing')
    parser.add_argument('--input_tensor_shape', type=str, default=None, help="Default: (3, h, w), String value format example: 3x192x320")
    parser.add_argument('--num_samples', type=int, default=50, help="num samples to process for the fps test")
    args = parser.parse_args()

    # segmentation_obstacle.pt: 725 FPS

    model = torch.jit.load(args.jit_ckpt)

    input_shape = tuple(int(i) for i in args.input_tensor_shape.split("x"))
    data = torch.zeros(input_shape, dtype=torch.float32)

    use_cuda = torch.cuda.is_available() and args.use_gpu
    if use_cuda:
        model = model.cuda()
        data = data.cuda()

    warmup = 40
    with torch.jit.optimized_execution(True):
        for i in range(warmup):
            predict = model(data)
        start_time = time.time()
        for i in range(args.num_samples):
            predict = model(data)
        fps = args.num_samples / (time.time() - start_time)
    print("{} FPS: {}".format(args.input_tensor_shape, fps))


if __name__ == '__main__':
    main()
