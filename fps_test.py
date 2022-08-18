import time
import argparse

import torch

from model.InferenceWrapper import InferenceWrapper
from model.build_model import build_model, add_architecture_args
from dataset.io_data_utils import smart_parse_args


def main():
    parser = argparse.ArgumentParser()
    add_architecture_args(parser)
    parser.add_argument('--data', type=str, default=None, help="Path to a dataset (required for some lane detection models)")
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu for testing')
    parser.add_argument('--num_samples', type=int, default=100, help="num samples to process for the fps test")
    parser.add_argument('--script', action="store_true", help="Use torch.jit.script() instead of torch.jit.trace()")
    args = smart_parse_args(parser)

    model = build_model(args, data_parallel=False, scriptable=args.script, inference_wrapper=True)
    model = model.cpu()
    data = torch.zeros((3, args.crop_height, args.crop_width), dtype=torch.float32)

    use_cuda = torch.cuda.is_available() and args.use_gpu
    model.eval()
    if args.script:
        model = torch.jit.script(model)
    else:
        model = torch.jit.trace(model, data)
    if use_cuda:
        model = model.cuda()
        data = data.cuda()

    model.eval()
    warmup = 40
    start_time = None
    with torch.jit.optimized_execution(True):
        for i in range(warmup):
            predict = model(data)
        start_time = time.time()
        for i in range(args.num_samples + warmup):
            predict = model(data)
        fps = args.num_samples / (time.time() - start_time)
    print("{} {}x{} FPS: {}".format(args.network, args.crop_width, args.crop_height, fps))


if __name__ == '__main__':
    main()
