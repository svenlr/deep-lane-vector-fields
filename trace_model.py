import argparse

import torch
from model.InferenceWrapper import InferenceWrapper, BatchOnlyInferenceWrapper
from model.build_model import build_model, add_architecture_args, create_inference_wrapper
from nn_utils.train_utils import load_matching_weights
from dataset.io_data_utils import smart_parse_args


def main():
    parser = argparse.ArgumentParser()
    add_architecture_args(parser)
    parser.add_argument('--data', type=str, default=None, help="Path to a dataset (required for some lane detection models)")
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, nargs="+", required=True)
    parser.add_argument('--cpu', action="store_true", help="Sometimes works when cuda doesn't. "
                                                           "When loading it later in C++, the model is then moved to GPU either way.")
    parser.add_argument('--input_tensor_shape', type=str, default="192x320", help="Default: (3, h, w), String value format example: 3x192x320")
    parser.add_argument('--no_normalization', action="store_true")
    args = smart_parse_args(parser)

    model = build_model(args, data_parallel=False, only_inference=True)
    if args.input_tensor_shape is None:
        model = create_inference_wrapper(model, args, scriptable=False)
    load_matching_weights(model, args.pretrained_model_path)
    input_shape = tuple(int(i) for i in args.input_tensor_shape.split("x"))
    data = torch.zeros(input_shape, dtype=torch.float32)

    if args.no_normalization:
        model = BatchOnlyInferenceWrapper(model, input_shape[-1], input_shape[-2])
    else:
        model = InferenceWrapper(model, input_shape[-1], input_shape[-2])

    model.eval()
    if not args.cpu:
        model = model.cuda()
        data = data.cuda()
    with torch.jit.optimized_execution(True):
        model = torch.jit.trace(model, data)

    torch.jit.save(model, args.save_path)


if __name__ == '__main__':
    main()
