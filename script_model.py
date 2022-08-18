import argparse

import torch
from model.InferenceWrapper import ScriptTupleInferenceWrapper, ScriptInferenceWrapper
from model.build_model import build_model, add_architecture_args
from nn_utils.train_utils import load_matching_weights
from dataset.io_data_utils import smart_parse_args


def main():
    parser = argparse.ArgumentParser()
    add_architecture_args(parser)
    parser.add_argument('--data', type=str, default=None, help="Path to a dataset (required for some lane detection models)")
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--pretrained_model_path', type=str, nargs="+", required=True)
    parser.add_argument('--cpu', action="store_true", help="Sometimes works when cuda doesn't. "
                                                           "when loading it later in C++, the model is then moved to GPU either way.")
    args = smart_parse_args(parser)

    model = build_model(args, data_parallel=False, scriptable=True)
    if args.network in ["bisenetv2+lane"]:
        model = ScriptTupleInferenceWrapper(model, args.model_width, args.model_height)
    else:
        model = ScriptInferenceWrapper(model, args.model_width, args.model_height)
    load_matching_weights(model, args.pretrained_model_path)

    model.eval()
    if not args.cpu:
        model = model.cuda()
    with torch.jit.optimized_execution(True):
        model = torch.jit.script(model)
    torch.jit.save(model, args.save_path)


if __name__ == '__main__':
    main()
