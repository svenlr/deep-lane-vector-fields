#!/usr/bin/env python

import argparse
import collections
import os

import torch


def main():
    parser = argparse.ArgumentParser(description="This script can be used to downgrade checkpoints from torch>=1.6 to the torch 1.5 checkpoint format.")
    parser.add_argument('--ckpt', type=str, default=os.getcwd(), help='path to checkpoint or recursive directory with checkpoints')
    args = parser.parse_args()
    convert_file(args.ckpt)


def convert_file(file_name):
    if os.path.isdir(file_name):
        files = os.listdir(file_name)
        for f in files:
            path = os.path.join(file_name, f)
            convert_file(path)
    elif file_name[-3:] == ".pt":
        output_path = file_name.replace(".jit.pt", ".pt").replace(".pt", ".1.5.0.pt")
        if not os.path.exists(output_path) and ".1.5.0.pt" not in file_name:
            state_dict = torch.load(file_name)
            if not isinstance(state_dict, collections.OrderedDict):
                # handling of checkpoints saved with torch.jit.save()
                state_dict = state_dict.state_dict()
                state_dict_copy = collections.OrderedDict()
                for key, value in state_dict.items():
                    state_dict_copy[key] = torch.zeros_like(value)
                    state_dict_copy[key].copy_(state_dict[key])
                state_dict = state_dict_copy
            torch.save(state_dict, output_path, _use_new_zipfile_serialization=False)


if __name__ == '__main__':
    main()
