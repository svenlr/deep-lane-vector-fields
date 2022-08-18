import argparse
import json
import os

from dataset.io_data_utils import smart_parse_args, init_data_loaders, read_categories_from_dataset, get_coordinate_limits_from_dataset
from eval import inference_test_data
from model.build_model import add_architecture_args, build_model
from nn_utils.train_utils import load_matching_weights


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None, help='path to dataset')
    parser.add_argument('--hyper_params', type=str, default=None, help='path to hyper params')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to weights to load')
    args = parser.parse_args()
    loaded_args = json.load(open(args.hyper_params))
    loaded_args.update(args.__dict__)
    args.__dict__.update(loaded_args)
    args = smart_parse_args(parser, args=args)
    dataloader_train, dataloader_val, dataloader_test = init_data_loaders(args)

    # build model
    model = build_model(args, img_coordinate_limits=get_coordinate_limits_from_dataset(dataloader_train.dataset))
    if args.pretrained_model_path is not None and len(args.pretrained_model_path) > 0:
        load_matching_weights(model, args.pretrained_model_path)
    else:
        print("NO WEIGHTS GIVEN?!")

    categories = read_categories_from_dataset(args)
    inference_test_data(args, model, dataloader_test, os.path.join(args.save_model_path, "test_imgs_final"), categories=categories)


if __name__ == '__main__':
    main()
