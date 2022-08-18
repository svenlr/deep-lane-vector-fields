import argparse
import json
import os
import shutil
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from model.build_model import build_model, add_architecture_args
from nn_utils.train_utils import load_matching_weights
from dataset.io_data_utils import write_categories, smart_parse_args, make_paths_absolute, init_data_loaders, read_categories_from_dataset
from eval import write_attractor_output, AsyncImgWriter


def predict_on_image(model, args, color_map=True):
    # pre-processing on image
    image = cv2.imread(args.inference_input, -1)
    writer = AsyncImgWriter()
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (args.crop_width, args.crop_height), interpolation=cv2.INTER_LINEAR)
    image = Image.fromarray(image).convert('RGB')
    image = transforms.ToTensor()(image)
    image = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(image).unsqueeze(0)
    # predict
    model.eval()
    predict = model(image)
    write_attractor_output(writer, os.path.dirname(args.inference_output), predict, os.path.basename(args.inference_input), 0)
    predict_seg = predict.get("seg", None) if isinstance(predict, dict) else predict
    if predict_seg is not None and predict_seg.ndim == 4:
        predict_seg = torch.argmax(predict_seg, dim=1)
        predict_seg = predict_seg.squeeze()
        predict_seg = np.array(predict_seg.cpu(), dtype=np.uint8)
        if color_map:
            predict_seg *= 250 // args.num_classes
            predict_seg = np.array(predict_seg, dtype=np.uint8)
            predict_seg = cv2.applyColorMap(predict_seg, cv2.COLORMAP_JET)
            cv2.imwrite(args.inference_output, predict_seg)
        else:
            shutil.copy(args.inference_input, args.inference_output)
            cv2.imwrite(args.inference_output[:-4] + "_labels.png", np.array(predict_seg, dtype=np.uint8))
    writer.stop_and_join()


def predict(model, args, color_map=True, categories=None):
    if os.path.isdir(args.inference_input):
        assert not os.path.isfile(args.inference_output)
        args.inference_output = os.path.expanduser(args.inference_output)
        if not os.path.exists(args.inference_output):
            os.makedirs(args.inference_output, exist_ok=True)
        if categories is not None:
            write_categories(os.path.join(args.inference_output, "categories"), categories)
        imgs = os.listdir(args.inference_input)
        single_args = deepcopy(args)
        for img_name in tqdm(imgs, "running test data"):
            if ".png" not in img_name:
                continue
            single_args.inference_input = os.path.join(args.inference_input, img_name)
            single_args.inference_output = os.path.join(args.inference_output, img_name)
            predict_on_image(model, single_args, color_map=color_map)
    else:
        predict_on_image(model, args, color_map=color_map)


def main(params=None):
    # basic parameters
    parser = argparse.ArgumentParser()
    add_architecture_args(parser)
    parser.add_argument('--data', type=str, default=None, help='path to dataset')
    parser.add_argument('--inference_input', type=str, required=True, help="Path to image, folder of images or video for prediction")
    parser.add_argument('--hyper_params', type=str, required=True, help='path to hyper params of trained model')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to weights to load')
    parser.add_argument('--inference_output', type=str, default=None, required=True, help='Path to save predict image')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--cpu', action="store_false", dest="use_gpu")
    args = parser.parse_args()
    loaded_args = json.load(open(args.hyper_params))
    loaded_args.update(args.__dict__)
    args.__dict__.update(loaded_args)
    args = smart_parse_args(parser, args=args)

    if os.path.isfile(args.inference_input):
        img = cv2.imread(args.inference_input, cv2.IMREAD_UNCHANGED)
        if img.shape[0] != args.crop_height or img.shape[1] != args.crop_width:
            print("auto resizing input image")
            img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_LINEAR)
            img = img[196:-10, 80:-80, ...]
            cv2.imwrite("/tmp/img_tmp.png", img)
            args.inference_input = "/tmp/img_tmp.png"

    model = build_model(args)

    # load pretrained model if exists
    load_matching_weights(model, args.pretrained_model_path)

    with torch.no_grad():
        predict(model, args)


if __name__ == '__main__':
    main()
