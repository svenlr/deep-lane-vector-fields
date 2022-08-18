import argparse
import json
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader

from dataset.ISFLDataset import ISFLDataset, SequentialInputConfig
from dataset.io_data_utils import make_paths_absolute, smart_parse_args
from nn_utils.geometry_utils import world_to_ipm


def main():
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='path of training data')
    parser.add_argument('--crop_height', type=int, default=192, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=320, help='Width of cropped/resized input image to network')
    parser.add_argument('--mode', type=str, default="train", choices=["train", "val", "test"], help="which part of dataset you want to see")
    args = smart_parse_args(parser)

    make_paths_absolute(args)

    # create dataset and dataloader
    input_path = os.path.join(args.data, args.mode)
    label_path = os.path.join(args.data, args.mode + '_labels')
    seq_input_cfg = SequentialInputConfig()
    seq_input_cfg.num_images = 3
    seq_input_cfg.num_pixel_expand_bottom = 32
    seq_input_cfg = None
    aug_settings = json.load(open("dataset/default_augmentation.json"))
    dataset_train = ISFLDataset(input_path, label_path, normalize=False, seq_input_config=seq_input_cfg,
                                use_augmentation=args.mode != "test", require_seg_labels=False, require_lane_labels=False,
                                aug_settings=aug_settings, seg_ignore_idx=0, seg_ground_idx=1, local_map_length=7)
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )

    if dataset_train.local_map_transform is not None:
        pixels_per_meter = dataset_train.local_map_transform["pixels_per_meter"]
        car_to_image_offset = dataset_train.local_map_transform["car_to_image_offset"]
    else:
        pixels_per_meter = car_to_image_offset = None

    for i, sample in enumerate(dataloader_train):
        data = sample["img"].cpu()
        if "seg" in sample and sample["seg"] is not None:
            seg = sample["seg"].permute([1, 2, 0]).squeeze().cpu().numpy()
            seg = (seg.astype(np.float32) * 255 / args.num_classes).clip(0, 255).astype(np.uint8)
        else:
            seg = None
        img = np.squeeze(data.numpy(), 0)
        img = np.swapaxes(img, 0, 2)
        img = np.swapaxes(img, 0, 1)
        # set red channel to 1 - weights
        # img[..., 2] = 1 - distance_decaying_loss_weights(img=img, mode="euclid", start_decay_dist_percentage=0.2, end_decay_dist_percentage=0.6,
        #                                                  x_decay=1.0, y_decay=0.7) # x_decay and y_decay are shape parameters for the decay
        img *= 255
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)
        img = img[..., ::-1]
        img = img.copy()
        width, height = img.shape[1], img.shape[0]
        if img.ndim == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        # img = cv2.copyMakeBorder(img, 0, 300, 0, 0, cv2.BORDER_CONSTANT, value=0)

        if "local_map" in sample and "right_lane" in sample["local_map"]:
            last_pos = None
            for j in range(sample["local_map"]["right_lane"]["left_marking_all"].shape[2]):
                pos = sample["local_map"]["right_lane"]["left_marking_all"][0, :, j].cpu().numpy()
                if last_pos is not None:
                    p1 = np.array(world_to_ipm(last_pos, width, height, pixels_per_meter, car_to_image_offset)).astype(np.int32)
                    p2 = np.array(world_to_ipm(pos, width, height, pixels_per_meter, car_to_image_offset)).astype(np.int32)
                    if j <= sample["local_map"]["num_visible_points"][0]:
                        cv2.line(img, tuple(p1), tuple(p2), (255, 0, 0))
                    else:
                        cv2.line(img, tuple(p1), tuple(p2), (60, 0, 120))
                last_pos = pos

        if seg is not None:
            seg = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
            img = np.hstack([img, seg])
        cv2.imshow("test", cv2.resize(img, (int(500 * img.shape[1] / img.shape[0]), 500), interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey(0)
        if k == 27 or k == ord('q'):
            break


if __name__ == '__main__':
    main()
