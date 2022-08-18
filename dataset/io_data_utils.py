import json
import os
import shutil

import cv2
import torch
from torch.utils.data import DataLoader

from dataset.ISFLDataset import ISFLDataset, SequentialInputConfig
from dataset.LocalMapSample import LocalMapSample


def make_paths_absolute(args):
    if args.data is not None and "~" in args.data:
        args.data = os.path.expanduser(args.data)
    if hasattr(args, "save_model_path") and "~" in args.save_model_path:
        args.save_model_path = os.path.expanduser(args.save_model_path)
    if hasattr(args, "pretrained_model_path") and args.pretrained_model_path is not None and "~" in args.pretrained_model_path:
        args.pretrained_model_path = os.path.expanduser(args.pretrained_model_path)
    if hasattr(args, "save_model_path") and not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path, exist_ok=True)


def read_categories_file(args, categories_file_path):
    if hasattr(args, "save_model_path"):
        shutil.copy(categories_file_path, os.path.join(args.save_model_path, "categories"))
    with open(categories_file_path) as f:
        categories = f.read().split("\n")
        categories = [c for c in categories if c.strip() != ""]
        args.num_classes = len(categories)
    return categories


def read_categories_from_dataset(args):
    """ read a file with one segmentation category/class per line """
    if args.data is not None and os.path.exists(os.path.join(args.data, "categories")):
        categories = read_categories_file(args, os.path.join(args.data, "categories"))
    else:
        categories = ["unknown{}".format(i) for i in range(args.num_classes)]
        print("NO CATEGORIES FILE IN DATASET ://")
        import time
        time.sleep(1)
    return categories


def get_coordinate_limits_from_dataset(dataset):
    img = cv2.imread(dataset.image_list[0])
    sample = LocalMapSample(img.shape[1], img.shape[0], dataset.local_map_label_list[0], max_length=5, step_size=0.1)
    return torch.tensor(sample.limits, dtype=torch.float32)


def write_categories(target_path, categories):
    """ write a file with one segmentation category/class per line """
    if os.path.exists(target_path) and os.path.isdir(target_path):
        target_path = os.path.join(target_path, "categories")
    else:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w+") as f:
        f.write("\n".join(categories))


def use_dataset_img_size_if_unspecified(args):
    if not hasattr(args, "data") or args.data is None:
        return
    val_path = os.path.join(args.data, 'val')
    if not os.path.isdir(val_path):
        return
    any_img_path = os.path.join(val_path, os.listdir(val_path)[0])
    img = cv2.imread(any_img_path, cv2.IMREAD_UNCHANGED)
    if not hasattr(args, "crop_width") or args.crop_width is None or args.crop_height is None:
        args.crop_width = img.shape[1]
        args.crop_height = img.shape[0]
    else:
        if args.crop_width != img.shape[1]:
            print("WARNING: args.crop_width != dataset img width")
        if args.crop_height != img.shape[0]:
            print("WARNING: args.crop_width != dataset img height")


def smart_parse_args(parser, args=None):
    if args is None:
        args = parser.parse_args()
    defaults = {
        "batch_size": 16,
        "eval_batch_size": 32,
        "no_normalize": False,
        "num_workers": 8,
        "augment_on_cpu": True,
        "ignore_class_idx": 0,
        "augmentation": [],
        "multi_img_num": 1,
        "val_dataset_tags": ["real", "syn"],
        "use_gpu": True,
        "num_classes": 19,
        "crop_width": 320,
        "crop_height": 192,
    }
    defaults.update(args.__dict__)
    args.__dict__.update(defaults)
    make_paths_absolute(args)
    if args.data is not None and not os.path.exists(os.path.join(args.data, "categories")) and not os.path.exists(os.path.join(args.data, "clusters.json")):
        if os.path.exists(os.path.join(os.path.dirname(args.data), "categories")):
            args.data = os.path.dirname(args.data)
        elif os.path.exists(os.path.join(os.path.dirname(os.path.dirname(args.data)), "categories")):
            args.data = os.path.dirname(os.path.dirname(args.data))
    categories = read_categories_from_dataset(args)
    if args.num_classes is None:
        args.num_classes = len(categories)
    use_dataset_img_size_if_unspecified(args)
    args.model_width = args.crop_width
    args.model_height = args.crop_height
    return args


def init_data_loaders(args, require_lane_labels=False, require_seg_labels=False, input_channels=3, quantitative_test_data=False, shuffle=True):
    categories = read_categories_from_dataset(args)
    if args.num_classes is None:
        args.num_classes = len(categories)
    aug_settings = json.load(open(os.path.join(os.path.dirname(os.path.realpath(__file__)), "default_augmentation.json")))
    aug_settings.update(dict(s.split("=") for s in args.augmentation if s.count("=") == 1))
    for k in aug_settings.keys():
        if isinstance(aug_settings[k], str):
            aug_settings[k] = aug_settings[k].replace("\n", "")
            try:
                aug_settings[k] = json.loads(aug_settings[k])
            except ValueError:
                pass
    args.augmentation = ["{}={}".format(k, json.dumps(v)) for k, v in aug_settings.items()]
    use_augmentation = (not args.disable_augmentation) if hasattr(args, "disable_augmentation") else True

    # create dataset and dataloader
    train_path = os.path.join(args.data, 'train')
    train_label_path = os.path.join(args.data, 'train_labels')
    val_path = os.path.join(args.data, 'val')
    val_label_path = os.path.join(args.data, 'val_labels')
    test_path = os.path.join(args.data, 'test')
    test_label_path = os.path.join(args.data, 'test_labels')

    if "ground" in categories:
        seg_ground_idx = categories.index("ground")
    else:
        print("CLASS ground NOT IN CATEGORIES")
        seg_ground_idx = 1

    if hasattr(args, "multi_img_num") and args.multi_img_num > 1:
        seq_input_cfg = SequentialInputConfig(num_imgs=args.multi_img_num, num_pixel_expand_bottom=args.multi_img_expand,
                                              restrict_overlap=args.multi_img_overlap)
    else:
        seq_input_cfg = None
    dataset_train = ISFLDataset(train_path, train_label_path, normalize=not args.no_normalize, seq_input_config=seq_input_cfg,
                                require_lane_labels=require_lane_labels, require_seg_labels=require_seg_labels, seg_ignore_idx=args.ignore_class_idx,
                                aug_settings=aug_settings, seg_ground_idx=seg_ground_idx, use_augmentation=use_augmentation,
                                input_channels=input_channels)
    if len(dataset_train) > 0:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            drop_last=True
        )
    else:
        dataloader_train = None

    dataset_val = ISFLDataset(val_path, val_label_path, normalize=not args.no_normalize, seq_input_config=seq_input_cfg,
                              require_lane_labels=require_lane_labels, require_seg_labels=require_seg_labels, seg_ignore_idx=args.ignore_class_idx,
                              aug_settings=aug_settings, seg_ground_idx=seg_ground_idx, use_augmentation=True,
                              input_channels=input_channels)
    if dataset_train.local_map_transform is None:
        dataset_train.local_map_transform = dataset_val.local_map_transform
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    if not quantitative_test_data:
        require_lane_labels = require_seg_labels = False
    dataset_test = ISFLDataset(test_path, test_label_path, normalize=not args.no_normalize, seq_input_config=seq_input_cfg,
                               require_lane_labels=require_lane_labels, require_seg_labels=require_seg_labels, seg_ignore_idx=args.ignore_class_idx,
                               aug_settings=aug_settings, seg_ground_idx=seg_ground_idx, use_augmentation=False,
                               input_channels=input_channels, local_map_transform=dataset_train.local_map_transform)
    dataloader_test = DataLoader(
        dataset_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    return dataloader_train, dataloader_val, dataloader_test
