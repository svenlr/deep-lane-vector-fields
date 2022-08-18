import json
import os
import random
from pathlib import Path

import cv2
# np.random.bit_generator = np.random._bit_generator
import imgaug.augmenters as iaa
import numpy as np
import scipy.signal
import torch
from PIL import Image
from torchvision import transforms

from dataset.LocalMapSample import LocalMapSample
from dataset.data_sample_utils import rotate_image
from loss.loss_utils import distance_decaying_loss_weights
from nn_utils.geometry_utils import world_to_ipm_vector, world_to_ipm, ipm_to_world, vector_of, extrapolate_or_shrink_points, points_to_tensor, left_orthogonal
from nn_utils.local_map_utils import make_coordinate_image
from nn_utils.math_utils import rbf_activations


def distance_increasing_blur(img, filters, start_rel=0.2, end_rel=0.8, ellipse_shape=1.0, mode="euclid"):
    filter_mus = np.linspace(start_rel, end_rel, num=len(filters))[::-1]
    filter_sigmas = 0.6 * np.abs(end_rel - start_rel) / len(filters)
    w = distance_decaying_loss_weights(start_decay_dist=start_rel, end_decay_dist=end_rel,
                                       img=img, v_decay=1, h_decay=1 / ellipse_shape, mode=mode)
    filter_weights = rbf_activations(w, filter_mus, filter_sigmas)
    ret = np.zeros(shape=img.shape, dtype=img.dtype)
    for i, f in enumerate(filters):
        ret += (1 + (1 - w) * 0.5) * filter_weights[:, :, i] * f(image=img)
    return ret


class SequentialInputConfig:
    def __init__(self, num_imgs=3, num_pixel_expand_bottom=32, restrict_overlap=8):
        self.num_images = num_imgs
        self.num_pixel_expand_bottom = num_pixel_expand_bottom
        self.restrict_overlap = restrict_overlap


class ISFLDataset(torch.utils.data.Dataset):
    def __init__(self, image_path, label_path, normalize=True, use_augmentation=True, local_map_transform=None, seq_input_config=None,
                 local_map_length=4.8, local_map_step_size=0.1, gray_scale=True, require_seg_labels=True, require_lane_labels=True,
                 disable_augmentation_by_tags="real", aug_settings=None, seg_ignore_idx=None, seg_ground_idx=None, input_channels=3):
        super().__init__()
        if LocalMapSample is None:
            print("LocalMapSample could not be imported, please start PyCharm from command line (or switch to Linux).")
        aug_settings = dict() if aug_settings is None else aug_settings
        if use_augmentation:
            self.street_viz_aug_probability = aug_settings["viz_aug"]
            self.lane_mask_for_seg_train = aug_settings["lane_mask_for_seg_train"] > 0
            self.perspective_aug_px = aug_settings["tf_px_max"]
            self.perspective_aug_px_noise = aug_settings["tf_px_noise"]
            if self.street_viz_aug_probability > 0 and require_seg_labels:
                # when training segmentation with visibility augmentation, we need an ignore index
                assert seg_ignore_idx is not None and seg_ground_idx is not None  # training seg with viz aug requires ignore_idx!
        self.image_dir = image_path
        self.label_dir = label_path
        self.seq_input_config = seq_input_config  # type: SequentialInputConfig
        self.require_seg_labels = require_seg_labels
        self.require_lane_labels = require_lane_labels
        self.gray_scale = gray_scale
        self.input_channels = input_channels
        self.local_map_transform = local_map_transform
        self.local_map_length = local_map_length
        self.local_map_step_size = local_map_step_size
        self.local_map_num_points = int(local_map_length / local_map_step_size)
        self.use_augmentation = use_augmentation
        self.disable_augmentation_by_tags = disable_augmentation_by_tags
        self.seg_ignore_idx = seg_ignore_idx
        self.seg_ground_idx = seg_ground_idx
        image_file_names = os.listdir(image_path)
        image_file_names.sort()
        label_file_names = set(os.listdir(label_path))
        self.existing_seg_label_files = set([f for f in label_file_names if f[-4:] == ".png"])
        self.existing_lane_label_files = set([f for f in label_file_names if f[-5:] == ".json"])
        # reduce image list if some labels are missing and labels are required
        if self.require_seg_labels:
            image_file_names = [f for f in image_file_names if f in self.existing_seg_label_files]
        if self.require_lane_labels:
            image_file_names = [f for f in image_file_names if f.replace(".png", ".json") in self.existing_lane_label_files]
        self.image_list = [os.path.join(image_path, f) for f in image_file_names]
        self.label_list = [os.path.join(label_path, f) for f in image_file_names]
        self.local_map_label_list = [os.path.join(label_path, f.replace(".png", ".json")) for f in image_file_names]
        if len(self.image_list) > 0:
            self.h, self.w = cv2.imread(self.image_list[0]).shape[:2]
        else:
            self.w, self.h = 320, 192
        if self.local_map_transform is None and len(self.local_map_label_list) > 0 and os.path.exists(self.local_map_label_list[0]):
            json_data = json.load(open(self.local_map_label_list[0]))
            self.local_map_transform = json_data["transform"]
        if self.local_map_transform is not None:
            self.coordinate_limits = LocalMapSample.transform_to_limits(self.w, self.h, {"transform": self.local_map_transform})
        else:
            self.coordinate_limits = None

        # normalization
        if normalize:
            self.to_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            self.to_tensor = transforms.Compose([transforms.ToTensor()])

        self.simplex_noise_alpha = iaa.BlendAlphaSimplexNoise(
            foreground=iaa.Add(40),
            per_channel=False,
            upscale_method="linear",
            aggregation_method="max",
            size_px_max=(15, 40),
            sigmoid_thresh=5,
        )
        self.gaussian_noise = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=(0, 3)),
        ])
        self.jpeg_compression = iaa.JpegCompression(compression=(5, 10))

    def seed_augmentation(self, seed):
        self.simplex_noise_alpha.reseed(random_state=seed)
        self.gaussian_noise.reseed(random_state=seed)
        random.seed(seed)

    def augment_sample(self, img, label, local_map_label):
        # AUGMENTATION ##############
        img = np.array(img, dtype=np.float32)
        wh_shape = img.shape[:2]

        # these augmentation settings are the same for all channels of the input image
        increase_marking_intensity = random.random() < 0.4
        motion_blur = random.random() < 0.7
        gaussian_before_blur = random.random() < 0.7
        if random.random() < 0.8:
            motion_blur_angle = -20 + random.random() * 40
        else:
            motion_blur_angle = random.uniform(70, 100)
        end_motion_blur = 3 + int(8 * random.random() * int(increase_marking_intensity))

        random_add = random.uniform(-30, 30)
        random_brightness = random.uniform(0.3, 1.3)

        if local_map_label is not None and len(local_map_label.points) >= 2:
            img, label, local_map_label = self.augment_visibility(img, label, local_map_label, aug_probability=self.street_viz_aug_probability)

        img, label, local_map_label = self.augment_perspective(img, label, local_map_label)

        img += random_add

        if img.ndim == 2:
            # add channel dimension
            img = np.expand_dims(img, -1)

        if motion_blur and gaussian_before_blur:
            for i in range(0, img.shape[2], 3):
                img[..., i:i + 3] = self.gaussian_noise(image=img[..., i:i + 3])

        img = img.astype(np.float32)
        for c in range(img.shape[2]):
            # only one channel of the image
            imc = img[..., c]
            # these random augmentation are different per channel
            marking_intensity_scaling = random.random()
            use_simplex_before_blur = random.random() < 0.75
            use_simplex_after_blur = random.random() < 0.75 and not use_simplex_before_blur
            # make markings thicker and object boundaries thicker by applying a random gauss-like filter (adding weighted neighbour pixels)
            # This looks like increasing the marking intensity and "glow", which was inspired by cup-style images.
            if increase_marking_intensity:
                r1 = 0.5 + marking_intensity_scaling * 0.5
                r2 = 0.2 * (1 - marking_intensity_scaling)
                imc[1:-1, 1:-1] = (r1 * imc[1:-1, 1:-1] + r2 * (imc[1:-1, :-2] + imc[1:-1, 2:] + imc[2:, 1:-1] + imc[:-2, 1:-1]))

            if motion_blur:
                if use_simplex_before_blur:
                    imc += rotate_image(self.simplex_noise_alpha(image=np.zeros(shape=wh_shape, dtype=img.dtype)), -45 + random.random() * 90)
                filters = [iaa.MotionBlur(angle=motion_blur_angle, k=k) for k in range(3, end_motion_blur + 1)]
                imc = distance_increasing_blur(imc, filters, start_rel=0.3, end_rel=0.8, ellipse_shape=8, mode="square")

            if use_simplex_after_blur:
                imc += rotate_image(self.simplex_noise_alpha(image=np.zeros(shape=wh_shape, dtype=img.dtype)), -45 + random.random() * 90)

            img[..., c] = imc

        img *= random_brightness

        img = np.array(np.clip(img, 0, 255), dtype=np.uint8)

        if random.random() < 0.2:
            # jpeg compression has to be done on uint8
            for i in range(0, img.shape[2], 3):
                img[..., i:i + 3] = self.jpeg_compression(image=img[..., i:i + 3])
        if not gaussian_before_blur:
            for i in range(0, img.shape[2], 3):
                img[..., i:i + 3] = self.gaussian_noise(image=img[..., i:i + 3])

        # END AUGMENTATION #############
        return img, label, local_map_label

    def augment_visibility(self, img, seg, local_map_label, aug_probability=0.4):
        median_kernel_size = int(np.round(3 / self.local_map_step_size) + 1)  # allow visibility gaps of up to 1.5 meters => needs 3 meters median filter
        median_kernel_size += median_kernel_size % 2 == 0  # make odd
        if random.random() < aug_probability:
            idx_viz_aug = -1
            viz_end_dist = 2 + random.random() * 3
            viz_red_start_dist = viz_end_dist - random.random() * 0.5
            w = distance_decaying_loss_weights(start_decay_dist=viz_red_start_dist, img=img, v_decay=1, h_decay=1,
                                               min_weight=0, end_decay_dist=viz_end_dist, car_pos=[0, 0], limits=self.coordinate_limits)
            img *= w
            if seg is not None and self.seg_ignore_idx is not None:
                seg[w == 0] = self.seg_ignore_idx  # we can't train what we don't see
            for points in [local_map_label.points, local_map_label.left_points, local_map_label.right_points]:
                points_dist = np.sqrt((points ** 2).sum(axis=1))
                visibility = points_dist < viz_end_dist
                idx_viz_aug = self.determine_visibility_idx(idx_viz_aug, visibility, median_kernel_size)
                if idx_viz_aug == len(visibility):
                    break
            local_map_label.cut_from(idx_viz_aug)
        if self.lane_mask_for_seg_train:
            # points = np.expand_dims(np.expand_dims(local_map_label.points[::10, ...], 0), 0)
            # distances = np.sqrt(((np.expand_dims(coord_img, 2) - points) ** 2).sum(axis=3).min(axis=2))
            mask = np.ones(shape=seg.shape[:2], dtype=np.uint8)
            points_ipm = world_to_ipm(local_map_label.points, img.shape[1], img.shape[0], self.local_map_transform["pixels_per_meter"],
                                      self.local_map_transform["car_to_image_offset"])
            for p in points_ipm[::5]:
                mask = cv2.circle(mask, tuple(p), int(1.1 * self.local_map_transform["pixels_per_meter"]), 0, thickness=-1)
            if seg is not None and self.seg_ignore_idx is not None:
                seg[np.logical_and(mask, seg != self.seg_ground_idx)] = self.seg_ignore_idx
        return img, seg, local_map_label

    def configure_visibility(self, img, seg, local_map_label, occluded_triangles_slope=0.625):
        car_to_image_offset = self.local_map_transform.get("car_to_image_offset", 0.1)
        if local_map_label is not None and local_map_label.points.ndim == 2 and len(local_map_label.points) > 2:
            median_kernel_size = int(np.round(3 / self.local_map_step_size) + 1)  # allow visibility gaps of up to 1.5 meters => needs 3 meters median filter
            median_kernel_size += median_kernel_size % 2 == 0  # make odd
            idx_triangle = -1
            for points in [local_map_label.points, local_map_label.left_points, local_map_label.right_points]:
                visibility = np.abs(points[:, 0] + car_to_image_offset) / (np.abs(points[:, 1]) + 1e-5) > occluded_triangles_slope
                idx_triangle = self.determine_visibility_idx(idx_triangle, visibility, median_kernel_size)
                if idx_triangle == len(visibility):
                    break
            local_map_label.cut_from(idx_triangle)
        coord_img = make_coordinate_image(img.shape[1], img.shape[0], self.coordinate_limits)
        triangles = np.logical_and(np.abs(coord_img[..., 0] + car_to_image_offset) / np.abs(coord_img[..., 1] + 1e-5) < occluded_triangles_slope,
                                   np.abs(coord_img[..., 1]) < 0.8)
        if seg is not None and self.seg_ignore_idx is not None:
            seg[triangles] = self.seg_ignore_idx
        return img, seg, local_map_label

    def augment_perspective(self, img, seg, local_map_label):
        src_points = np.array([[0.0, 0], [img.shape[1], 0],  # top row: left right
                               [0, img.shape[0]], [img.shape[1], img.shape[0]]], dtype=np.float32)  # bottom row: left right
        dst_points = src_points.copy()
        num = self.perspective_aug_px
        num_noise = self.perspective_aug_px_noise
        x_top = random.uniform(-1, 1) * num
        y_top = x_top + num_noise * random.uniform(-1, 1)
        # x_top = y_top = 0
        dst_points[0, 0] -= x_top + random.uniform(-num_noise, num_noise)
        dst_points[1, 0] += x_top + random.uniform(-num_noise, num_noise)
        dst_points[0, 1] -= y_top + random.uniform(-num_noise, num_noise)
        dst_points[1, 1] -= y_top + random.uniform(-num_noise, num_noise)
        dst_points[2, 0] += random.uniform(-num_noise, num_noise)
        dst_points[2, 1] += random.uniform(-num_noise / 2, num_noise)
        dst_points[3, 0] += random.uniform(-num_noise, num_noise)
        dst_points[3, 1] += random.uniform(-num_noise / 2, num_noise)
        transform = cv2.getPerspectiveTransform(src_points, dst_points)
        img = cv2.warpPerspective(img, transform, img.shape[:2][::-1], flags=cv2.INTER_LINEAR)
        if seg is not None:
            seg = cv2.warpPerspective(seg, transform, seg.shape[:2][::-1], flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=self.seg_ignore_idx if self.seg_ignore_idx is not None else 0)
        if local_map_label is not None and local_map_label.num_points > 0:
            local_map_label = transform_lane_label(local_map_label, transform, img.shape)
            # we need to re-cut the lane label because it might exceed the limit in case of perspective augmentations
            try:
                local_map_label.cut_points_to_rect(local_map_label.limits, self.local_map_length)
            except AssertionError:
                pass
        return img, seg, local_map_label

    @staticmethod
    def determine_visibility_idx(current_max_viz_idx, visibility, median_kernel_size):
        padded_visibility = np.concatenate([np.repeat(1, median_kernel_size), visibility], axis=0)
        visibility = scipy.signal.medfilt(padded_visibility, kernel_size=median_kernel_size)[median_kernel_size:]
        if len(visibility) == 0:
            return 0
        idx_per_points = np.argmin(visibility)
        if idx_per_points == 0 and np.all(visibility > 0):
            current_max_viz_idx = len(visibility)
        else:
            current_max_viz_idx = max(current_max_viz_idx, idx_per_points)
        return current_max_viz_idx

    def _get_paths(self, index):
        if index < 0 or index >= len(self):
            return None, None, None
        label_path = None
        local_map_path = None
        if self.require_seg_labels or os.path.basename(self.label_list[index]) in self.existing_seg_label_files:
            label_path = self.label_list[index]
        if self.require_lane_labels or os.path.basename(self.local_map_label_list[index]) in self.existing_lane_label_files:
            local_map_path = self.local_map_label_list[index]
        return self.image_list[index], label_path, local_map_path

    def _get_single_sample(self, index):
        image_path, seg_label_path, lane_label_path = self._get_paths(index)
        img = Image.open(image_path)
        if seg_label_path is not None:
            seg_label = Image.open(seg_label_path)
            seg_label = np.array(seg_label, dtype=np.uint8)
        else:
            seg_label = None
        if lane_label_path is not None:
            try:
                lane_label = LocalMapSample(img.width, img.height, lane_label_path, max_length=20,
                                            step_size=self.local_map_step_size, cut_points_to_rect=True)
            except AssertionError:
                lane_label = None
        else:
            lane_label = None
        img = np.array(img, dtype=np.uint8)
        if img.ndim == 3 and self.gray_scale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, seg_label, lane_label

    def _should_augment(self, lane_label):
        if not self.use_augmentation:
            return False
        if lane_label is None:
            return True
        for tag in lane_label.json_data.get("tags", []):
            if tag in self.disable_augmentation_by_tags:
                return False
        return True

    def __getitem__(self, index):
        if self.seq_input_config is not None:
            img, seg_label, lane_label = self._stack_multi_input(index, self.seq_input_config.num_images)
            img, seg_label, lane_label = self.configure_visibility(img, seg_label, lane_label)
            if self._should_augment(lane_label):
                img, seg_label, lane_label = self.augment_sample(img, seg_label, lane_label)
            while img.shape[2] < 3:
                img = np.concatenate([img, np.zeros(shape=(*img.shape[:2], 1), dtype=img.dtype)], axis=2)
        else:
            img, seg_label, lane_label = self._get_single_sample(index)
            img, seg_label, lane_label = self.configure_visibility(img, seg_label, lane_label)
            if self._should_augment(lane_label):
                img, seg_label, lane_label = self.augment_sample(img, seg_label, lane_label)
            if self.input_channels == 3:
                if img.ndim == 2:
                    img = np.expand_dims(img, -1)
                while img.shape[2] < 3:
                    img = np.concatenate([img, img[..., :1]], axis=2)

        # image -> [3, H, W]
        img = self.to_tensor(img / 255).float()
        ret = {
            "img": img,
            "identifier": Path(self.image_list[index]).stem,
        }
        if seg_label is not None:
            seg_label = np.array(seg_label)
            if seg_label.ndim == 3:
                seg_label = seg_label[:, :, 0]
            ret["seg"] = torch.from_numpy(seg_label).long()

        # ensure that the first point is just below the lower border of the image
        if lane_label is not None and lane_label.left_points.size >= 4 and lane_label.right_points.size >= 4:
            first_idx = min(np.argmax(lane_label.left_points[:, 0] >= self.coordinate_limits[0, 0]),
                            np.argmax(lane_label.right_points[:, 0] >= self.coordinate_limits[0, 0]))
            if first_idx < lane_label.points.shape[0]:
                lane_label.cut_before(max(first_idx - 1, 0))

        if lane_label is not None:
            visible_points_mask = np.concatenate([lane_label.visibility_mask, np.zeros(max(0, self.local_map_num_points - lane_label.num_points, ))])
            visible_points_mask = visible_points_mask[:self.local_map_num_points]
            num_visible_points = lane_label.num_points if np.all(lane_label.visibility_mask) else np.argmin(lane_label.visibility_mask)
        else:
            visible_points_mask = np.repeat(0, self.local_map_num_points).astype(np.float32)
            num_visible_points = 0
        max_length = 8
        max_num_points = int(np.round(max_length / self.local_map_step_size))
        if lane_label is not None and lane_label.num_points > 0:
            right_tensor = points_to_tensor(extrapolate_or_shrink_points(lane_label.right_points, self.local_map_num_points))
            left_tensor = points_to_tensor(extrapolate_or_shrink_points(lane_label.points, self.local_map_num_points))
            all_right_tensor = points_to_tensor(extrapolate_or_shrink_points(lane_label.right_points, max_num_points))
            all_left_tensor = points_to_tensor(extrapolate_or_shrink_points(lane_label.points, max_num_points))
        else:
            all_right_tensor = points_to_tensor(np.zeros((max_num_points, 2), np.float32))
            all_left_tensor = points_to_tensor(np.zeros((max_num_points, 2), np.float32))
            right_tensor = points_to_tensor(np.zeros((self.local_map_num_points, 2), np.float32))
            left_tensor = points_to_tensor(np.zeros((self.local_map_num_points, 2), np.float32))
        ret["local_map"] = {
            "num_visible_points": int(num_visible_points),
            "visibility_mask": visible_points_mask.astype(np.float32),
            "right_lane": {
                "right_marking": right_tensor,
                "right_marking_all": all_right_tensor,
                "left_marking": left_tensor,
                "left_marking_all": all_left_tensor,
            },
            "step_size": self.local_map_step_size,
            "transform": self.local_map_transform,
        }
        ret["index"] = index
        if lane_label is not None:
            ret["tags"] = " ".join(lane_label.json_data.get("tags", []))
        else:
            ret["tags"] = ""
        return ret

    def __len__(self):
        return len(self.image_list)

    def _stack_multi_input(self, index, num_images, prob_drop_early=0.1, max_step=2):
        """
        Build an input image with sequential images stacked in the channel dimension.
        """
        img, label, local_map_label = self._get_single_sample(index)
        img_stacked = cv2.copyMakeBorder(img, 0, self.seq_input_config.num_pixel_expand_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        # img_stacked = img
        img_stacked = np.expand_dims(img_stacked, -1)
        if local_map_label is not None:
            car_pos = np.array(local_map_label.json_data["transform"]["global_car_pos"])
            car_angle = local_map_label.json_data["transform"]["global_car_angle"]
            pixels_per_meter = local_map_label.json_data["transform"]["pixels_per_meter"]
            expand_bottom_m = self.seq_input_config.num_pixel_expand_bottom / pixels_per_meter
            car_to_image_offset = local_map_label.json_data["transform"]["car_to_image_offset"]
            other_index = index
            if random.random() < prob_drop_early and self.use_augmentation:
                num_images_seen = int(random.random() * num_images)
            else:
                num_images_seen = num_images
            for i in range(num_images_seen - 1):
                # go backwards in time
                other_index -= 1
                # search the next images that allows to fill all of the expanded area at the bottom of the image
                while True:
                    _, _, other_local_map_path = self._get_paths(other_index)
                    if other_local_map_path is None:
                        break
                    other_transform = json.load(open(other_local_map_path))["transform"]
                    if np.linalg.norm(np.array(other_transform["global_car_pos"]) - car_pos) > expand_bottom_m * 0.75 * (i + 1):
                        break
                    other_index -= 1
                if self.use_augmentation:
                    other_index -= int(random.random() * max_step)
                if other_index < 0:
                    break
                img2, label2, local_map_label2 = self._get_single_sample(other_index)
                if local_map_label2 is None:
                    break
                img2 = cv2.copyMakeBorder(img2, 0, self.seq_input_config.num_pixel_expand_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
                if "seq" in local_map_label.json_data and "seq" in local_map_label2.json_data:
                    if local_map_label.json_data["seq"] != local_map_label2.json_data["seq"]:
                        break
                car_pos2 = np.array(local_map_label2.json_data["transform"]["global_car_pos"])
                car_angle2 = local_map_label2.json_data["transform"]["global_car_angle"]
                center = (img.shape[1] / 2., img.shape[0] + car_to_image_offset * pixels_per_meter)
                # img2 = cv2.line(img2, tuple(np.array(center, dtype=np.int)), tuple(np.array( center, dtype=np.int)), 255, thickness=10)
                # cv2.imshow("img", img2)
                # cv2.waitKey(0)
                img2 = rotate_image(img2, np.rad2deg(car_angle2 - car_angle), mode=cv2.INTER_LINEAR, center=center)
                # label2 = rotate_image(label2, car_angle2 - car_angle, mode=cv2.INTER_NEAREST)
                world_diff_vec = car_pos2 - car_pos
                rel_pos = np.array([np.dot(vector_of(car_angle), world_diff_vec), np.dot(left_orthogonal(vector_of(car_angle)), world_diff_vec)])
                move = np.array(world_to_ipm_vector(rel_pos, pixels_per_meter)).astype(np.int)
                if np.max(np.abs(move)) < img.shape[1]:
                    move_x = move[0]
                    move_y = move[1]
                    if self.use_augmentation:
                        move_x += random.uniform(-4, 4)
                        move_y += random.uniform(-4, 4)
                    translation_mat = np.array([[1, 0, move_x],
                                                [0, 1, move_y]], dtype=np.float32)
                    img2 = cv2.warpAffine(img2, translation_mat, (img2.shape[1], img2.shape[0]))
                if img2.ndim == 2:
                    img2 = np.expand_dims(img2, -1)
                if np.max(np.abs(move)) < img.shape[1]:
                    if self.seq_input_config.restrict_overlap is not None and self.seq_input_config.restrict_overlap > 0:
                        img2[:img.shape[0] - self.seq_input_config.restrict_overlap, ...] = 0
                    img_stacked = np.concatenate([img_stacked, img2], axis=2)
        return img_stacked, label, local_map_label


def create_sharp_edge_reflection_mask(img_width, img_height, start_dist_rel=0.3, end_dist_rel=0.8, start_angle=0.1, end_angle=0.3,
                                      smoothness_longitudinal=0.3, smoothness_transverse=0.2, shape_func=lambda x: x, shape=2):
    smoothness_longitudinal = np.maximum(smoothness_longitudinal, 0.001)
    smoothness_transverse = np.maximum(smoothness_transverse, 0.001)
    w = 1 - distance_decaying_loss_weights(start_decay_dist=start_dist_rel, end_decay_dist=end_dist_rel,
                                           img_width=img_width, img_height=img_height, v_decay=shape, h_decay=1 / shape, min_weight=0, mode="l1")
    longitudinal_act = np.logical_and(w >= start_dist_rel, w <= end_dist_rel).astype(np.float32)
    longitudinal_act = np.maximum((smoothness_longitudinal - shape_func(np.abs(w - start_dist_rel))) / smoothness_longitudinal, longitudinal_act)
    longitudinal_act = np.maximum((smoothness_longitudinal - shape_func(np.abs(w - end_dist_rel))) / smoothness_longitudinal, longitudinal_act)
    # cv2.imshow("act_test", longitudinal_act)
    longitudinal_act = longitudinal_act
    car_pos = np.array([img_height, (img_width - 1) / 2.0])
    indices = np.indices((img_height, img_width))
    indices = np.swapaxes(indices, 0, 2)
    indices = np.swapaxes(indices, 0, 1)
    vectors = indices - car_pos
    angles = np.arctan2(vectors[..., 1], -vectors[..., 0])
    # cv2.imshow("vectors0", (vectors[..., 1] - np.min(vectors[..., 1])) / np.max(vectors[..., 1] - np.min(vectors[..., 1])))
    # print(np.max(angles), np.min(angles))
    # cv2.imshow("angles", (angles - np.min(angles)) / (np.max(angles) - np.min(angles)))
    transverse_act = np.logical_and(angles >= start_angle, angles <= end_angle).astype(np.float32)
    transverse_act = np.maximum((smoothness_transverse - shape_func(np.abs(angles - start_angle))) / smoothness_transverse, transverse_act)
    transverse_act = np.maximum((smoothness_transverse - shape_func(np.abs(angles - end_angle))) / smoothness_transverse, transverse_act)
    combined_activations = longitudinal_act * transverse_act
    # cv2.imshow("lin_act", longitudinal_act / np.max(longitudinal_act) * 0.5)
    # cv2.waitKey(0)
    return combined_activations


def create_sharp_edge_reflections(img_width, img_height):
    mask = None
    for i in range(5):
        start_angle = random.uniform(-np.pi / 2, np.pi / 2)
        end_angle = start_angle + random.uniform(0.03, 0.5)
        start_dist = random.uniform(0.3, 0.8)
        end_dist = start_dist + random.uniform(0.2, 1.0)
        intensity = random.uniform(0.1, 0.3)
        start_dist = 0.3
        smoothness_l = np.minimum(start_dist * 0.7, 0.3)
        smoothness_t = random.uniform(0.01, 0.1)
        m = intensity * create_sharp_edge_reflection_mask(img_width, img_height, start_dist, end_dist, start_angle, end_angle, smoothness_l, smoothness_t)
        mask = m if mask is None else mask + m
    mask = np.minimum(mask, 0.4)
    return mask


def transform_lane_label(local_map_sample, tf, img_shape):
    pixels_per_meter = local_map_sample.json_data["transform"]["pixels_per_meter"]
    car_to_image_offset = local_map_sample.json_data["transform"]["car_to_image_offset"]
    height, width = img_shape[:2]
    for points in [local_map_sample.points, local_map_sample.right_points, local_map_sample.left_points]:
        points_ipm = world_to_ipm(points, width, height, pixels_per_meter, car_to_image_offset)
        points_ipm = cv2.perspectiveTransform(np.array([points_ipm]), tf)
        if points_ipm is not None:
            points[...] = ipm_to_world(points_ipm[0], width, height, pixels_per_meter, car_to_image_offset)
    return local_map_sample


if __name__ == '__main__':
    white_img = np.ones((300, 300), dtype=np.uint8) * 255
    h = cv2.getPerspectiveTransform(np.array([[0, 0], [300, 0], [300, 300], [0, 300]], np.float32),
                                    np.array([[10, 20], [280, 0], [290, 250], [0, 300]], np.float32))
    white_img = cv2.warpPerspective(white_img, h, white_img.shape[::-1])
    print(cv2.perspectiveTransform(np.array([[[300, 0.0], [0, 10]]]), h))
    p = (h @ np.array([300, 0, 1]))
    p_B = 1 / p[2] * p[:2]
    print(p_B)
    # cv2.imshow("test", white_img)
    # cv2.waitKey(0)
    # mask = create_sharp_edge_reflections(304, 200)
    # cv2.imshow("mask", mask)
    # cv2.waitKey(0)
