import cv2
import numpy as np
import torch


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def tensor_to_np_uint8(img):
    return torch.tensor(((img - img.min()) / (img.max() - img.min()) * 255)).cpu().numpy().clip(0, 255).astype(np.uint8)


def _dict_to(d, func, convert_class=torch.Tensor):
    if isinstance(d, dict):
        ret = {}
        for key in d.keys():
            ret[key] = _dict_to(d[key], func, convert_class)
    elif isinstance(d, convert_class):
        ret = func(d)
    elif isinstance(d, list):
        ret = [_dict_to(e, func, convert_class) for e in d]
    else:
        ret = d
    return ret


def dict_to_cuda(label_batch):
    return _dict_to(label_batch, lambda x: x.cuda())


def dict_to_cpu(label_batch):
    return _dict_to(label_batch, lambda x: x.cpu())


def dict_to_numpy(label_batch):
    return _dict_to(label_batch, lambda x: x.cpu().numpy())


def dict_to_list(label_batch):
    return _dict_to(label_batch, lambda x: x.cpu().numpy().tolist())


def rotate_image(image, angle, mode=cv2.INTER_LINEAR, center=None):
    if center is None:
        center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=mode)
    return result
