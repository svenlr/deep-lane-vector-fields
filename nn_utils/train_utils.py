import torch.nn as nn
import torch
import numpy as np

from nn_utils.seg_metrics import one_hot


def load_matching_weights(model, pretrained_model_path, verbose=True):
    """ load weights from path into the given model if the name and the tensor shape matches """
    if isinstance(pretrained_model_path, list):
        for path in pretrained_model_path:
            load_matching_weights(model, path, verbose=verbose)
        return
    if verbose:
        print('load model from %s ...' % pretrained_model_path)
    loaded_state_dict = torch.load(pretrained_model_path)
    if hasattr(loaded_state_dict, "state_dict"):
        loaded_state_dict = loaded_state_dict.state_dict()
    own_state = model.state_dict()
    restored_parameter_names = []
    for name_, param in loaded_state_dict.items():
        found = False
        shape_mismatch = None, None
        for name in [name_, "module." + name_, "module.encoder." + name_, "module.module." + name_, "module.module.encoder." + name_, name_[7:]]:
            if name not in own_state or own_state[name] is None:
                continue
            if param.shape != own_state[name].shape:
                shape_mismatch = own_state[name].shape, param.shape, name
                continue
            own_state[name].copy_(param)
            found = True
            break
        if found:
            restored_parameter_names.append(name)
        else:
            if shape_mismatch[0] is not None:
                details = " own vs restored: {} vs {}".format(shape_mismatch[0], shape_mismatch[1])
                name = shape_mismatch[2]
            else:
                details = ' not in own model'
                name = name_
            if verbose:
                print("[restore] failure: " + name + details)
    if verbose:
        print('[restore] {} | re-use rate: {:.1f}%'.format(pretrained_model_path, len(restored_parameter_names) * 100. / len(loaded_state_dict.keys())))
    return restored_parameter_names


def freeze_module(m):
    for p in m.parameters(recurse=True):
        p.requires_grad = False


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


if __name__ == '__main__':
    import cv2

    logits = one_hot(np.array([[0, 1], [2, 1]]), 4)
    cv2.imshow("logits class 1", logits[1, :, :])
    cv2.waitKey(0)
