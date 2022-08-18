import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.loss_utils import distance_decaying_loss_weights
from nn_utils.math_utils import rbf_activations
from nn_utils.seg_metrics import one_hot_torch


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = 1e-5

    @staticmethod
    def flatten(tensor):
        """Flattens a given tensor such that the channel axis is first.
        The shapes are transformed as follows:
           (N, C, D, H, W) -> (C, N * D * H * W)
        """
        C = tensor.size(1)
        # new axis order
        axis_order = (1, 0) + tuple(range(2, tensor.dim()))
        # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
        transposed = tensor.permute(axis_order)
        # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
        return transposed.contiguous().view(C, -1)

    def forward(self, output, target):
        target = one_hot_torch(target, output.shape[1])
        assert output.size() == target.size(), "'input' and 'target' must have the same shape"
        output = F.softmax(output, dim=1)
        output = self.flatten(output)
        target = self.flatten(target)
        # intersect = (output * target).sum(-1).sum() + self.epsilon
        # denominator = ((output + target).sum(-1)).sum() + self.epsilon

        intersect = (output * target).sum(-1)
        denominator = (output + target).sum(-1)
        dice = intersect / denominator
        dice = torch.mean(dice)
        return 1 - dice
        # return 1 - 2. * intersect / denominator


def combine_pixel_wise_and_class_weights(class_weights, pixel_wise_weights=None, width=None, height=None):
    """ create combined (C, H, W) weights from (H, W) location based (pixel-wise) weights and (C) class weights"""
    if pixel_wise_weights is None and (width is None or height is None):
        return class_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)  # (1, cls, 1, 1)
    else:
        if pixel_wise_weights is None:
            pixel_wise_weights = distance_decaying_loss_weights(img_width=width, img_height=height, min_weight=1.0)
            pixel_wise_weights = torch.FloatTensor(pixel_wise_weights)
            pixel_wise_weights = pixel_wise_weights.to(class_weights.device)
        # add logits (classes) dimension to pixel wise weights
        pixel_wise_weights = pixel_wise_weights.unsqueeze(0)
        # expand class weights to match image tensors (repeat for each pixel). afterwards shape: (cls, h, w)
        class_weights = class_weights.unsqueeze(-1).unsqueeze(-1).expand(-1, *pixel_wise_weights.shape[-2:])
        # add batch dimension to both weightings
        class_weights = class_weights.unsqueeze(0)
        pixel_wise_weights = pixel_wise_weights.unsqueeze(0)
        # combine both weightings in one tensor
        return class_weights * pixel_wise_weights


class PixelWeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights, pixel_wise_weights=None):
        super(PixelWeightedCrossEntropyLoss, self).__init__()
        self.weights = combine_pixel_wise_and_class_weights(class_weights, pixel_wise_weights)

    def forward(self, predicted_logits, target):
        # Calculate log probabilities for all classes (logp will have shape [bs, cls, h, w])
        logp = F.log_softmax(predicted_logits)
        # weighting per class and locality combined
        logp = logp * self.weights
        # Gather log probabilities with respect to target (interpret target tensor as indices in the logits tensor)
        logp = logp.gather(1, target.unsqueeze(1))
        # Average over mini-batch and height/width
        # (This is a minimization, thus the negation)
        return - logp.mean()


class PixelWeightedAutoMaxPoolCELoss(nn.Module):
    """ To be used for auxiliary loss heads only.
        The labels are automatically max pooled to the prediction size to allow for multiple classes at the same pixel.
        Idea: this might be useful for learning fine-grained detailed classes
        """

    def __init__(self, class_weights, img_width, img_height, pixel_wise_weights=None):
        super(PixelWeightedAutoMaxPoolCELoss, self).__init__()
        self.weights = combine_pixel_wise_and_class_weights(class_weights, pixel_wise_weights, width=img_width, height=img_height)
        self.normal_pwce = PixelWeightedCrossEntropyLoss(class_weights, pixel_wise_weights)

    def forward(self, predicted_logits, target):
        if target.shape[-2:] == predicted_logits.shape[-2:]:
            return self.normal_pwce(predicted_logits, target)
        else:
            target = one_hot_torch(target, predicted_logits.shape[1])
            # weighting per class and locality combined
            weights = F.adaptive_avg_pool2d(self.weights, output_size=predicted_logits.shape[-2:])
            # bring target logits on lower shape of the predicted logits using max_pool, which produces multiple labels but preserves all classes in image
            target = F.adaptive_max_pool2d(target, output_size=predicted_logits.shape[-2:])
            # calculate num gt logits per pixel for each pixel
            num_labels_per_pixel = torch.sum(target, dim=1, keepdim=True)
            # softmax activation for the prediction
            # why add log(labels_per_pixel)?
            # => achieve loss=0 when all predicted probabilities are equal. Example: log(0.5) + log(0.5) + 2 * log(2) = log(0.5 * 2) + log(0.5 * 2) = 0.
            logp = F.log_softmax(predicted_logits, dim=1) + torch.log(num_labels_per_pixel)
            # divide:   normalize loss so that multi label pixels contribute equally than single label pixels (not used right now)
            # logp = logp / num_labels_per_pixel

            # gather log probabilities based on ground truth (target), also considering weighting
            logp = logp * target * weights
            return - torch.mean(logp)


class OhemCELoss(nn.Module):
    """ Online Hard Example Mining Cross Entropy Loss """
    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        n_min = labels[labels != self.ignore_lb].numel() // 16
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


def build_auto_aux_loss(single_loss_func, aux_loss_func, num=None):
    """ simply combine auxiliary loss heads by mean """
    if num is None:
        num = 10000

    def loss(out, lbl):
        l = single_loss_func(out[0], lbl)
        for i in range(max(0, len(out) - 1 - num), len(out) - 1):
            l += aux_loss_func(out[i + 1], lbl)
        return l / min(num + 1, len(out))

    return loss


def boundary_reduced_loss_weights(start_decay_dist_percentage=0.9, img=None, img_width=None, img_height=None, end_decay_dist_percentage=1.0, min_weight=0.1):
    """ weights that reduce loss at image boundaries to avoid training too much half objects """
    if img is not None:
        img_width = img.shape[1]
        img_height = img.shape[0]
    img_center = np.array([img_height / 2.0, img_width / 2.0])
    indices = np.indices((img_height, img_width))
    indices = np.swapaxes(indices, 0, 2)
    indices = np.swapaxes(indices, 0, 1)
    vectors = indices - img_center
    distances_y = np.abs(vectors[:, :, 0])
    distances_x = np.abs(vectors[:, :, 1])
    start_decay_dist_x = start_decay_dist_percentage * img_width / 2.0
    offset_distances_x = np.clip(distances_x - start_decay_dist_x, 0, img_width / 2.0)
    start_decay_dist_y = start_decay_dist_percentage * img_height / 2.0
    offset_distances_y = np.clip(distances_y - start_decay_dist_y, 0, img_height / 2.0)
    weights_x = np.clip(1 - offset_distances_x / (end_decay_dist_percentage * img_width / 2.0), 0, 1)
    weights_y = np.clip(1 - offset_distances_y / (end_decay_dist_percentage * img_height / 2.0), 0, 1)
    weights = np.minimum(weights_x, weights_y)
    weights = weights * (1 - min_weight) + min_weight
    return weights


if __name__ == '__main__':
    w = distance_decaying_loss_weights(img_width=304, img_height=200, mode="euclid", min_weight=0,
                                       start_decay_dist=0.7, end_decay_dist=0.8)
    a = rbf_activations(w, [0.8, 0.6, 0.4, 0.2], np.repeat(0.12, 4))
    import cv2

    while cv2.waitKey(0) != 27:
        cv2.imshow("test", w)
