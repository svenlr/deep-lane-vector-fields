import torch

from dataset.io_data_utils import get_coordinate_limits_from_dataset
from loss.lane_mse_loss import MSEAndIndexOrderLoss, visibility_grid_loss
from loss.loss_utils import distance_decaying_loss_weights
from loss.seg_loss import DiceLoss, PixelWeightedCrossEntropyLoss, OhemCELoss, build_auto_aux_loss
from loss.vector_field_loss import direct_xy_attractor_training_loss, direct_sq_attractor_training_loss, indirect_xy_attractor_integration_loss, \
    indirect_main_flow_loss

LOSS_FUNCTIONS = ["seg_ce", "seg_pwce", "seg_dice", "seg_ce_ohem",
                  "lane_mse+idx", "lane_cluster_mse+idx",
                  "lane_attractor", "lane_integration_sampling", "lane_integration_sampling_2it", "lane_integration_sampling_2it_rep",
                  "lane_main_flow", "lane_visibility_grid"]


def build_loss(args, model, dataset, class_weights, use_cuda, for_train=True):
    class_weights = torch.FloatTensor(class_weights)
    dist_weights = None
    if use_cuda:
        class_weights = class_weights.cuda()
    if "seg_dice" in args.loss:
        seg_loss = DiceLoss()
    elif "seg_ce" in args.loss:
        seg_loss = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=args.ignore_class_idx)
        # loss_func = PixelWeightedCrossEntropyLoss(class_weights, pixel_wise_weights=None)
    elif "seg_pwce" in args.loss:
        # x_decay and y_decay are shape parameters for the decay
        dist_weights = distance_decaying_loss_weights(img_height=args.crop_height, img_width=args.crop_width,
                                                      mode="euclid", start_decay_dist=0.2, end_decay_dist=0.6,
                                                      v_decay=1.0, h_decay=0.7)
        dist_weights = torch.FloatTensor(dist_weights)
        if use_cuda:
            dist_weights = dist_weights.cuda()
        seg_loss = PixelWeightedCrossEntropyLoss(class_weights, dist_weights)  # TODO ignore index
    elif 'seg_ce_ohem' in args.loss:
        seg_loss = OhemCELoss(0.7, ignore_lb=args.ignore_class_idx if args.ignore_class_idx is not None else 255)
    else:
        seg_loss = None
    loss_functions = []
    incompatible_losses = []
    if seg_loss is not None:
        if use_cuda:
            seg_loss = seg_loss.cuda()
        if for_train:
            if args.network == "bisenet" and model.module.aux_loss:
                single_loss_func = seg_loss
                seg_loss = lambda out, lbl: single_loss_func(out[0], lbl) + single_loss_func(out[1], lbl) + single_loss_func(out[2], lbl)
            elif args.network == "bisenetv2":
                single_loss_func = seg_loss
                seg_loss = build_auto_aux_loss(single_loss_func, single_loss_func, num=args.bisenetv2_aux_num)
        seg_loss_raw = seg_loss
        loss_functions.append(lambda pred, gt: seg_loss_raw(pred["seg"], gt["seg"]))
    if "lane_mse+idx" in args.loss:
        street_shape_loss = MSEAndIndexOrderLoss(ohem_thresh=args.lane_loss_ohem_thresh if for_train else None)
        if use_cuda:
            street_shape_loss = street_shape_loss.cuda()
        loss_functions.append(street_shape_loss)
    if "lane_cluster_mse+idx" in args.loss:
        street_shape_loss = MSEAndIndexOrderLoss(ohem_thresh=args.lane_loss_ohem_thresh if for_train else None, use_visibility_mask=False,
                                                 prediction_suffix="_cluster")
        if use_cuda:
            street_shape_loss = street_shape_loss.cuda()
        loss_functions.append(street_shape_loss)
    if "lane_attractor" in args.loss:
        if "xy" in args.network:
            loss_functions.append(lambda out, sample: direct_xy_attractor_training_loss(out, sample, get_coordinate_limits_from_dataset(dataset)))
        elif "sq" in args.network:
            loss_functions.append(lambda out, sample: direct_sq_attractor_training_loss(out, sample, get_coordinate_limits_from_dataset(dataset)))
        else:
            incompatible_losses.append("lane_attractor")
    if "lane_integration_sampling" in args.loss:
        loss_functions.append(lambda out, sample: indirect_xy_attractor_integration_loss(out, sample, get_coordinate_limits_from_dataset(dataset),
                                                                                         ohem_thresh=args.lane_loss_ohem_thresh))
    if "lane_integration_sampling_2it" in args.loss:
        loss_functions.append(lambda out, sample: indirect_xy_attractor_integration_loss(out, sample, get_coordinate_limits_from_dataset(dataset),
                                                                                         iterations=2, ohem_thresh=args.lane_loss_ohem_thresh))
    if "lane_integration_sampling_2it_rep" in args.loss:
        loss_functions.append(lambda out, sample: indirect_xy_attractor_integration_loss(out, sample, get_coordinate_limits_from_dataset(dataset),
                                                                                         iterations=2, ohem_thresh=args.lane_loss_ohem_thresh,
                                                                                         only_train_representable=True))
    if "lane_main_flow" in args.loss:
        coord_limits = get_coordinate_limits_from_dataset(dataset)
        loss_functions.append(lambda out, sample: indirect_main_flow_loss(out, sample, coord_limits))
    if "lane_visibility_grid" in args.loss:
        coord_limits = get_coordinate_limits_from_dataset(dataset)
        loss_functions.append(lambda out, sample: visibility_grid_loss(out, sample, coord_limits))
    for loss_name in incompatible_losses:
        print("{} loss incompatible with network {}".format(loss_name, args.network))
    return lambda out, sample: combined_loss(out, sample, loss_functions)


def combined_loss(prediction, label, loss_functions):
    if not isinstance(prediction, dict):
        prediction = {"seg": prediction}
    loss = 0.0
    for loss_func in loss_functions:
        loss += loss_func(prediction, label)
    return loss
