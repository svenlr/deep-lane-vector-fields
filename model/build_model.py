import json
import os

import numpy as np
import torch
import torch.nn
import torch.nn as nn

from dataset.io_data_utils import get_coordinate_limits_from_dataset, init_data_loaders
from model.InferenceWrapper import ScriptTupleInferenceWrapper, ScriptInferenceWrapper, InferenceWrapper, BatchOnlyInferenceWrapper
from nn_utils.local_map_utils import cluster_prototypes_to_same_length


def add_architecture_args(arg_parser):
    arg_parser.add_argument('network', type=str, default="bisenetv2",
                            choices=["bisenet", "bisenetv2", "lane_cluster",
                                     "lane_cluster_fit", "lane_iter", "lane_cluster_fit_gs",
                                     "bisenetv2+lane", "af+df_booster", "lane_attractor_sq",
                                     "proj_attractor_xy", "deep_attractor_xy", "proj_xy_no_ce", "proj_xy_no_ce_no_tanh",
                                     "seg+affinity_field", "seg+affinity_field+proj"],
                            help='Which network architecture to train.')
    arg_parser.add_argument('--street_step_size', type=float, default=0.1, help="Step size between street points in labels and predictions")
    arg_parser.add_argument('--street_length', type=float, default=7, help="Length in meters of predicted street and street labels")
    arg_parser.add_argument('--edge_net_scale', type=float, default=2, help="The s parameter for the edge nets: espnetv2, dicenet")
    arg_parser.add_argument('--espnet_p', type=int, default=3, help="ESPNetV1 p scaling parameter")
    arg_parser.add_argument('--espnet_q', type=int, default=5, help="ESPNetV1 q scaling parameter")
    arg_parser.add_argument('--bisenetv2_head_chn', type=int, default=1024, help="Mid channels in BiSeNetV2 head")
    arg_parser.add_argument('--context_path', type=str, default="resnet18",
                            help='(only for bisenetv1) The context path model you are using, resnet18, resnet101.')
    arg_parser.add_argument('--num_classes', type=int, default=None, help='num of object classes (with void). Default=from dataset')
    arg_parser.add_argument('--crop_height', type=int, default=None, help='Height of cropped/resized input image to network. Default=from dataset')
    arg_parser.add_argument('--crop_width', type=int, default=None, help='Width of cropped/resized input image to network. Default=from dataset')
    arg_parser.add_argument('--tensor_debug', action="store_true", help='visualize certain tensors and save them as images to gain insights')
    arg_parser.add_argument('--multi_img_num', type=int, default=1,
                            help="Use a number > 1 for training and testing with image sequence input instead of single image")
    arg_parser.add_argument('--multi_img_expand', type=int, default=32, help="Expand the network by this number of pixels when using past images as well")
    arg_parser.add_argument('--multi_img_overlap', type=int, default=8, help="Number of overlap pixels for multi image input")
    arg_parser.add_argument('--unfreeze_all', action="store_true", help="for some networks, backbone layers are frozen by default. Use this option to unfreeze")
    arg_parser.add_argument('--freeze_cluster_head', action="store_true", help="for lane_cluster_fit: freeze clustering head when training fitting head")


def build_model(args, img_coordinate_limits=None, data_parallel=True, scriptable=False, inference_wrapper=False, only_inference=False):
    if args.tensor_debug:
        os.makedirs("tensor_debug", exist_ok=True)
    if args.multi_img_num > 1:
        input_extra_height = args.multi_img_expand
        lane_extra_height = args.multi_img_expand
    else:
        input_extra_height = 0
        lane_extra_height = 0
    if img_coordinate_limits is None:
        if hasattr(args, "data") and args.data is not None:
            dataloader_train, dataloader_val, _ = init_data_loaders(args)
            dataset = dataloader_train.dataset if dataloader_train is not None and len(dataloader_train.dataset) > 0 else dataloader_val.dataset
            img_coordinate_limits = get_coordinate_limits_from_dataset(dataset)
        else:
            img_coordinate_limits = [[0, 0], [1, 1]]
    if hasattr(args, "cuda"):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    if not hasattr(args, "freeze_cluster_head"):
        args.freeze_cluster_head = False
    freeze_segment_branch = not args.unfreeze_all
    if args.network == "bisenetv2":
        from model.BiSeNetV2 import BiSeNetV2
        args.bisenetv2_head_chn = args.bisenetv2_head_chn if hasattr(args, "bisenetv2_head_chn") else 1024
        model = BiSeNetV2(args.num_classes, head_mid_channels=args.bisenetv2_head_chn, input_extra_height=input_extra_height,
                          only_inference=only_inference)
    elif args.network == "bisenetv2+lane":
        from model.lane_anchor_based.lane_detection_clustering import BiseNetV2WithLaneScript, BiseNetV2WithLane
        model_cls = BiseNetV2WithLaneScript if scriptable else BiseNetV2WithLane
        model = model_cls((args.crop_height + lane_extra_height, args.crop_width), load_cluster_prototypes(args), img_coordinate_limits,
                          segmentation_classes=args.num_classes, tensor_debug=args.tensor_debug,
                          freeze_segment_branch=freeze_segment_branch)
    elif args.network == "lane_iter":
        from model.lane_detection_iter import IterativeStreetMatchingNet
        model = IterativeStreetMatchingNet((args.crop_height + lane_extra_height, args.crop_width), img_coordinate_limits,
                                           freeze_segment_branch=freeze_segment_branch)
    elif args.network == "lane_cluster":
        from model.lane_anchor_based.lane_detection_clustering import StreetShapeClassifierNet
        model = StreetShapeClassifierNet((args.crop_height + lane_extra_height, args.crop_width), load_cluster_prototypes(args),
                                         img_coordinate_limits, freeze_segment_branch=freeze_segment_branch)
    elif args.network == "lane_cluster_fit":
        from model.lane_anchor_based.lane_detection_clustering import StreetFittingNet
        model = StreetFittingNet((args.crop_height + lane_extra_height, args.crop_width), load_cluster_prototypes(args), img_coordinate_limits,
                                 freeze_segment_branch=freeze_segment_branch, freeze_shape_classifier=args.freeze_cluster_head, fully_conv=True,
                                 tensor_debug=args.tensor_debug)
    elif args.network == "lane_cluster_fit_gs":
        from model.lane_anchor_based.lane_detection_clustering import StreetFittingNet
        model = StreetFittingNet((args.crop_height + lane_extra_height, args.crop_width), load_cluster_prototypes(args), img_coordinate_limits,
                                 freeze_segment_branch=freeze_segment_branch, freeze_shape_classifier=args.freeze_cluster_head, fully_conv=True,
                                 tensor_debug=args.tensor_debug, smooth_fitting_head=False)
    elif args.network == "af+df_booster":
        from model.vector_fields.lane_detection_vf import ProjectedAttractorNet
        model = ProjectedAttractorNet(img_coordinate_limits, freeze_segment_branch=freeze_segment_branch,
                                      direction_field_projection=False)
    elif args.network == "proj_xy_no_ce":
        from model.vector_fields.lane_detection_vf import ProjectedAttractorNet
        model = ProjectedAttractorNet(img_coordinate_limits, freeze_segment_branch=freeze_segment_branch, context_embedding=False)
    elif args.network == "proj_xy_no_ce_no_tanh":
        from model.vector_fields.lane_detection_vf import ProjectedAttractorNet
        model = ProjectedAttractorNet(img_coordinate_limits, freeze_segment_branch=freeze_segment_branch, context_embedding=False,
                                      vf_out_activation=nn.Identity)
    elif args.network == "seg+affinity_field":  # export only (via trace_model.py), since vf_integration is turned off
        from model.vector_fields.lane_detection_vf import ProjectedAttractorNet
        model = ProjectedAttractorNet(img_coordinate_limits, freeze_segment_branch=freeze_segment_branch, context_embedding=False, vf_integration=False,
                                      direction_field_prediction=False, direction_field_projection=False, num_seg_classes=args.num_classes)
    elif args.network == "seg+affinity_field+proj":  # export only (via trace_model.py), since vf_integration is turned off
        from model.vector_fields.lane_detection_vf import ProjectedAttractorNet
        model = ProjectedAttractorNet(img_coordinate_limits, freeze_segment_branch=freeze_segment_branch, context_embedding=False, vf_integration=False,
                                      direction_field_prediction=True, direction_field_projection=True, num_seg_classes=args.num_classes)
    else:
        from model.BiSeNet import BiSeNet
        model = BiSeNet(args.num_classes, args.context_path, slow_up_sampling=args.slow_up_sampling)
    if data_parallel and torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    if inference_wrapper:
        model = create_inference_wrapper(model, args, scriptable)
    return model


def create_inference_wrapper(model, args, scriptable=False, segmentation=True):
    if segmentation:
        if scriptable:
            if args.network in ["bisenetv2+lane", "bisenetv2+feat_s"]:
                model = ScriptTupleInferenceWrapper(model, args.model_width, args.model_height)
            else:
                model = ScriptInferenceWrapper(model, args.model_width, args.model_height)
        else:
            model = InferenceWrapper(model, args.model_width, args.model_height)
    else:
        model = BatchOnlyInferenceWrapper(model, args.model_width, args.model_height)
    return model


def load_cluster_prototypes(args):
    num_points = int(np.round(args.street_length / args.street_step_size))
    cluster_prototypes = json.load(open(os.path.join(args.data, "clusters.json")))["prototypes"]
    return cluster_prototypes_to_same_length(cluster_prototypes, target_num=num_points)
