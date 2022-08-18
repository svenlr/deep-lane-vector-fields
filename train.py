import argparse
import json
import os
import shutil

import numpy as np
import torch
import torch.jit
import tqdm

from dataset.data_sample_utils import dict_to_cuda
from dataset.io_data_utils import read_categories_from_dataset, get_coordinate_limits_from_dataset, init_data_loaders, smart_parse_args
from dataset.seg_class_weights import enet_class_weighting
from eval import evaluation_with_labels, inference_test_data
from loss.build_loss import build_loss, LOSS_FUNCTIONS
from model.InferenceWrapper import InferenceWrapper
from model.build_model import build_model, add_architecture_args
from model.building_blocks import AutoScaleTanh
from nn_utils.lr_scheduler import WarmupPolyLrScheduler, CyclicThenLinearLR, AdaptiveCyclicLR, OneCycleLR, plot_learning_rate_test_and_exit
from nn_utils.train_utils import load_matching_weights


def train(args, model, optimizer, dataloader_train, dataloader_val, dataloader_test, class_weights):
    if os.path.exists(os.path.join(args.data, "test")) and len(os.listdir(os.path.join(args.data, "test"))) > 0:
        test_input_dir = os.path.join(args.data, "test")
    else:
        test_input_dir = None
    use_cuda = torch.cuda.is_available() and args.use_gpu
    train_statistics = []
    loss_func = build_loss(args, model, dataloader_train.dataset, class_weights, use_cuda)
    if not args.disable_jit_trace:
        inference_wrapper = create_jit_inference_wrapper(dataloader_train, model, use_cuda, num_channels=args.multi_img_num)
    else:
        inference_wrapper = None
    best_val_metric_value = None
    step = 0
    lr_scheduler = create_lr_scheduler(args, optimizer, dataloader_train)
    for epoch in range(args.num_epochs):
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d' % epoch)
        loss_record = []
        lr_record = []
        for i, batch in enumerate(dataloader_train):
            if use_cuda:
                batch = dict_to_cuda(batch)
            lr = np.mean(lr_scheduler.get_lr())
            output = model(batch["img"])
            loss = loss_func(output, batch)
            if loss.item() > 1e10 and args.lr_scheduler == "lr_range_test":
                break
            tq.update(args.batch_size)
            tq.set_postfix(loss='{:05f}'.format(loss.item()), lr='{:.05f}'.format(lr))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            step += 1
            loss_record.append(loss.item())
            lr_record.append(lr)
            if args.lr_scheduler == "adaptive_cyclic":
                lr_scheduler.report_loss(loss.item())
        tq.close()
        if args.lr_scheduler == "lr_range_test":
            plot_learning_rate_test_and_exit(args, loss_record, lr_record)
        loss_train_mean = np.mean(loss_record)
        print('loss for train : %f' % loss_train_mean)
        if epoch % args.validation_step == 0:
            val_loss_func = build_loss(args, model, dataloader_val.dataset, class_weights, use_cuda, for_train=False)
            val_report = evaluation_with_labels(args, model, dataloader_val, os.path.join(args.save_model_path, "val_imgs"), val_loss_func)
            val_metric = args.main_val_metric
            main_val_tag = args.main_val_metric_tag
            val_metric_multiplier = 100 if val_metric in ["seg_miou_mean", "seg_precision"] else 1
            if val_report[main_val_tag]["all"].get(val_metric, -1) == -1:
                print("specified validation metric " + val_metric + " was not calculated. Defaulting to val loss for NN quality evaluation.")
                val_metric = "loss_val_mean"
            print(val_metric + "=" + "{:2.4f}".format(val_report[main_val_tag]["all"][val_metric] * val_metric_multiplier))
            identifier = "{}{:2.4f}".format(val_metric, val_report[main_val_tag]["all"][val_metric] * val_metric_multiplier)
            val_output_imgs_dir = os.path.join(args.save_model_path, "val_imgs_{}".format(identifier))
            if os.path.exists(val_output_imgs_dir):
                shutil.rmtree(val_output_imgs_dir)
            shutil.move(os.path.join(args.save_model_path, "val_imgs"), val_output_imgs_dir)
            shutil.copy(os.path.join(args.data, "categories"), os.path.join(val_output_imgs_dir, "categories"))
            val_report["loss_train_mean"] = float(loss_train_mean)
            val_report["loss_train_std"] = np.std(loss_record).tolist()
            val_report["epoch"] = epoch
            val_report["total_epochs"] = args.num_epochs
            val_report["num_imgs"] = step * args.batch_size
            val_report["lr"] = float(np.mean(lr_scheduler.get_lr()))
            val_report["min_lr"] = float(np.min(lr_record))
            val_report["max_lr"] = float(np.max(lr_record))
            train_statistics.append(val_report)
            statistics_json_str = json.dumps(train_statistics)
            with open(os.path.join(args.save_model_path, "train_statistics.json"), "w+") as f:
                f.write(statistics_json_str)
            val_metric_lower_is_better = "mse" in val_metric.lower() or "loss" in val_metric.lower() or "mse" in val_metric.lower()
            if best_val_metric_value is None:
                best_val_metric_value = val_report[main_val_tag]["all"][val_metric]
                is_better = True
            else:
                if val_metric_lower_is_better:
                    # print(val_metric + ": lower is better")
                    is_better = val_report[main_val_tag]["all"][val_metric] < best_val_metric_value
                else:
                    # print(val_metric + ": greater is better")
                    is_better = val_report[main_val_tag]["all"][val_metric] > best_val_metric_value
            if is_better:
                best_val_metric_value = val_report[main_val_tag]["all"][val_metric]
                save_path = os.path.join(args.save_model_path, 'cls{}{}.pt'.format(len(class_weights), identifier))
                if inference_wrapper is None:
                    if torch.cuda.is_available():
                        torch.save(model.module.state_dict(), save_path)
                    else:
                        torch.save(model.state_dict(), save_path)
                else:  # inference_wrapper is not None
                    if torch.cuda.is_available():
                        inference_wrapper.module = model.module
                    else:
                        inference_wrapper.module = model
                    print("exporting for inference: torch.jit.trace() ...")
                    example_input = torch.mean(dataloader_val.dataset[0]["img"], dim=0)
                    if use_cuda:
                        example_input = example_input.cuda()
                    with torch.jit.optimized_execution(True):
                        traced_model = torch.jit.trace(inference_wrapper, (example_input,))
                    traced_model.save(save_path.replace(".pt", ".jit.pt"))
                if test_input_dir is not None:
                    print("NEW BEST: {}! Running tests...".format(identifier))
                    test_output_dir = os.path.join(args.save_model_path, "test_imgs_{}".format(identifier))
                    os.makedirs(test_output_dir, exist_ok=True)
                    status_file = os.path.join(test_output_dir, "status.json")
                    status = {
                        "num_imgs": step * args.batch_size,
                        "loss_train_mean": loss_train_mean,
                        "epoch": epoch,
                        "lr": float(np.mean(lr_scheduler.get_lr())),
                    }
                    status.update(val_report)
                    json.dump(status, open(status_file, "w"), indent=2)
                    inference_test_data(args, model, dataloader_test, test_output_dir, categories=read_categories_from_dataset(args), use_cuda=use_cuda,
                                        test_step=pow(2, args.test_step_exp))


def create_lr_scheduler(args, optimizer, dataloader_train):
    if args.min_learning_rate is None:
        args.min_learning_rate = args.learning_rate * 0.1
    max_iter = args.num_epochs * len(dataloader_train)
    step_size_up = min(100, len(dataloader_train) // 10)
    step_size_down = len(dataloader_train) - step_size_up
    if args.lr_scheduler == "1cycle":
        lr_scheduler = OneCycleLR(optimizer, args.learning_rate, max_iter, 0.5, initial_warmup_iter=1000)
    elif args.lr_scheduler == "cyclic_decay_auto":
        lr_scheduler = AdaptiveCyclicLR(optimizer, args.min_learning_rate, args.learning_rate, step_size_up, step_size_down,
                                        decay_mode="auto")
    elif args.lr_scheduler == "cyclic_decay_linear":
        lr_scheduler = AdaptiveCyclicLR(optimizer, args.min_learning_rate, args.learning_rate, step_size_up, step_size_down,
                                        decay_mode="linear", max_iter=max_iter)
    elif args.lr_scheduler == "lr_range_test":
        print("\n\n########\n LEARNING_RATE RANGE TEST \n########\n\n")
        iterations = min(args.lr_range_test_it, len(dataloader_train))
        dataloader_train.dataset.image_list = dataloader_train.dataset.image_list[:iterations * dataloader_train.batch_size]
        lr_scheduler = WarmupPolyLrScheduler(optimizer, power=0.9, max_iter=max_iter,
                                             warmup_iter=iterations, warmup_ratio=0.0005, warmup='exp', last_epoch=-1)
    elif args.lr_scheduler in ["poly", "warmup_poly"]:
        # from BiseNetV2 paper
        warmup_iter = 0 if args.lr_scheduler == "poly" else 1000
        # if warmup_iter=0, then this is a simple poly lr scheduler
        lr_scheduler = WarmupPolyLrScheduler(optimizer, power=0.9, max_iter=max_iter,
                                             warmup_iter=warmup_iter, warmup_ratio=0.1, warmup='exp', last_epoch=-1)
    elif args.lr_scheduler == "cyclic_then_linear":
        lr_scheduler = CyclicThenLinearLR(optimizer, base_lr=args.min_learning_rate, max_lr=args.learning_rate, max_iter=max_iter)
    else:
        lr_scheduler = WarmupPolyLrScheduler(optimizer, warmup_iter=0, power=0.9, max_iter=max_iter)
    return lr_scheduler


def create_jit_inference_wrapper(dataloader_val, model, use_cuda, num_channels=1):
    height, width = dataloader_val.dataset[0]["img"].shape[-2], dataloader_val.dataset[0]["img"].shape[-1]
    inference_wrapper = InferenceWrapper(model.module if torch.cuda.is_available() else model, width, height)
    if use_cuda:
        inference_wrapper = inference_wrapper.cuda()
    if num_channels == 1:
        example_input = torch.mean(dataloader_val.dataset[0]["img"], dim=0)
    else:
        example_input = dataloader_val.dataset[0]["img"]
    if use_cuda:
        example_input = example_input.cuda()
    print("testing if torch.jit.trace works...")
    model.eval()
    torch.jit.trace(inference_wrapper, (example_input,))
    print("success: torch.jit.trace works!")
    return inference_wrapper


def main(params=None):
    # basic parameters
    parser = argparse.ArgumentParser()
    add_architecture_args(parser)
    parser.add_argument('--bisenetv2_aux_num', type=int, default=4, help="Number of auxiliary loss heads. Removes the heads from the beginning of NN")
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--test_step_exp', type=int, default=0, help='only run test image # 2^test_step_exp. -1 to disable test runs')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
    parser.add_argument('--eval_batch_size', type=int, default=1, help='Number of images in each batch when running validation and tests')
    parser.add_argument('--data', type=str, default=None, help='path of training data')
    parser.add_argument('--num_workers', type=int, default=12, help='num of workers')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, nargs="+", default=None, help='Loads all matching weights from all these checkpoints')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--overwrite_log', action="store_true", help="allow reusing and overwriting if save_model_path already exists")
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer, support rmsprop, sgd, adam, adamw')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate used for train')
    parser.add_argument('--lr_scheduler', type=str, default='warmup_poly', help='learning rate scheduler used for train',
                        choices=["lr_range_test", "1cycle", "warmup_poly", "poly", "cyclic_then_linear", "cyclic_decay_auto", "cyclic_decay_linear"])
    parser.add_argument('--min_learning_rate', type=float, default=None, help='minimum learning rate when using cyclic learning rate')
    parser.add_argument('--lr_range_test_it', type=int, default=100, help='how many mini batches to test in range test')
    parser.add_argument('--lr_range_test_max', type=int, default=0.5, help='maximum learning rate for range test')
    parser.add_argument('--grad_momentum', type=float, default=0.9, help='gradient momentum for SGD and RMSProp')
    parser.add_argument('--min_grad_momentum', type=float, default=0.8, help='min gradient momentum when cycling momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay used for train')
    parser.add_argument('--loss', nargs="+", type=str, default='crossentropy', help='select one or multiple loss functions to use for training',
                        choices=LOSS_FUNCTIONS)
    parser.add_argument('--main_val_metric', type=str, default="seg_miou_mean",
                        choices=["loss_val_mean", "lane_mse_mean", "seg_miou_mean", "lane_f1_mean"],
                        help="main metric to decide when to save a checkpoint (also used for early stopping)")
    parser.add_argument('--main_val_metric_tag', type=str, default="all",
                        help="dataset tag for main metric. For example, use 'real' to evaluate only real data for validation")
    parser.add_argument('--val_dataset_tags', type=str, default=["real", "syn"], nargs="+",
                        help="dataset tags that you want individual validation scores for. By default, scores are reported for real, syn")
    parser.add_argument('--lane_loss_ohem_thresh', type=float, default=0.02 ** 2, help="online hard example mining threshold for lane detection loss")
    parser.add_argument('--aux_loss_booster', action="store_true", default=False, help='use auxiliary loss heads')
    parser.add_argument('--slow_up_sampling', action="store_true", default=False, help='[bisenet] up sample all at once or slowly with multiple layers')
    parser.add_argument('--no_normalize', action="store_true", default=False, help='turn off image net normalization')
    parser.add_argument('--ignore_class', default=None, help='no loss class')
    parser.add_argument('--custom_weights', default=[], nargs="+",
                        help="specify weights individually. Format --custom_weights <class1>=<weight1> <class2>=<weight2>")
    parser.add_argument("--augmentation", default=["{}={}".format(k, json.dumps(v)) for k, v in json.load(open("dataset/default_augmentation.json")).items()],
                        nargs="+", help="augmentation settings. Format <key1>=<value1> <key2>=<value2>")
    parser.add_argument("--disable_augmentation", action="store_true", help="disable data augmentation")
    parser.add_argument('--disable_jit_trace', default=False, action="store_true",
                        help="use this flag to disable calling torch.jit.trace(model) for your model. Useful if model is not traceable")

    args = smart_parse_args(parser)

    dataloader_train, dataloader_val, dataloader_test = init_data_loaders(args)
    categories = read_categories_from_dataset(args)
    args.ignore_class_idx = categories.index(args.ignore_class) if args.ignore_class is not None else None

    # build model
    model = build_model(args, img_coordinate_limits=get_coordinate_limits_from_dataset(dataloader_train.dataset))
    dataloader_train, dataloader_val, dataloader_test = adapt_data_loaders_to_model_type(args, model, dataloader_train)

    # load pretrained model if exists
    if args.pretrained_model_path is not None and len(args.pretrained_model_path) > 0:
        load_matching_weights(model, args.pretrained_model_path)

    if args.lr_scheduler == "lr_range_test":
        args.learning_rate = args.lr_range_test_max
    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(get_optimizer_params(model, args), args.learning_rate, momentum=args.grad_momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(get_optimizer_params(model, args), args.learning_rate, momentum=args.grad_momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(get_optimizer_params(model, args), args.learning_rate, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(get_optimizer_params(model, args), args.learning_rate, weight_decay=args.weight_decay)
    else:  # rmsprop
        print('not supported optimizer')
        print(args.optimizer)
        return None

    train_label_path = os.path.join(args.data, 'train_labels')
    label_files = os.listdir(train_label_path)
    label_paths = [os.path.join(train_label_path, l_file) for l_file in label_files if l_file.endswith(".png")]
    class_weights = enet_class_weighting(label_paths, num_classes=args.num_classes)

    for cw in args.custom_weights:
        c, w = cw.split("=")
        class_weights[categories.index(c)] = float(w)

    if args.ignore_class_idx is not None:
        class_weights[args.ignore_class_idx] = 0

    print("============ Classes and weights =============")
    for i, c in enumerate(categories):
        print("{:2d}: {} = {:5f}{}".format(i, c, class_weights[i], " (ignore)" if i == args.ignore_class_idx else ""))

    os.makedirs(args.save_model_path, exist_ok=True)
    saved_files = set(os.listdir(args.save_model_path))
    if "train_statistics.json" in saved_files and not args.overwrite_log:
        print("\nAborting...\nDirectory {} has already been used as save_model_path".format(args.save_model_path))
        return
    json.dump(args.__dict__, open(os.path.join(args.save_model_path, "hyper_params.json"), "w+"), indent=2)
    # train
    # with torch.autograd.set_detect_anomaly(True):
    train(args, model, optimizer, dataloader_train, dataloader_val, dataloader_test, class_weights)

    # val(args, model, dataloader_val, csv_path)


def adapt_data_loaders_to_model_type(args, model, dataloader_train):
    model = model.eval()
    predict = model(dataloader_train.dataset[0]["img"].unsqueeze(0))
    require_seg_labels = require_lane_labels = False
    if isinstance(predict, torch.Tensor) or "seg" in predict:
        require_seg_labels = True
    if isinstance(predict, dict) and "local_map_rl" in predict:
        require_lane_labels = True
    dataloader_train, dataloader_val, dataloader_test = init_data_loaders(args, require_lane_labels=require_lane_labels, require_seg_labels=require_seg_labels)
    return dataloader_train, dataloader_val, dataloader_test


def get_optimizer_params(model, args, force_weight_decay_params=None):
    """ get params list to feat to optimizers with weight decay deactivated for PReLU layers """
    if force_weight_decay_params is None:
        force_weight_decay_params = []
    wd_params = []
    non_wd_params = []
    reduce_lr_params = []
    for param_name, param in model.named_parameters():
        use_weight_decay_for_param = True
        if ("act" in param_name or param.dim() == 1) and param_name not in force_weight_decay_params:
            module_name = param_name.replace(".weight", "").replace(".bias", "")
            module = model
            for module_path_element in module_name.split("."):
                if module_path_element.isnumeric():
                    module = module[int(module_path_element)]
                else:
                    module = getattr(module, module_path_element)
            if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.PReLU) or isinstance(module, AutoScaleTanh):
                use_weight_decay_for_param = False
        if "module.step" not in param_name:
            if use_weight_decay_for_param:
                wd_params.append(param)
            else:
                non_wd_params.append(param)
        else:
            print(param_name)
    return [
        {"params": wd_params},
        {"params": non_wd_params, "weight_decay": 0},
    ]


if __name__ == '__main__':
    main()
