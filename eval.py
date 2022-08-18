import argparse
import json
import os
import shutil
import threading

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset.data_sample_utils import dict_to_cuda
from dataset.io_data_utils import smart_parse_args, init_data_loaders
from dataset.io_data_utils import write_categories
from model.build_model import build_model
from nn_utils.lane_metrics import lane_mse, lane_f1
from nn_utils.local_map_utils import extract_local_map_json_from_predict_dict
from nn_utils.seg_metrics import compute_precision_torch, fast_hist_torch, per_class_iu
from nn_utils.train_utils import load_matching_weights


def inverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def evaluation_with_labels(args, model, dataloader, val_imgs_save_path, val_loss_func=None):
    val_imgs_save_path = os.path.expanduser(val_imgs_save_path)
    if not os.path.exists(val_imgs_save_path):
        os.makedirs(val_imgs_save_path, exist_ok=True)
    # label_info = get_label_info(csv_path)
    cut_bottom = -args.multi_img_expand if args.multi_img_num > 1 else None
    if not args.no_normalize:
        def un_normalize_transform(x):
            return inverse_normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        def un_normalize_transform(x):
            return x
    # dataloader.dataset.seed_augmentation(42)
    writer = AsyncImgWriter()
    stats = MetricStats(args.val_dataset_tags, args.ignore_class_idx)
    with torch.no_grad():
        use_cuda = torch.cuda.is_available() and args.use_gpu
        model.eval()
        tq = tqdm(total=len(dataloader) * dataloader.batch_size)
        tq.set_description('validation')
        for i, sample in enumerate(dataloader):
            if use_cuda:
                sample = dict_to_cuda(sample)

            # get RGB predict image
            predict = model(sample["img"])
            predict = predict[0] if isinstance(predict, tuple) else predict
            predict_seg = predict.get("seg", None) if isinstance(predict, dict) else predict
            if val_loss_func is not None and (predict_seg is not None) == ("seg" in sample):
                stats.add(sample, "loss_val", val_loss_func(predict, sample).item())
            if isinstance(predict, dict) and "local_map_rl" in predict.keys():
                stats.add(sample, "lane_mse", lane_mse(predict["local_map_rl"], sample["local_map"]["right_lane"]["left_marking"],
                                                       sample["local_map"]["visibility_mask"]))
                if "visibility_mask" in predict.keys():
                    l_f1, l_precision, l_recall = lane_f1(predict["local_map_rl"], predict["visibility_mask"], sample)
                    stats.add(sample, "lane_f1", l_f1)
                    stats.add(sample, "lane_precision", l_precision)
                    stats.add(sample, "lane_recall", l_recall)

            if predict_seg is not None and "seg" in sample:
                predict_seg = torch.argmax(predict_seg, dim=1)  # predict = reverse_one_hot(predict)
                if len(sample["seg"].shape) == len(predict_seg.shape) and sample["seg"].shape[-2] == args.num_classes:
                    sample["seg"] = torch.argmax(sample["seg"], dim=1)  # label = reverse_one_hot(label)
                precision = compute_precision_torch(predict_seg, sample["seg"])
                stats.add(sample, "seg_miou", fast_hist_torch(sample["seg"], predict_seg, args.num_classes, args.ignore_class_idx))
                stats.add(sample, "seg_precision", precision)
                predict_seg = np.array(predict_seg.cpu(), dtype=np.uint8)

            for j in range(sample["img"].shape[0]):
                img_base_name = sample["identifier"][j]
                writer.append(os.path.join(val_imgs_save_path, img_base_name + ".png"),
                              (un_normalize_transform(sample["img"][j, :, :cut_bottom, :])[0, ...] * 255).cpu().numpy())
                if predict_seg is not None:
                    writer.append(os.path.join(val_imgs_save_path, img_base_name + "_labels.png"), predict_seg[j, ...])
                if isinstance(predict, dict) and ("local_map_rl" in predict.keys() or "local_map_rr" in predict.keys()):
                    writer.append(os.path.join(val_imgs_save_path, img_base_name + "_local_map.json"),
                                  extract_local_map_json_from_predict_dict(predict, sample, batch_idx=j, stats=stats))
                if isinstance(predict, dict) and "lane_attractor" in predict:
                    write_attractor_output(writer, val_imgs_save_path, predict, img_base_name, j)
                    if not os.path.exists(os.path.join(val_imgs_save_path, img_base_name + "_local_map.json")):
                        writer.append(os.path.join(val_imgs_save_path, img_base_name + "_local_map.json"),
                                      {"transform": {"pixels_per_meter": 50, "car_to_image_offset": 0.1}})
            tq.update(sample["img"].shape[0])
    writer.stop_and_join()
    tq.close()
    return stats.summarize()


class MetricStats:
    def __init__(self, dataset_tags, seg_ignore_idx):
        self.dataset_tags = dataset_tags + ["all"]
        self.data = dict((tag, {"all": {}}) for tag in self.dataset_tags)
        self.seg_ignore_idx = seg_ignore_idx
        self.metrics_per_sample = {}

    def add(self, sample, metric, value):
        for i in range(sample["img"].shape[0]):
            if isinstance(value, torch.Tensor):
                assert value.shape[0] == sample["img"].shape[0]  # ensure that the first dimension is the batch dimension
                single_value = value[i:i + 1, ...]  # take only the current batch
            else:
                single_value = value  # fallback to same value for the whole batch
            if sample["identifier"][i] not in self.metrics_per_sample:
                self.metrics_per_sample[sample["identifier"][i]] = {}
            if isinstance(single_value, torch.Tensor) and single_value.numel() > 1:
                self.metrics_per_sample[sample["identifier"][i]][metric] = single_value.cpu().numpy().tolist()
            else:
                self.metrics_per_sample[sample["identifier"][i]][metric] = float(single_value)
            sample_tags = set(sample["tags"][i].strip().split(" "))
            for dataset_tag in self.dataset_tags:
                if dataset_tag == "all" or dataset_tag in sample_tags:
                    self.data[dataset_tag]["all"][metric] = self.data[dataset_tag]["all"].get(metric, []) + [single_value]
                    for tag in sample_tags:
                        # track metric per tag
                        if tag in self.dataset_tags:
                            # skip dataset tags
                            continue
                        self.data[dataset_tag][tag] = self.data[dataset_tag].get(tag, {})
                        self.data[dataset_tag][tag][metric] = self.data[dataset_tag][tag].get(metric, []) + [single_value]

    def get(self, sample_identifier):
        return self.metrics_per_sample.get(sample_identifier, {})

    def std_and_mean(self, metric_name, values):
        ret = {}
        if len(values) > 0:
            if metric_name == "seg_miou":
                hist = values[0].to(torch.float32)
                for v in values[1:]:
                    hist += v
                miou_list = per_class_iu(hist.squeeze(0).cpu().numpy())
                ret["seg_miou_per_class"] = miou_list.tolist()
                ret["seg_miou_mean"] = float(np.mean(miou_list[np.arange(0, len(miou_list)) != self.seg_ignore_idx]))
                ret["seg_miou_std"] = float(np.std(miou_list[np.arange(0, len(miou_list)) != self.seg_ignore_idx]))
            else:
                if isinstance(values[0], torch.Tensor):
                    ret[metric_name + "_mean"] = float(torch.tensor(values).mean().cpu().numpy())
                    ret[metric_name + "_median"] = float(torch.tensor(values).median().cpu().numpy())
                    ret[metric_name + "_std"] = float(torch.tensor(values).std().cpu().numpy())
                else:
                    ret[metric_name + "_mean"] = float(np.array(values).mean())
                    ret[metric_name + "_median"] = float(np.median(values))
                    ret[metric_name + "_std"] = float(np.array(values).std())
        return ret

    def _internal_summarize(self, data_dict, summarizer):
        ret = {}
        for key in data_dict:
            if isinstance(data_dict[key], dict):
                ret[key] = self._internal_summarize(data_dict[key], summarizer)
            elif isinstance(data_dict[key], list):
                ret.update(summarizer(key, data_dict[key]))
        return ret

    def summarize(self, summarizer=None):
        if summarizer is None:
            summarizer = self.std_and_mean
        return self._internal_summarize(self.data, summarizer)


def inference_test_data(args, model, dataloader_test, save_path, categories=None, use_cuda=True, test_step=1, normalize=True):
    save_path = os.path.expanduser(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    if categories is not None:
        write_categories(os.path.join(save_path, "categories"), categories)
    cut_bottom = -args.multi_img_expand if args.multi_img_num > 1 else None
    writer = AsyncImgWriter()
    if normalize:
        def un_normalize_transform(x):
            return inverse_normalize(x, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    else:
        def un_normalize_transform(x):
            return x
    with torch.no_grad():
        model.eval()
        img_counter = 0
        test_image_list_backup = dataloader_test.dataset.image_list
        dataloader_test.dataset.image_list = dataloader_test.dataset.image_list[::test_step]
        tq = tqdm(desc="running test data", total=len(dataloader_test.dataset.image_list))
        for i, sample in enumerate(dataloader_test):
            data = sample["img"]
            if use_cuda:
                predict = model(data.cuda())
            else:
                predict = model(data)
            predict = predict[0] if isinstance(predict, tuple) else predict
            predict_seg = predict.get("seg", None) if isinstance(predict, dict) else predict
            if predict_seg is not None:
                predict_seg = torch.argmax(predict_seg, dim=1)
                predict_seg = np.array(predict_seg.cpu(), dtype=np.uint8)
            for j in range(sample["img"].shape[0]):
                img_base_name = sample["identifier"][j]
                if predict_seg is not None:
                    writer.append(os.path.join(save_path, img_base_name + "_labels.png"), predict_seg[j, ...])
                writer.append(os.path.join(save_path, img_base_name + ".png"), (un_normalize_transform(data[j, :, :cut_bottom, :])[0, ...] * 255).cpu().numpy())
                if isinstance(predict, dict) and ("local_map_rl" in predict.keys() or "local_map_rr" in predict.keys()):
                    writer.append(os.path.join(save_path, img_base_name + "_local_map.json"),
                                  extract_local_map_json_from_predict_dict(predict, sample, batch_idx=j))
                if isinstance(predict, dict) and "lane_attractor" in predict:
                    write_attractor_output(writer, save_path, predict, img_base_name, j)
                    if not os.path.exists(os.path.join(save_path, img_base_name + "_local_map.json")):
                        writer.append(os.path.join(save_path, img_base_name + "_local_map.json"),
                                      {"transform": {"pixels_per_meter": 50, "car_to_image_offset": 0.1}})
                img_counter += test_step
            tq.update(sample["img"].shape[0])
        tq.close()
        dataloader_test.dataset.image_list = test_image_list_backup
    writer.stop_and_join()


class AsyncImgWriter:
    def __init__(self, num_threads=24):
        self.running = True
        self._lock = threading.Lock()
        self._queue = []

        self.threads = [threading.Thread(target=self._background_thread_main) for i in range(num_threads)]
        self.new_img_cond = threading.Condition(self._lock)
        self.img_dequeue_cond = threading.Condition(self._lock)
        for t in self.threads:
            t.start()

    def stop_and_join(self):
        self.running = False
        for t in self.threads:
            t.join()

    def append(self, path, img):
        with self._lock:
            self._queue.append((path, img))
            self.new_img_cond.notify()
        with self._lock:
            if len(self._queue) >= 100:
                self.img_dequeue_cond.wait(timeout=1)

    def _background_thread_main(self):
        while self.running:
            with self._lock:
                if len(self._queue) > 0:
                    item = self._queue[0]
                    self._queue = self._queue[1:]
                    self.img_dequeue_cond.notify()
                else:
                    item = None
            if item is None:
                with self._lock:
                    self.new_img_cond.wait(timeout=0.1)
            else:
                if isinstance(item[1], str):
                    shutil.copy(item[1], item[0])
                if isinstance(item[1], dict):
                    with open(item[0], "w+") as f:
                        json.dump(item[1], f)
                else:
                    try:
                        cv2.imwrite(item[0], item[1], [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    except:
                        pass


def write_attractor_output(writer, save_path, predict, img_base_name, j):
    attractor = predict["lane_attractor"][j]
    if "visibility_grid" in predict:
        writer.append(os.path.join(save_path, img_base_name + "_visibility_grid.png"), predict["visibility_grid"][j, 0, ...].cpu().numpy() * 255)
    if "lane_attractor_no_proj" in predict:
        attractor_np = predict["lane_attractor_no_proj"][j]
        writer.append(os.path.join(save_path, img_base_name + "_x_attractor_np.png"), (attractor_np[0, ...].cpu().numpy() + 1) * 127)
        writer.append(os.path.join(save_path, img_base_name + "_y_attractor_np.png"), (attractor_np[1, ...].cpu().numpy() + 1) * 127)
    if "main_flow" in predict:
        main_flow = predict["main_flow"][j]
        writer.append(os.path.join(save_path, img_base_name + "_x_main_flow.png"), (main_flow[0, ...].cpu().numpy() + 1) * 127)
        writer.append(os.path.join(save_path, img_base_name + "_y_main_flow.png"), (main_flow[1, ...].cpu().numpy() + 1) * 127)
    if attractor.shape[0] == 2:
        writer.append(os.path.join(save_path, img_base_name + "_x_attractor.png"), (attractor[0, ...].cpu().numpy() + 1) * 127)
        writer.append(os.path.join(save_path, img_base_name + "_y_attractor.png"), (attractor[1, ...].cpu().numpy() + 1) * 127)
    if attractor.shape[0] > 2:
        writer.append(os.path.join(save_path, img_base_name + "_x_attractor.png"), (attractor[0::2, ...].mean(dim=0).cpu().numpy() + 1) * 127)
        writer.append(os.path.join(save_path, img_base_name + "_y_attractor.png"), (attractor[1::2, ...].mean(dim=0).cpu().numpy() + 1) * 127)
    if attractor.shape[0] == 1:
        writer.append(os.path.join(save_path, img_base_name + "_sq_attractor.png"), (attractor[0, ...].cpu().numpy() * 255).clip(0, 255))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--street_length', type=float, default=7, help="Length in meters of predicted street and street labels")
    parser.add_argument('--log_dir', type=str, default=None, help='path to training log that you want to evaluate')
    parser.add_argument('--data', type=str, default=None, help='path to dataset with evaluation data')
    parser.add_argument('--hyper_params', type=str, default=None, help='path to hyper params')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to weights to load')
    parser.add_argument('--cuda', type=str, default='0', help="visible cuda devices")
    parser.add_argument('--qualitative_test', action="store_true", help="specify this if your test set has no labels")
    parser.add_argument('--save_model_path', type=str, help="where to save results")
    args = parser.parse_args()
    if args.log_dir is not None:
        probe_path = os.path.join(args.log_dir, "hyper_params.json")
        if args.hyper_params is None and os.path.exists(probe_path):
            args.hyper_params = probe_path
    if args.hyper_params is not None:
        loaded_args = json.load(open(args.hyper_params))
        loaded_args.update(args.__dict__)
        args.__dict__.update(loaded_args)
    if args.pretrained_model_path is not None:
        try:
            model = torch.jit.load(args.pretrained_model_path)
            if hasattr(model, "module"):
                model = model.module
        except RuntimeError:
            model = None
    else:
        model = None
    if args.save_model_path is None:
        args.save_model_path = args.log_dir
    args = smart_parse_args(parser, args=args)
    dataloader_train, dataloader_val, dataloader_test = init_data_loaders(args, quantitative_test_data=not args.qualitative_test)

    # build model
    if model is None:
        print("Building model instead of loading with JIT")
        model = build_model(args)
        if args.pretrained_model_path is not None and len(args.pretrained_model_path) > 0:
            load_matching_weights(model, args.pretrained_model_path)
        elif args.log_dir is None:
            print("NO WEIGHTS GIVEN?!")

    val_eval_finished = False
    if args.log_dir is not None and args.pretrained_model_path is None:
        if args.pretrained_model_path is None:
            files = [os.path.join(args.log_dir, f) for f in os.listdir(args.log_dir) if ".pt" == f[-3:]]
            files.sort()
            best_weights = files[-1]
            best_score = 0
            for f in files[-5:]:
                args.pretrained_model_path = f
                load_matching_weights(model, args.pretrained_model_path, verbose=False)
                val_report = evaluation_with_labels(args, model, dataloader_val, os.path.join(args.save_model_path, "eval_val_imgs"))
                if val_report["real"]["all"]["lane_f1_mean"] > best_score:
                    best_score = val_report["real"]["all"]["lane_f1_mean"]
                    json.dump(val_report, open(os.path.join(args.save_model_path, "val_report.json"), "w+"))
                    best_weights = f
                    val_eval_finished = True
            load_matching_weights(model, best_weights)
    if not val_eval_finished:
        val_report = evaluation_with_labels(args, model, dataloader_val, os.path.join(args.save_model_path, "eval_val_imgs"))
        json.dump(val_report, open(os.path.join(args.save_model_path, "val_report.json"), "w+"))
    if args.qualitative_test:
        inference_test_data(args, model, dataloader_test, args.save_model_path)
    else:
        test_report = evaluation_with_labels(args, model, dataloader_test, os.path.join(args.save_model_path, "eval_test_imgs"))
        json.dump(test_report, open(os.path.join(args.save_model_path, "test_report.json"), "w+"))


if __name__ == '__main__':
    main()
