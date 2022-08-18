#!/usr/bin/env python3


import argparse
import json
import os
import random

import numpy as np

from model.lane_anchor_based.street_clustering_algo import clustering_street_labels


def collect_data(args):
    samples = []
    for folder in args.data:
        files = os.listdir(os.path.join(folder, "train"))
        for f in files:
            local_map_path = os.path.join(folder, "train_labels", f.replace(".png", ".json"))
            if not os.path.exists(local_map_path):
                continue
            samples.append({
                "img_path": os.path.join(folder, "train", f),
                "local_map_path": local_map_path,
            })
    return samples


def main():
    parser = argparse.ArgumentParser(description="Attempt to create a minimal clustering that represents all lane detection labels")
    parser.add_argument("data", type=str, nargs="+", default="", help="Path to the directory of the dataset.")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--max-num-samples", type=int, default=5000,
                        help="Random draw of this num of samples from full dataset in order to reduce run time")
    parser.add_argument("--piece-wise", action="store_true",
                        help="Enable piece-wise clustering.")
    parser.add_argument("--piece-length", type=float, default=1.6,
                        help="Use piece-wise clustering at the given length.")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Distance threshold between a sample and the cluster prototype.")
    parser.add_argument("--degree-threshold", type=float, default=45,
                        help="Degree threshold between a sample and the cluster prototype.")
    args = parser.parse_args()
    setattr(args, "sub_dirs", [])

    samples = collect_data(args)
    samples = random.sample(samples, min(args.max_num_samples, len(samples)))

    if args.piece_wise:
        cluster_json_path = os.path.join(args.data[0], "clusters_piece_wise.json")
    else:
        cluster_json_path = os.path.join(args.data[0], "clusters.json")
    if os.path.exists(cluster_json_path):
        with open(cluster_json_path, "r") as f:
            cluster_json = json.loads(f.read())
        initial_prototypes = [np.array(prototype) for prototype in cluster_json["prototypes"]]
    else:
        initial_prototypes = None
    piece_length = args.piece_length if args.piece_wise else None
    clustering_street_labels(samples, cluster_json_path, initial_prototypes=initial_prototypes, visualize=args.visualize,
                             piece_length=piece_length, threshold=args.threshold, rad_threshold=np.deg2rad(args.degree_threshold))


if __name__ == '__main__':
    main()
