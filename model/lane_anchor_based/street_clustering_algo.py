import json
import random
import time

import cv2
import joblib
import numpy as np
from tqdm import tqdm

from dataset.LocalMapSample import LocalMapSample
from nn_utils.geometry_utils import calc_pairwise_distances, transform_to_ipm, calc_similarities, calc_similarity, calc_alignment_indices, \
    calc_max_aligned_distance, calc_relative_angles


def clustering_street_labels(samples, cluster_json_dst_path, num_iter=500, initial_prototypes=None, visualize=False, piece_length=None, threshold=0.7,
                             rad_threshold=np.pi / 4):
    # type: (list, str, int, list, bool, float, float, float) -> None
    piece_wise = piece_length is not None
    img = cv2.imread(samples[0]["img_path"])
    json_data = json.load(open(samples[0]["local_map_path"]))
    clusters = []
    similarity_cache = SimilarityCache(threshold, rad_threshold, max_size=20e6)
    samples = smart_multi_core_exe(read_data_single_core, samples, (img.shape, piece_length), desc="reading data")
    if initial_prototypes is None:
        random.shuffle(samples)
        random_init_samples = samples[:500]
        cluster_unassigned_samples(random_init_samples, clusters, similarity_cache)
        cluster_num_history = [len(samples)]
        best_prototypes = samples
    else:
        prototypes = []
        for prototype_points in tqdm(initial_prototypes, desc="restore clustering"):
            min_dist = np.inf
            prototype = None
            for sample in samples:
                pw_distances = calc_pairwise_distances(prototype_points, sample.points)
                mean_distance = np.mean(np.min(pw_distances, axis=1))
                if mean_distance < min_dist:
                    prototype = sample
                    min_dist = mean_distance
            prototypes.append(prototype)
        clusters = [Cluster(prototype, similarity_cache) for prototype in prototypes]
        cluster_num_history = [len(clusters)]
        best_prototypes = prototypes
    tq = tqdm(desc="clustering", total=num_iter)
    for i in range(num_iter):
        show_status(tq, clusters, best_prototypes, status="removing conflicting clusters")
        clusters = remove_conflicting_clusters(clusters)
        show_status(tq, clusters, best_prototypes, status="updating prototypes")
        for cluster in clusters:
            # if cluster.has_changed():
            cluster.select_best_prototype()
        show_status(tq, clusters, best_prototypes, status="assigning samples to clusters")
        unassigned = assign_samples_to_clusters(samples, clusters, similarity_cache)
        random.shuffle(unassigned)
        show_status(tq, clusters, best_prototypes, status="clustering unassigned samples")
        cluster_unassigned_samples(unassigned, clusters, similarity_cache)
        show_status(tq, clusters, best_prototypes, status="removing redundant clusters")
        clusters = remove_redundant_clusters(clusters)
        # assert_feasible_clusters(clusters)
        if len(clusters) <= np.min(cluster_num_history):
            if len(clusters) < np.min(cluster_num_history):
                show_status(tq, clusters, best_prototypes, status="saving best clustering")
                save_clustering(cluster_json_dst_path, clusters)
            best_prototypes = [c.prototype for c in clusters]
        cluster_num_history.append(len(clusters))
        if visualize:
            debug_img = draw_clustering(best_prototypes, img, json_data, piece_wise=piece_wise)
            cv2.imshow("debug_img", debug_img)
            cv2.waitKey(10)
        tq.update(1)
        similarity_cache.shrink_to_max_size()
    clusters = [Cluster(prototype, similarity_cache) for prototype in best_prototypes]
    assign_samples_to_clusters(samples, clusters, similarity_cache)
    save_clustering(cluster_json_dst_path, clusters)
    tq.close()


def save_clustering(cluster_json_dst_path, clusters):
    with open(cluster_json_dst_path, "w+") as f:
        f.write(json.dumps({
            "num_clusters": len(clusters),
            "prototypes": [c.prototype.points.tolist() for c in clusters]
        }))


def draw_clustering(best_clustering, img, json_data, piece_wise=False):
    debug_img = np.zeros(shape=img.shape[:2], dtype=np.uint8)
    for prototype in best_clustering:
        if hasattr(prototype, "points"):
            prototype = prototype.points
        points_ipm = transform_to_ipm(prototype, img.shape[1], img.shape[0], json_data)
        if piece_wise:
            points_ipm[..., 1] -= img.shape[0] / 2
        draw_lane_points_on_img(debug_img, points_ipm, (0, 255, 0))
    return debug_img


def draw_lane_points_on_img(img, points, color, visibility_mask=None, thickness=1):
    last_p = None
    if visibility_mask is not None:
        visibility_mask = ((np.array(visibility_mask) > 0.5).astype(np.float32) * 0.8 + 0.2).tolist()
    for i, p in enumerate(points):
        # if img.shape[0] > p[0] >= 0 and img.shape[1] > p[1] >= 0:
        p = np.array(p).astype(np.int32)
        if last_p is not None:
            if visibility_mask is not None:
                c = (color[0] * visibility_mask[i], color[1] * visibility_mask[i], color[2] * visibility_mask[i])
            else:
                c = color
            if img.ndim == 2:
                c = float(np.mean(color))
            cv2.line(img, tuple(p), tuple(last_p), c, thickness)
            # cv2.circle(img, tuple(p), 3, c, thickness=1)
        last_p = p


def assign_samples_to_clusters(samples, clusters, similarity_cache):
    comparisons = []
    for cluster in clusters:
        comparisons += [(cluster, sample) for sample in samples]
    comparison_samples = [(cluster.prototype, sample) for cluster, sample in comparisons]
    similarities = similarity_cache.similarities(comparison_samples)
    for (cluster, sample), similarity in zip(comparisons, similarities):
        if similarity == np.inf:
            cluster.remove(sample)
        else:
            cluster.add(sample)
    unassigned = [s for s in samples if len(s.clusters) == 0]
    return unassigned


def show_status(tq, clusters, best_clustering, status=""):
    tq.set_postfix_str("{} clusters (best feasible: {}) | {} | cache {}".format(len(clusters), len(best_clustering), status, len(clusters[0]._cache._cache)))


def assert_feasible_clusters(clusters):
    for cluster in clusters:
        for sample in cluster.samples:
            assert cluster.calc_similarity(sample) != np.inf


class Cluster:
    def __init__(self, prototype, similarity_cache):
        """
        :param prototype: cluster prototype. Everything that is similar enough to this prototype is considered part of the cluster.
        :type prototype: segmentation_data_generation.data_utils.LocalMapSample.LocalMapSample
        """
        self.prototype = prototype
        self.prototype.clusters.add(self)
        self.samples = {self.prototype}
        self._changes_since_proto_update = 0
        self._size_at_proto_update = 0
        self._cache = similarity_cache

    def __len__(self):
        return len(self.samples)

    @property
    def destroyed(self):
        return self.prototype is None

    def destroy(self):
        # samples = list(self.samples)
        for sample in self.samples:
            sample.clusters.remove(self)
        self.samples = {}
        self.prototype = None

    def add(self, sample):
        assert not self.destroyed
        if sample in self.samples:
            return
        self._changes_since_proto_update += 1
        self.samples.add(sample)
        sample.clusters.add(self)

    def remove(self, sample):
        assert not self.destroyed
        if sample not in self.samples:
            return
        self._changes_since_proto_update += 1
        self.samples.remove(sample)
        sample.clusters.remove(self)

    def has_changed(self):
        return self._changes_since_proto_update > 0

    def is_redundant(self):
        return len([s for s in self.samples if len(s.clusters) == 1]) == 0

    def select_best_prototype(self):
        assert not self.destroyed
        # Consider samples that are only in this cluster.
        # We don't want to select a sample as prototype that belongs to multiple clusters.
        # samples_list = np.array([sample for sample in self.samples if len(sample.clusters) == 1])
        prototype_candidates = sorted(self.samples, key=lambda sample: sample.straightness)
        lengths = np.array([sample.length for sample in prototype_candidates])
        max_length_idx = np.argmax(lengths)
        max_length = lengths[max_length_idx]
        # Only consider longest samples for the selection of a prototype. Other selections lead to the creation of more clusters.
        # prototype_candidates = [sample for sample in prototype_candidates if sample.length >= 0.8 * max_length]
        representatives = [sample for sample in self.samples if len(sample.clusters) <= 1]
        min_similarity_score = np.inf
        new_prototype = None
        for prototype_candidate in prototype_candidates:
            similarity_score = 0
            for representative in representatives:
                mean_distance = self._cache.similarity(prototype_candidate, representative)
                if mean_distance == np.inf:
                    similarity_score = np.inf
                    break
                similarity_score += mean_distance
            if similarity_score < min_similarity_score:
                min_similarity_score = similarity_score
                new_prototype = prototype_candidate
                break
        if new_prototype is not None and new_prototype != self.prototype:
            self._size_at_proto_update = len(self.samples)
            self._changes_since_proto_update = 0
            self.prototype = new_prototype
            return True
        else:
            return False

    def calc_similarity(self, sample):
        return self._cache.similarity(self.prototype, sample)


class SimilarityCache:
    def __init__(self, max_distance, max_angle_diff, max_size=5e6):
        self._cache = {}
        self._access_stamps = {}
        self._max_cache_size = int(max_size)
        self.max_distance = max_distance
        self.max_angle_diff = max_angle_diff

    def similarities(self, sample_pairs):
        ret = [0] * len(sample_pairs)
        to_process = []
        for i, pair in enumerate(sample_pairs):
            if pair in self._cache:
                ret[i] = self._cache[pair]
            else:
                to_process.append((i, pair[0].points, pair[1].points))
            self._access_stamps[pair] = time.time()
        processed = smart_multi_core_exe(calc_similarities, to_process, (self.max_distance, self.max_angle_diff))
        for sample_idx, sim in processed:
            ret[sample_idx] = sim
            pair = sample_pairs[sample_idx]
            self._cache[pair] = sim
        return ret

    def shrink_to_max_size(self):
        if len(self._cache) > self._max_cache_size * 1.2:  # 20% margin
            stamp_to_key_dict = dict((v, k) for k, v in self._access_stamps.items())
            stamps_sorted = sorted(list(stamp_to_key_dict.keys()))
            stamps_to_delete = stamps_sorted[:-self._max_cache_size]
            for stamp in stamps_to_delete:
                key = stamp_to_key_dict[stamp]
                del self._cache[key]
                del self._access_stamps[key]

    def similarity(self, sample1, sample2):
        key = (sample1, sample2)
        self._access_stamps[key] = time.time()
        if key not in self._cache:
            self._cache[key] = calc_similarity(sample1.points, sample2.points, self.max_distance, self.max_angle_diff)
        return self._cache[key]


def remove_redundant_clusters(clusters):
    while True:
        min_redundant_size = np.inf
        redundant_cluster = None
        for c in clusters:
            if len(c.samples) < min_redundant_size and c.is_redundant():
                min_redundant_size = len(c.samples)
                redundant_cluster = c
        if redundant_cluster is None:
            break
        else:
            redundant_cluster.destroy()
            clusters = [c for c in clusters if not c.destroyed]
    return clusters


def remove_conflicting_clusters(clusters):
    while True:
        conflicts_storage = []
        worst_num_conflicts = 0
        for j, cluster in enumerate(clusters):
            num_conflicts = 0
            for other_cluster in clusters[j + 1:]:
                if cluster.calc_similarity(other_cluster.prototype) != np.inf:
                    num_conflicts += 1
            worst_num_conflicts = max(num_conflicts, worst_num_conflicts)
            if num_conflicts > 0:
                conflicts_storage.append((cluster, num_conflicts))
        if len(conflicts_storage) == 0:
            break
        else:
            worst_conflicting_clusters = [c for c, n in conflicts_storage]
            worst_conflicting_clusters = sorted(worst_conflicting_clusters, key=lambda c: c.prototype.length)
            worst_conflicting_clusters[0].destroy()
        clusters = [c for c in clusters if not c.destroyed]
    return clusters


def smart_multi_core_exe(func, samples, args, n_jobs=12, desc=None, batch_size=200):
    subdivisions = [samples[i:i + batch_size] for i in range(0, len(samples), batch_size)]
    process_array = subdivisions if desc is None else tqdm(subdivisions, desc=desc)
    subdivisions = joblib.Parallel(n_jobs=n_jobs, timeout=60)(
        joblib.delayed(func)(sub_div, *args)
        for sub_div in process_array)
    samples = []
    for sub_div in subdivisions:
        samples += sub_div
    return samples


def read_data_single_core(samples, img_shape, piece_length=-1):
    ret = []
    for sample_dict in samples:
        try:
            if piece_length is not None and piece_length > 0:
                for i in np.arange(0, 10, piece_length / 2):
                    sample = LocalMapSample(img_shape[1], img_shape[0], json_path=sample_dict["local_map_path"], start_offset=i, max_length=piece_length)
                    if sample.length > piece_length * 0.8:
                        sample.points += np.expand_dims(np.array([0, 0]) - sample.points[0], 0)
                        ret.append(sample)
            else:
                ret.append(LocalMapSample(img_shape[1], img_shape[0], json_path=sample_dict["local_map_path"]))
        except AssertionError:
            # ignore samples that do not work
            pass
    return ret


def calc_sample_assignments(sample, clusters):
    clusters = list(clusters)
    similarities_and_clusters = [(c.calc_similarity(sample), c) for c in clusters]
    similarities_and_clusters = [(s, c) for (s, c) in similarities_and_clusters if s != np.inf]
    similarities_and_clusters.sort(key=lambda t: t[0])
    return [c for s, c in similarities_and_clusters]


def cluster_unassigned_samples(samples, clusters, similarity_cache):
    for sample in samples:
        min_sim = np.inf
        min_sim_cluster = None
        for c in clusters:  # type: Cluster
            sim = c.calc_similarity(sample)
            if sim < min_sim:
                min_sim = sim
                min_sim_cluster = c
        if min_sim_cluster is not None:
            min_sim_cluster.add(sample)
        else:
            clusters.append(Cluster(sample, similarity_cache))


def _main():
    distances = calc_pairwise_distances(np.array([[0., 0.], [0., 1.0]]), np.array([[0., 0.], [1., 0.], [2., 0.]]))
    print(distances)
    print(calc_alignment_indices(np.array([[0., 0.], [0., 2.0]]), np.array([[0., 0.], [0., 1.], [0., 2.]])))
    print(calc_max_aligned_distance(np.array([[0., 0.],
                                              [0., 2.0]]),
                                    np.array([[1., 0.],
                                              [1., 1.],
                                              [1., 2.]])))
    print(calc_relative_angles(np.array([[0., 0.], [0., 1.], [0., 2.]]), np.array([[0., 0.], [0., 2.0]])))
    print(calc_relative_angles(np.array([[0., 0.], [0., 2.0]]), np.array([[0., 0.], [0., 1.], [0., 2.]])))


if __name__ == '__main__':
    _main()
