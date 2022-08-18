import json
import os

from nn_utils.geometry_utils import *


class LocalMapSample:
    def __init__(self, img_width, img_height, json_path, max_length=5, step_size=0.1, add_to_bottom_limit=0, start_offset=0,
                 cut_points_to_rect=False, enable_caching=True):
        self.clusters = set()
        self.start_offset = start_offset
        self.json_path = json_path
        json_string = open(json_path).read()
        self.json_data = json.loads(json_string)
        self.img_width = img_width
        self.img_height = img_height
        use_restored_points = False
        if enable_caching:
            use_restored_points = self.attempt_cache_restore()
        if not use_restored_points:
            self.right_points = np.array(self.json_data.get("right_lane", {}).get("right_marking", []), dtype=np.float32)
            self.points = np.array(self.json_data.get("right_lane", {}).get("left_marking", []), dtype=np.float32)
            self.visibility_mask = np.array(self.json_data.get("visibility_mask", np.ones(shape=(self.points.shape[0],), dtype=np.float32)), dtype=np.float32)
            if len(self.points) > 0:
                self.left_points = self.points + self.points - self.right_points
                self._make_points_equidistant(step_size)
            else:
                self.left_points = np.zeros(shape=self.points, dtype=np.float32)
                self.visibility_mask = np.array([], dtype=np.float32)
            self._save_cache()

        self.limits = self.transform_to_limits(img_width, img_height, self.json_data)
        self.limits[0][0] -= add_to_bottom_limit
        self.length = 0.
        self.straightness = 10000
        if len(self.points) >= 2 and cut_points_to_rect:
            self.cut_points_to_rect(self.limits, max_length)
        else:
            self.length = street_length(self.points)

    def attempt_cache_restore(self):
        restored_points = False
        if self._has_cache():
            try:
                self._restore_cache()
                restored_points = True
            except Exception:
                restored_points = False
        return restored_points

    @property
    def num_points(self):
        return len(self.points)

    def _save_cache(self):
        np.savez(self.json_path.replace(".json", ".npz"),
                 points=self.points, right_points=self.right_points, left_points=self.left_points,
                 visibility_mask=self.visibility_mask)

    def _has_cache(self):
        return os.path.exists(self.json_path.replace(".json", ".npz"))

    def _restore_cache(self):
        with np.load(self.json_path.replace(".json", ".npz")) as cache:
            self.points = cache["points"]
            self.right_points = cache["right_points"]
            self.left_points = cache["left_points"]
            self.visibility_mask = cache["visibility_mask"]

    def get_ego_street_training_mask(self):
        path = self.json_path.replace(".json", "_street_mask.png")
        if os.path.exists(path):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask.shape[0] == self.img_height and mask.shape[1] == self.img_width:
                return mask
        mask = np.zeros((self.img_height, self.img_width), dtype=np.uint8)
        pixels_per_meter = self.json_data["transform"]["pixels_per_meter"]
        car_to_image_offset = self.json_data["transform"]["car_to_image_offset"]
        cv_points = [world_to_ipm(p, self.img_width, self.img_height, pixels_per_meter, car_to_image_offset)
                     for p in self.points]
        for p in cv_points:
            cv2.circle(mask, tuple(p), int(1.0 * pixels_per_meter), 1, thickness=-1)
        for p in cv_points:
            cv2.circle(mask, tuple(p), int(0.6 * pixels_per_meter), 0, thickness=-1)
        for p in cv_points:
            cv2.circle(mask, tuple(p), int(0.2 * pixels_per_meter), 2, thickness=-1)
        mask = cv2.medianBlur(mask, 3)
        # cv2.imshow("test", mask * 100)
        # cv2.waitKey(0)
        cv2.imwrite(path, mask)
        return mask

    def cut_points_to_rect(self, limits, max_length):
        self.cut_before(closest_point_index(self.points, np.array([0, 0], dtype=np.float32)))
        first_right_visible_idx = determine_first_visible_in_limits(self.right_points, limits)
        first_left_visible_idx = determine_first_visible_in_limits(self.left_points, limits)
        assert first_right_visible_idx is not None or first_left_visible_idx is not None  # well, at least one of the markings should be visible
        if first_right_visible_idx is None:
            first_right_visible_idx = first_left_visible_idx
        if first_left_visible_idx is None:
            first_left_visible_idx = first_right_visible_idx
        first_idx = min(first_right_visible_idx, first_left_visible_idx)
        last_right_visible_idx = determine_last_visible_in_limits(self.right_points, limits, first_right_visible_idx)
        last_left_visible_idx = determine_last_visible_in_limits(self.left_points, limits, first_left_visible_idx)
        assert last_right_visible_idx is not None or last_left_visible_idx is not None  # well, at least one of the markings should be visible
        if last_right_visible_idx is None:
            last_right_visible_idx = last_left_visible_idx
        if last_left_visible_idx is None:
            last_left_visible_idx = last_right_visible_idx
        last_idx = max(last_right_visible_idx, last_left_visible_idx)
        self.points = self.points[first_idx:last_idx]
        vectors = self.points[1:] - self.points[:-1]
        lengths = np.sqrt((vectors ** 2).sum(axis=1))
        distance_on_points = np.cumsum(lengths)
        distance_on_points = np.concatenate([[0], distance_on_points])
        if len(distance_on_points) == 0:
            self.length = 0
        elif distance_on_points[-1] > max_length + self.start_offset:
            cut_idx = np.argmax(distance_on_points > max_length + self.start_offset)
            self.points = self.points[:cut_idx]
            self.length = distance_on_points[cut_idx - 1]
        else:  # len(distance_on_points) > 0:
            self.length = distance_on_points[-1]
        self.left_points = self.left_points[first_idx:first_idx + len(self.points)]
        self.right_points = self.right_points[first_idx:first_idx + len(self.points)]
        self.visibility_mask = self.visibility_mask[first_idx:first_idx + len(self.points)]
        if self.start_offset > 0:
            before_offset_mask = distance_on_points < self.start_offset - 1e-5
            if np.all(before_offset_mask):
                start_idx = self.num_points
            else:
                start_idx = np.argmin(before_offset_mask)
            self.cut_before(start_idx)
        self.straightness = calc_straightness_score(self.points)  # lower is better

    def _make_points_equidistant(self, step_size):
        if not np.all(self.visibility_mask):
            first_non_visible_idx = np.argmin(self.visibility_mask)
            visible_points = make_equidistant(iterative_make_continuous(self.points[:first_non_visible_idx + 1]), step_size=step_size)[:-1]
            invisible_points = make_equidistant(iterative_make_continuous(self.points[first_non_visible_idx:]), step_size=step_size)
            self.points = np.concatenate([visible_points, invisible_points], axis=0)
            self.visibility_mask = np.concatenate([np.ones((visible_points.shape[0],)), np.zeros((invisible_points.shape[0],))]).astype(np.float32)
        else:
            self.points = iterative_make_continuous(self.points)
            self.points = make_equidistant(self.points, step_size=step_size)
            self.visibility_mask = np.ones(shape=(self.points.shape[0],)).astype(np.float32)
        vectors = self.points[1:] - self.points[:-1]
        lengths = np.sqrt((vectors ** 2).sum(axis=1))
        vectors_normed = vectors / np.expand_dims(lengths, -1)
        vectors_normed = np.concatenate([vectors_normed[:1], vectors_normed], axis=0)
        offset_vectors = left_orthogonal(vectors_normed) * 0.42
        self.left_points = self.points + offset_vectors
        self.right_points = self.points - offset_vectors

    def cut_from(self, idx):
        self.length -= street_length(self.points[idx:])
        self.points = self.points[:idx]
        self.left_points = self.left_points[:idx]
        self.right_points = self.right_points[:idx]
        self.visibility_mask = self.visibility_mask[:idx]

    def cut_before(self, idx):
        self.length -= street_length(self.points[:idx])
        self.points = self.points[idx:]
        self.left_points = self.left_points[idx:]
        self.right_points = self.right_points[idx:]
        self.visibility_mask = self.visibility_mask[idx:]

    @staticmethod
    def transform_to_limits(img_width, img_height, json_data):
        return np.array([
            transform_to_world([img_width, img_height], img_width, img_height, json_data),
            transform_to_world([0, 0], img_width, img_height, json_data)
        ], dtype=np.float32)
