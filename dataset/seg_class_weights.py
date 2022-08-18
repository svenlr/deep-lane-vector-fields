import os
import cv2
import joblib
import numpy as np
from tqdm import tqdm


def read_or_init_class_freq(label_file, num_classes):
    class_freq_cache_file = label_file + "_class_freq"
    if os.path.exists(class_freq_cache_file):
        with open(class_freq_cache_file) as f:
            frequencies = f.read().split("\n")
            if len(frequencies) == num_classes:
                frequencies = [float(freq) for freq in frequencies]
                return frequencies
    image = cv2.imread(label_file)

    frequencies = []
    # For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
    for i in range(num_classes):
        class_mask = np.equal(image, i)
        class_mask = class_mask.astype(np.float32)
        class_frequency = np.sum(class_mask)
        frequencies.append(float(class_frequency))
    with open(class_freq_cache_file, "w+") as f:
        frequencies_str = [str(fr) for fr in frequencies]
        f.write("\n".join(frequencies_str))
    return frequencies


def enet_class_weighting(image_files, num_classes=12, bg_class_idx=None):
    '''
    The custom class weighing function as seen in the ENet paper.

    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately

    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.

    '''
    # initialize dictionary with all 0
    label_to_frequency = {}
    for i in range(num_classes):
        label_to_frequency[i] = 0

    frequencies_list = joblib.Parallel(n_jobs=6)(
        joblib.delayed(read_or_init_class_freq)(im_file, num_classes) for im_file in tqdm(image_files, desc="calc class frequency"))

    for n in range(len(image_files)):
        frequencies = frequencies_list[n]

        # For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
        for i in range(num_classes):
            class_frequency = frequencies[i]

            label_to_frequency[i] += class_frequency

    # perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        # class_weight = total_frequency / frequency
        class_weights.append(class_weight)

    class_weights = np.array(class_weights)
    class_weights /= np.sum(class_weights)
    class_weights *= 100

    if bg_class_idx is not None:
        # Set the background class_weight to 0.0
        if not isinstance(bg_class_idx, int):
            for i in bg_class_idx:
                class_weights[i] = 0
        else:
            class_weights[bg_class_idx] = 0.0

    return class_weights
