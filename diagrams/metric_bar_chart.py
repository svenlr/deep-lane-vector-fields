from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json


def main():
    path = "/home/sven/bisenet-torch/log/train_statistics_proj2_xy_tanh.json"
    data = json.load(open(path))
    data = data[-1]

    per_tag_dict = data["all"]

    labels = list(tag.strip() for tag in per_tag_dict.keys() if tag.strip() != "")
    mse_per_tag = [per_tag_dict[tag]["lane_mse_mean"] for tag in labels]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects2 = ax.bar(x, mse_per_tag, width, label='MAE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('MAE')
    ax.set_title('MAE of ' + Path(path).stem)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
