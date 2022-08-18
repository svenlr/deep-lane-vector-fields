from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json


def get_curve(data, path):
    values = []
    for entry in data:
        value = entry[path[0]]
        for p in path[1:]:
            value = value[p]
        values.append(value)
    return values


def main():
    path = "/home/sven/bisenet-torch/log/proj/train_statistics.json"
    data = json.load(open(path))

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(get_curve(data, ["real", "all", "lane_f1_mean"]), 'b-')
    ax2.plot(get_curve(data, ["lr"]), 'g-')
    plt.title('1 Cycle Learning Rate Schedule')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('F1 measure', color='b')
    ax2.set_ylabel('learning rate', color='g')
    # plt.legend(['loss', 'f1', 'lr'], loc='upper left')

    plt.savefig("lr_f1.pdf")
    plt.show()


if __name__ == '__main__':
    main()
