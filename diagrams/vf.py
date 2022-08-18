import numpy as np
import os
import cv2
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def save_vector_field(x_attractor, y_attractor, hue_shift=0, scaling=1.0):
    y, x = np.mgrid[0:x_attractor.shape[0]:1, 0:y_attractor.shape[1]:1]

    y_attractor2 = y_attractor
    x_attractor2 = x_attractor
    fig, ax = plt.subplots()
    p = np.sqrt(x_attractor2 ** 2 + y_attractor2 ** 2)
    p /= p.max()
    p *= scaling
    p[0, 0] = 1
    # print(y.shape, x.shape)
    # print(p.shape)
    # im = ax.imshow(p[::-1, ...], extent=[x.min(), x.max(), y.min(), y.max()])

    # Choose colormap
    cmap = plt.get_cmap("cubehelix")

    # Get the colormap colors
    my_cmap = cmap(np.arange(cmap.N))

    # Set alpha
    my_cmap[:, -1] = np.sqrt(np.linspace(0, 1, cmap.N))

    # Create new colormap
    my_cmap = ListedColormap(my_cmap)

    ax.streamplot(x, y, x_attractor2, y_attractor2, linewidth=1,
                  color=p, density=2, cmap="cubehelix")
    ax.invert_yaxis()

    # cont = ax.contour(x, y, p, cmap='gist_earth', vmin=p.min(), vmax=p.max())
    # labels = ax.clabel(cont)
    #
    # plt.setp(labels, path_effects=[withStroke(linewidth=8, foreground='w')])

    ax.set(aspect=1)
    ax.axis("off")
    import time
    directory = os.path.expanduser("~/Downloads/vector_fields")
    os.makedirs(directory, exist_ok=True)
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(directory, "" + str(time.time()) + ".pdf"), bbox_inches='tight', pad_inches=0, transparent=True)


if __name__ == '__main__':
    # x_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524.png_x_main_flow.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # y_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524.png_y_main_flow.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # x_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524.png_x_attractor_np.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # y_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524.png_y_attractor_np.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524_x_attractor.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    y_move = cv2.imread("/home/sven/bisenet-torch/log/proj/1548790419.092194524_y_attractor.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    # x_move -= 127
    # y_move -= 127
    # x_move = cv2.blur(x_move, (5, 5))
    # y_move = cv2.blur(y_move, (5, 5))
    x_move = (x_move / 127) - 1
    y_move = (y_move / 127) - 1
    save_vector_field(x_move, y_move, scaling=0.5)
