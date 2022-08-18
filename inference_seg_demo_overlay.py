import argparse
import os
import cv2
import numpy as np

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jit_ckpt', type=str, default=None, help="Path to the jit model to be tested")
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for testing')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu for testing')
    parser.add_argument("--data", type=str, required=True, help="path to data to run inference on")
    parser.add_argument("--output_video", type=str, required=True, help="path to store output video")
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--hide_class", type=int, default=1)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    model = torch.jit.load(args.jit_ckpt)

    use_cuda = torch.cuda.is_available() and args.use_gpu
    if use_cuda:
        model = model.cuda()

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)

    if os.path.isdir(args.data):
        files = os.listdir(args.data)
        files.sort()
    else:
        files = [args.data]

    writer = None
    for f in files:
        if not os.path.isfile(os.path.join(args.data, f)):
            continue
        img = cv2.imread(os.path.join(args.data, f), cv2.IMREAD_UNCHANGED)
        if img is not None:
            img_color = process_image(args, img, model)
            writer.write(img_color)
        if writer is None and img is not None:
            writer = build_writer(args)
        cap = cv2.VideoCapture(os.path.join(args.data, f))
        while True:
            _, img = cap.read()
            if img is None:
                break
            if writer is None:
                writer = build_writer(args)
            img_color = process_image(args, img, model)
            writer.write(img_color)

    writer.release()


def build_writer(args):
    # writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*''), 30, (320, 192), isColor=True)
    # assert writer.isOpened()
    os.makedirs(args.output_video + "_pngs", exist_ok=True)
    return ImgWriter(args.output_video + "_pngs", os.path.basename(args.output_video) + ".avi")


class ImgWriter:
    def __init__(self, path, video_file):
        self.i = 0
        self.path = path
        self.video_file = video_file

    def write(self, x):
        self.i += 1
        f = os.path.join(self.path, "{:04d}.png".format(self.i))
        cv2.imwrite(f, x, [cv2.IMWRITE_JPEG_QUALITY, 100])

    def release(self):
        os.chdir(self.path)
        # os.system("ffmpeg -framerate 42 -pattern_type glob -i '*.jpg' -c:v copy ../" + self.video_file)


def process_image(args, img, model):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape == (600, 800):
        img_orig = img
        img = cv2.resize(img[176:-40, 80:-80, ...], dsize=(320, 192))
    else:
        img_orig = None
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    inp = torch.from_numpy(img)
    if args.use_gpu:
        inp = inp.cuda()
    output = model(inp)
    if isinstance(output, tuple):
        output = output[0]
    output = output.cpu().numpy()
    output_color = cv2.applyColorMap((output.astype(np.float) * 255 / args.num_classes).clip(0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    img_color[output != args.hide_class] = img_color[output != args.hide_class] * 0.4 + output_color[output != args.hide_class] * 0.6
    if img_orig is not None:
        img_orig = cv2.cvtColor(img_orig, cv2.COLOR_GRAY2BGR)
        img_orig[176:-40, 80:-80, ...] = cv2.resize(img_color, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        img_color = img_orig
    return img_color


if __name__ == '__main__':
    main()
