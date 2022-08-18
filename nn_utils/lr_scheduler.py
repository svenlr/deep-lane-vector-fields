#!/usr/bin/python
# -*- encoding: utf-8 -*-
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import CyclicLR

from nn_utils.math_utils import downsample_1d_np


class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            ratio = self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio


class WarmupPolyLrScheduler(WarmupLrScheduler):

    def __init__(
            self,
            optimizer,
            power,
            max_iter,
            warmup_iter=500,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.power = power
        self.max_iter = max_iter
        super(WarmupPolyLrScheduler, self).__init__(
            optimizer, warmup_iter, warmup_ratio, warmup, last_epoch)

    def get_main_ratio(self):
        real_iter = self.last_epoch - self.warmup_iter
        real_max_iter = self.max_iter - self.warmup_iter
        alpha = real_iter / real_max_iter
        ratio = (1 - alpha) ** self.power
        return ratio


class LinearLR(object):
    def __init__(self, base_lr, max_epochs):
        super(LinearLR, self).__init__()
        self.base_lr = base_lr
        self.max_epochs = max_epochs

    def get_lr(self, epoch):
        curr_lr = self.base_lr - (self.base_lr * (epoch / (self.max_epochs)))
        return [curr_lr]

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Base LR: {}\n'.format(self.base_lr)
        return fmt_str


class CyclicThenLinearLR(torch.optim.lr_scheduler._LRScheduler):
    """ Cyclic LR scheduler followed by linear fine-tuning as described in the ESPNetV2 paper. """
    def __init__(self, optimizer, base_lr, max_lr, max_iter, cyclic_max_rel=0.5, cycle_len_rel=0.05):
        cycle_len = int(np.round(cycle_len_rel * max_iter))
        self.linear_epochs = max_iter - int(cyclic_max_rel * max_iter) + 1
        self.clr = CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=int(cycle_len * 0.2), step_size_down=int(cycle_len * 0.8))
        self.decay_lr = LinearLR(base_lr=base_lr, max_epochs=self.linear_epochs)
        self.cyclic_epochs = int(cyclic_max_rel * max_iter)

        self.base_lr = base_lr
        self.max_epochs = max_iter
        self.clr_max = int(cyclic_max_rel * max_iter)
        self.cycle_len = cycle_len

        super(CyclicThenLinearLR, self).__init__(optimizer)

    def get_lr(self):
        epoch = self.last_epoch + 1
        if epoch < self.cyclic_epochs:
            curr_lr = self.clr.get_lr()
        else:
            curr_lr = self.decay_lr.get_lr(epoch - self.cyclic_epochs + 1)
        return curr_lr

    def step(self):
        super(CyclicThenLinearLR, self).step()
        if self.last_epoch + 1 < self.cyclic_epochs:
            self.clr.step()

    def __repr__(self):
        fmt_str = 'Scheduler ' + self.__class__.__name__ + '\n'
        fmt_str += '    Total Epochs: {}\n'.format(self.max_epochs)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.cycle_len, int(self.clr_max / self.cycle_len))
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.cycle_len, self.base_lr)
        fmt_str += '    Cycle with length of {}: {}\n'.format(self.linear_epochs, 1)
        fmt_str += '    Base LR with {} cycle length: {}\n'.format(self.linear_epochs, self.base_lr)
        return fmt_str


class OneCycleLR(torch.optim.lr_scheduler._LRScheduler):
    """ One cycle LR scheduler. Increase learning rate up to the turn point, than decreases it.
        Also supports an initial warmup phase. A final finetune phase is also recommended and supported.
        Based on a paper, which can be found on Google under "one cycle learning rate". """

    def __init__(self, optimizer, max_lr, max_iter, turn_point_ratio, finetune_ratio=0.05,
                 cycle_momentum=True, min_momentum=0.8, base_momentum=0.9, last_epoch=-1, initial_warmup_iter=0,
                 min_to_max_ratio=0.1):

        base_lr = max_lr * min_to_max_ratio
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.initial_warmup_iter = min(initial_warmup_iter, int(self.max_iter / 10))
        self.cycle_iter = int((1 - finetune_ratio) * (self.max_iter - self.initial_warmup_iter))
        self.turn_point = self.initial_warmup_iter + int(turn_point_ratio * self.cycle_iter)
        self.start_finetune_point = self.initial_warmup_iter + self.cycle_iter

        base_lrs = _format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if 'momentum' not in optimizer.defaults:
                raise ValueError('optimizer must support momentum with `cycle_momentum` option enabled')

            base_momentums = _format_param('base_momentum', optimizer, min_momentum)
            if last_epoch == -1:
                for momentum, group in zip(base_momentums, optimizer.param_groups):
                    group['momentum'] = momentum
            self.min_momentum = np.array(list(map(lambda group: group['momentum'], optimizer.param_groups)))
            self.base_momentums = np.array(_format_param('max_momentum', optimizer, base_momentum))

        self.max_lrs = _format_param('max_lr', optimizer, max_lr)
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
        self.base_lrs = base_lrs

    def set_momentums(self, momentums):
        if self.cycle_momentum:
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                param_group['momentum'] = momentum

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.initial_warmup_iter:
            # in initial warmup phase
            self.set_momentums(self.base_momentums)
            return float(epoch) / self.initial_warmup_iter * np.array(self.base_lrs)
        elif epoch < self.turn_point:
            # in rising lr phase
            percent = (epoch - self.initial_warmup_iter) / (self.turn_point - self.initial_warmup_iter)
            self.set_momentums(self.min_momentum * percent + self.base_momentums * (1 - percent))
            return np.array(self.base_lrs) * (1 - percent) + np.array(self.max_lrs) * percent
        elif epoch < self.start_finetune_point:
            # in falling lr phase
            percent = (epoch - self.turn_point) / (self.start_finetune_point - self.turn_point)
            self.set_momentums(self.min_momentum * (1 - percent) + self.base_momentums * percent)
            return np.array(self.base_lrs) * percent + np.array(self.max_lrs) * (1 - percent)
        else:
            # in finetune phase
            self.set_momentums(self.base_momentums)
            percent = (epoch - self.start_finetune_point) / (self.max_iter - self.start_finetune_point)
            return np.array(self.base_lrs) * (1 - percent)


class AdaptiveCyclicLR(torch.optim.lr_scheduler._LRScheduler):
    """ An LR scheduler for the fine-tuning phase that performs learning rate tests at each epoch in order to determine the next LR heuristically """
    def __init__(self, optimizer, base_lr, max_lr, step_size_up=2000, step_size_down=None, last_epoch=-1, min_lr_rel_gradient=0.05, lr_momentum=0.9,
                 decay_mode="sqrt", max_iter=None):
        if step_size_up < 100:
            print("[WARN] Might be too few warmup steps to estimate lr ranges ({})".format(step_size_up))

        self.optimizer = optimizer

        assert decay_mode == "auto" or max_iter is not None
        self.decay_mode = decay_mode
        self.max_iter = max_iter
        self.initial_max_lr = max_lr
        self.initial_base_lr = base_lr

        self.min_lr_rel_gradient = min_lr_rel_gradient
        self.lr_momentum = lr_momentum
        self.loss_record = []
        self.lr_record = []
        self.gradient_filter_rel = 0.1

        base_lrs = _format_param('base_lr', optimizer, base_lr)
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                group['lr'] = lr

        self.max_lrs = _format_param('max_lr', optimizer, max_lr)

        step_size_up = float(step_size_up)
        step_size_down = float(step_size_down) if step_size_down is not None else step_size_up
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        super(AdaptiveCyclicLR, self).__init__(optimizer, last_epoch)
        self.base_lrs = base_lrs

    def report_loss(self, loss):
        if self.calc_cycle_progress() > self.step_ratio and self.decay_mode == "auto":
            self.loss_record.append(loss)
            self.lr_record.append(np.mean(self.get_lr()))

    def step(self):
        super(AdaptiveCyclicLR, self).step()
        if self.last_epoch % self.total_size == 0 and self.decay_mode in ["sqrt", "linear", "square"]:
            progress = self.last_epoch / self.max_iter
            lr_percent = 1 - progress
            if self.decay_mode == "sqrt":
                lr_percent = np.sqrt(lr_percent)
            elif self.decay_mode == "square":
                lr_percent = lr_percent ** 2
            self.max_lrs = _format_param("max_lr", self.optimizer, lr_percent * self.initial_max_lr)
            self.base_lrs = _format_param("base_lr", self.optimizer, min(lr_percent * self.initial_base_lr, lr_percent * self.initial_max_lr / 5))
        if self.decay_mode == "auto" and self.calc_cycle_progress() <= self.step_ratio:
            if len(self.loss_record) > 0:
                self.recalculate_limits()
                self.loss_record = []
                self.lr_record = []

    def shape_fn(self, scale_factor, up=True):
        if up:
            scale_factor = np.exp(scale_factor * 5) / np.exp(5)
        else:
            # scale_factor = scale_factor
            scale_factor = np.sin(scale_factor * np.pi / 2)
        return scale_factor

    def recalculate_limits(self):
        window_size = max(int(self.gradient_filter_rel * len(self.lr_record)), 1)
        lr_record = np.convolve(self.lr_record, np.ones((window_size,)) / window_size, mode='valid')
        loss_record = np.convolve(self.loss_record, np.ones((window_size,)) / window_size, mode='valid')
        loss_gradients = loss_record[1:] - loss_record[:-1]
        loss_gradients = np.convolve(loss_gradients, np.ones((window_size,)) / window_size, mode='valid')
        offset = (len(loss_record) - len(loss_gradients)) // 2
        best_loss_gradient_idx = np.argmin(loss_gradients)
        new_max_lr_idx = int(np.clip(offset + best_loss_gradient_idx, 0, len(loss_record) - 1))
        new_max_lr = lr_record[new_max_lr_idx]
        new_min_lr = lr_record[new_max_lr_idx // 2]
        for i in range(int(best_loss_gradient_idx)):
            # find the first gradient that is at least min_lr_rel_gradient percent as fast as the best
            if loss_gradients[i] < loss_gradients[best_loss_gradient_idx] * self.min_lr_rel_gradient:
                new_min_lr = lr_record[offset + i]
                break
        new_max_lr = new_max_lr * (1 - self.lr_momentum) + np.mean(self.max_lrs) * self.lr_momentum
        new_base_lr = new_max_lr * self.initial_base_lr / self.initial_max_lr
        self.base_lrs = _format_param("base_lr", self.optimizer, new_base_lr)
        self.max_lrs = _format_param("max_lr", self.optimizer, new_max_lr)

    def get_lr(self):
        x = self.calc_cycle_progress()
        if x <= self.step_ratio:
            scale_factor = min(1, 1 / (1 - 1.5 * self.gradient_filter_rel) * x / self.step_ratio)
            up = True
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)
            up = False

        base_heights = (np.array(self.max_lrs) - self.base_lrs) * self.shape_fn(scale_factor, up)
        lrs = self.base_lrs + base_heights

        return lrs

    def calc_cycle_progress(self):
        cycle = np.floor(1 + self.last_epoch / self.total_size)
        x = 1. + self.last_epoch / self.total_size - cycle
        return x


def _format_param(name, optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""
    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError("expected {} values for {}, got {}".format(
                len(optimizer.param_groups), name, len(param)))
        return param
    else:
        return [param] * len(optimizer.param_groups)


def plot_learning_rate_test_and_exit(args, loss_record, lr_record):
    """ Save the learning rate test performed by train.py in a plot, so that one can visually determine the maximum learning rate.
        A learning rate test can be started using train.py, see argparse help. Please set all other hyper parameters before. """
    import matplotlib.pyplot as plt
    import sys
    # fix problems by removing extreme large values
    lr_record = np.array(lr_record)
    loss_record = np.array(loss_record)
    lr_record = lr_record[np.array(loss_record) < 1e6]
    loss_record = loss_record[np.array(loss_record) < 1e6]
    window_size = min(10, int(len(lr_record) / 10) + 1)
    loss_record = np.convolve(loss_record, np.ones((window_size,)) / window_size, mode='valid')
    lr_record = lr_record[-len(loss_record):]
    num_points = 50
    lr_record = downsample_1d_np(lr_record, num_points)
    loss_record = downsample_1d_np(loss_record, num_points)
    # loss_gradients = loss_record[1:] - loss_record[:-1]
    # plt.subplot(221)
    plt.plot(lr_record, loss_record)
    plt.grid()
    axes = plt.gca()
    axes.set_ylim([-0.00, min(np.min(loss_record) * 3, np.median(loss_record) * 1.5)])
    plt.xscale('log')
    # plt.subplot(222)
    # window_size = min(15, int(len(lr_record) / 10) + 1)
    # loss_gradients = np.convolve(loss_gradients, np.ones((window_size,)) / window_size, mode='valid')
    # offset = (len(lr_record) - len(loss_gradients)) // 2
    # plt.plot(lr_record[offset:offset + len(loss_gradients)], loss_gradients)
    # plt.grid()
    # axes = plt.gca()
    # axes.set_ylim([max(-0.3, np.min(loss_gradients)), min(0.3, -np.min(loss_gradients))])
    # plt.xscale('log')
    plt.savefig(os.path.join(args.save_model_path, "lr_range_test.svg"))
    sys.exit(0)


def main_test():
    import matplotlib.pyplot as plt

    model = torch.nn.Conv2d(3, 16, 3, 1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1e-2)

    max_iter = 10000
    # lr_scheduler = WarmupPolyLrScheduler(optim, 0.9, max_iter, max_iter - 1, 0.001, 'exp', -1)
    lr_scheduler = CyclicThenLinearLR(optim, 0.1, 0.9, max_iter)
    lr_scheduler = AdaptiveCyclicLR(optim, 0.1, 1, int(max_iter * 0.2 / 20), int(max_iter * 0.8 / 20),
                                    max_iter=max_iter, decay_mode="linear")
    lr_scheduler = OneCycleLR(optim, 1, max_iter, 0.5, initial_warmup_iter=0)
    lrs = []
    momentums = []
    step = 0
    for _ in range(max_iter):
        step += 1
        lr = lr_scheduler.get_lr()
        print(lr)
        lrs.append(np.mean(lr))
        optim.step()
        lr_scheduler.step()
        if "momentum" in optim.param_groups[0]:
            momentums.append(optim.param_groups[0]["momentum"])
        # lr_scheduler.report_loss(np.sqrt((max_iter - _) / max_iter))

    lrs = np.array(lrs)
    n_lrs = len(lrs)
    plt.plot(np.arange(n_lrs), lrs)
    if len(momentums) == n_lrs:
        plt.plot(np.arange(n_lrs), momentums)
    plt.grid()
    plt.yscale("linear")
    plt.show()


if __name__ == "__main__":
    main_test()
