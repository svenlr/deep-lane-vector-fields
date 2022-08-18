import torch
import torch.nn as nn


class ConvBN(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBN, self).__init__()
        self.conv = nn.Conv2d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        return feat


class CutReduce(nn.Module):
    def __init__(self, in_channels, cut_top=0, cut_bottom=0):
        super().__init__()
        self.cut_top = cut_top
        self.cut_end = in_channels - cut_bottom

    def forward(self, x):
        return x[:, self.cut_top:self.cut_end, ...]


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, dilation1=1, mid_relu=True, out_channels=None, res_projection="conv", bottleneck_channels=None,
                 out_activation_fn=None):
        super(ResBlock, self).__init__()
        if out_activation_fn is None:
            out_activation_fn = nn.PReLU
        if out_channels is None:
            out_channels = channels
        assert channels >= out_channels
        if channels == out_channels:
            self.res_projection = nn.Identity()
        else:
            if res_projection == "cut_top":
                self.res_projection = CutReduce(channels, cut_top=channels - out_channels)
            elif res_projection == "cut_bottom":
                self.res_projection = CutReduce(channels, cut_bottom=channels - out_channels)
            else:  # conv
                self.res_projection = ConvBN(channels, out_channels, ks=1, bias=True, padding=0)
        if bottleneck_channels is not None:
            conv_mid = ConvBN(bottleneck_channels, bottleneck_channels, ks=kernel_size, bias=True, padding=padding + dilation1 - 1, dilation=dilation1)
            if mid_relu:
                conv_mid = nn.Sequential(conv_mid, nn.PReLU())
            self.conv1 = nn.Sequential(
                ConvBN(channels, bottleneck_channels, ks=1, bias=True, padding=0),
                nn.PReLU(),
            )
            self.conv2 = nn.Sequential(
                conv_mid,
                ConvBN(bottleneck_channels, out_channels, ks=1, bias=True, padding=0),
            )
        else:
            self.conv1 = ConvBN(channels, out_channels, ks=kernel_size, bias=True, padding=padding + dilation1 - 1, dilation=dilation1)
            if mid_relu:
                self.conv1 = nn.Sequential(self.conv1, nn.PReLU())
            self.conv2 = ConvBN(out_channels, out_channels, ks=kernel_size, bias=True, padding=padding)
        self.out_activation = out_activation_fn()

    def forward(self, x):
        return self.out_activation(self.res_projection(x) + self.conv2(self.conv1(x)))


class ChannelPad2d(nn.Module):
    def __init__(self, num_channels):
        super(ChannelPad2d, self).__init__()
        self.num_channels = num_channels

    def forward(self, x):
        return torch.cat([x, torch.zeros((x.shape[0], self.num_channels, x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)])


class ResBlockDown(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, mid_relu=True, out_activation_fn=nn.PReLU, skip_mode="conv"):
        super(ResBlockDown, self).__init__()
        self.conv1 = ConvBN(channels_in, channels_in * 2, ks=kernel_size, stride=2, bias=True, padding=1)
        if mid_relu:
            self.conv1 = nn.Sequential(self.conv1, nn.PReLU())
        self.conv2 = ConvBN(channels_in * 2, channels_out, ks=kernel_size, bias=True, padding=1)
        if skip_mode == "conv":
            self.conv_skip = ConvBN(channels_in, channels_out, ks=1, stride=1, padding=0)
        else:
            self.conv_skip = nn.Identity() if channels_out == channels_in else ChannelPad2d(channels_out - channels_in)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.relu = out_activation_fn()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        x = self.max_pool(self.conv_skip(x))
        return self.relu(x + feat)


class FullyConnected2LayerHead(nn.Module):
    def __init__(self, in_chn, out_chn, mid_chn=None):
        super().__init__()
        if mid_chn is None:
            mid_chn = in_chn // 2
        self.seq = nn.Sequential(
            nn.Linear(in_chn, mid_chn, bias=True),
            nn.PReLU(),
            nn.BatchNorm1d(mid_chn),
            nn.Linear(mid_chn, out_chn, bias=True),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.seq(x)


class ScaledTanh(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x / self.scale)


class AutoScaleTanh(nn.Module):
    def __init__(self, channels, init_weight=1.0):
        super().__init__()
        self.register_parameter("factors", torch.nn.Parameter(torch.ones((channels,), dtype=torch.float32) * init_weight))

    def forward(self, x):
        return self.factors.view(1, self.factors.shape[0], *[1 for i in x.shape[2:]]) * torch.tanh(x)


class ConvBN1D(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBN1D, self).__init__()
        self.conv = nn.Conv1d(
            in_chan, out_chan, kernel_size=ks, stride=stride,
            padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_chan)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        return feat


class ResBlock1D(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, mid_relu=True):
        super(ResBlock1D, self).__init__()
        self.conv1 = ConvBN1D(channels, channels, ks=kernel_size, bias=True, padding=padding)
        if mid_relu:
            self.conv1 = nn.Sequential(self.conv1, nn.PReLU())
        self.conv2 = ConvBN1D(channels, channels, ks=kernel_size, bias=True, padding=padding)
        self.relu = nn.RReLU()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        return self.relu(x + feat)


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.mean(x, dim=[2, 3])
