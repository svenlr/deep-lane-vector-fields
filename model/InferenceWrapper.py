import torch
import torch.nn as nn

from nn_utils.geometry_utils import decode_viz_mask
from nn_utils.seg_metrics import reverse_one_hot_uint8


class InferenceWrapper(nn.Module):
    def __init__(self, module, width, height):
        super().__init__()
        self.module = module
        # self.normalize_transform = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.input_width = width
        self.input_height = height
        self.register_buffer("mean", torch.FloatTensor((0.485, 0.456, 0.406)))
        self.register_buffer("std", torch.FloatTensor((0.229, 0.224, 0.225)))

    def internal_forward(self, inp):
        # convert input to 32FC1 and then to "color" image 32FC3
        if inp.ndim == 2:
            inp = inp.type(torch.cuda.FloatTensor)
            inp /= 255
            inp = torch.stack((inp, inp, inp))
        if inp.shape[0] == 3:
            # normalize input
            inp = (inp - self.mean.view(-1, 1, 1)) / self.std.view(-1, 1, 1)
        # add batch dimension, because NN is used to it
        inp = inp.unsqueeze(0)
        out = self.module(inp)
        return out

    def forward(self, inp):
        out = self.internal_forward(inp)
        # remove batch dimension before returning
        if isinstance(out, tuple):
            out = list(out)
            for key in range(len(out)):
                out[key] = out[key].squeeze(0)
            out[0] = reverse_one_hot_uint8(out[0])
            return tuple(out)
        elif isinstance(out, dict):
            for key in out.keys():
                out[key] = out[key].squeeze(0)
            if "seg" in out:
                out["seg"] = reverse_one_hot_uint8(out["seg"])
            out_array = []
            if "seg" in out:
                out_array.append(out["seg"])
            if "local_map_rl" in out:
                out_array.append(out["local_map_rl"])
                if "local_map_rr" in out:
                    out_array.append(out["local_map_rr"])
            if "visibility_mask" in out:
                out_array.append(decode_viz_mask(out["visibility_mask"].unsqueeze(0)).squeeze(0))
            if "lane_attractor" in out:
                out_array.append(out["lane_attractor"])
            if "visibility_grid" in out:
                out_array.append(out["visibility_grid"])
            if "main_flow" in out:
                out_array.append(out["main_flow"])
            return tuple(out_array)
        else:
            return reverse_one_hot_uint8(out.squeeze(0))


class ScriptTupleInferenceWrapper(InferenceWrapper):
    def __init__(self, module, width, height):
        super().__init__(module, width, height)

    def forward(self, inp):
        out = self.internal_forward(inp)
        # remove batch dimension before returning
        return reverse_one_hot_uint8(out[0].squeeze(0)), out[1].squeeze(0)


class ScriptInferenceWrapper(InferenceWrapper):
    def __init__(self, module, width, height):
        super().__init__(module, width, height)

    def forward(self, inp):
        out = self.internal_forward(inp)
        # remove batch dimension before returning
        return reverse_one_hot_uint8(out[0].squeeze(0))


class BatchOnlyInferenceWrapper(nn.Module):
    def __init__(self, module, width, height):
        super().__init__()
        self.module = module
        self.input_width = width
        self.input_height = height

    def forward(self, inp):
        return self.module(inp.unsqueeze(0)).squeeze(0)
