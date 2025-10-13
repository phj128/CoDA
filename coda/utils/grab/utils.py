import numpy as np
import torch
from copy import copy


def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k, v in out_dict.items():
        if isinstance(v, dict):
            out_dict[k] = DotDict(v)
    return dotdict(out_dict)


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)


def params2torch(params, dtype=torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}


def to_np(tensor):
    return tensor.detach().cpu().numpy()


def to_cpu(tensor):
    return tensor.detach().cpu()


def to_cuda(tensor):
    return tensor.cuda()


def params2cuda(params):
    return {k: to_cuda(v) for k, v in params.items()}


def params2np(params):
    return {k: to_np(v) for k, v in params.items()}


def params2cpu(params):
    return {k: to_cpu(v) for k, v in params.items()}
