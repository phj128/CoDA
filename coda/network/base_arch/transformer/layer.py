import torch
import torch.nn as nn
import torch.nn.functional as F


def zero_module(module, nozero=False):
    """
    Zero out the parameters of a module and return it.
    """
    if nozero:
        return module
    for p in module.parameters():
        p.detach().zero_()
    return module


def modulate(x, shift, scale):
    return x * (1 + scale) + shift
