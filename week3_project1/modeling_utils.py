import torch
from torch import nn
import torch.nn.functional as F

def _make_divisible(v: float, divisor: int, min_value = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Conv2dNormActivation(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, stride=1, groups=1, padding=0, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = kernel_size, stride=stride, groups=groups, padding=padding),
            norm_layer(out_dim),
            activation_layer()
        ) 
        
    def forward(self, x):
        return self.layers(x)
    