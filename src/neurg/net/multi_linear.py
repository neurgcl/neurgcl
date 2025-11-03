import math

import torch
from torch import Tensor, nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter


class MultiLinear(Module):
    """
    x: (ch, b, in)
    w: (ch, in, out)
    bias: (ch, 1, out)

    out: (ch, b, out)  = x @ w + bias
    """

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self, in_features: int, out_features: int, channels: int, bias: bool = True, device=None, dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.channels = channels

        self.weight = Parameter(torch.empty((channels, in_features, out_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(channels, 1, out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        #     init.uniform_(self.bias, -bound, bound)

        # init weight --------------------
        a = math.sqrt(5)
        nonlinearity = 'leaky_relu'
        fan_in = self.in_features
        gain = init.calculate_gain(nonlinearity, a)
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        # init bias --------------------
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        """
        Args:
            input (Tensor): (ch, batch, in)

        Returns:
            Tensor: (ch, batch, out)
        """

        return torch.bmm(input, self.weight) + self.bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, channels={}, bias={}'.format(
            self.in_features, self.out_features, self.channels, self.bias is not None
        )


class MultiMlp(nn.Sequential):
    def __init__(
        self, sizes: int, n_ch: int, activation: nn.Module, output_activation: nn.Module = nn.Identity
    ) -> None:
        layers = []
        for j in range(len(sizes) - 1):
            act_cls = activation if j < len(sizes) - 2 else output_activation
            if isinstance(act_cls, nn.Module):
                args = {}
                for k in act_cls.__constants__:
                    args[k] = getattr(act_cls, k)
                act = act_cls.__class__(**args)
            else:
                act = act_cls()
            layers += [MultiLinear(sizes[j], sizes[j + 1], n_ch), act]
        super(MultiMlp, self).__init__(*layers)

        self.sizes: int = sizes
        self.n_ch: int = n_ch

    def reset_from_mlps(self, mlps):
        with torch.no_grad():
            for idx_layer in range(0, len(self), 2):
                for idx_ch in range(self.n_ch):
                    self[idx_layer].weight[idx_ch].copy_(mlps[idx_ch][idx_layer].weight.T)
                    self[idx_layer].bias[idx_ch, 0].copy_(mlps[idx_ch][idx_layer].bias)
