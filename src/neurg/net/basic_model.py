from torch import nn


def mlp(sizes, activation, output_activation=nn.Identity, bias=True):
    """Create a MLP with the given sizes and activations.
    Args:
        sizes (list[int]): The sizes of the layers.
        activation (nn.Module): The activation function to use for all layers except the last.
        output_activation (nn.Module): The activation function to use for the last layer."""
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
        layers += [nn.Linear(sizes[j], sizes[j + 1], bias=bias), act]
    return nn.Sequential(*layers)
