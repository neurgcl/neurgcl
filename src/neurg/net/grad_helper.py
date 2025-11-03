import torch
from typing import List, Optional


def cal_grad(y: torch.Tensor, x: torch.Tensor, retain_graph: bool = False):
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones(y.shape, device=x.device)]
    grads = torch.autograd.grad(
        outputs=[y],
        inputs=[x],
        grad_outputs=grad_outputs,
        create_graph=retain_graph,
        retain_graph=retain_graph,
    )[0]
    if grads is None:
        raise ValueError("Gradient is of type None")
    return grads
