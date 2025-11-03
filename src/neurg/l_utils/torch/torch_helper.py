import torch


def custom_repr(self):
    # return f'[{tuple(self.shape)}] {original_repr(self)}'
    if len(self.shape) == 0:
        return f"[{self.device.type}; {self.dtype}] {self.item()}"
    else:
        return f"[{'x'.join([str(x) for x in self.shape])}; {self.device.type}; {self.dtype}] {original_repr(self)}"


original_repr = torch.Tensor.__repr__


def custom_torch_repr():
    torch.Tensor.__repr__ = custom_repr


def try_cuda():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


if __name__ == "__main__":
    custom_torch_repr()
    a = torch.tensor([1, 2, 3]).to("cuda")
    print(a)
