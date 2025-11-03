import torch

class NormPointSampler:
    def __init__(self, global_sigma, local_sigma=0.01):
        self.global_sigma = global_sigma
        self.local_sigma = local_sigma
        self.n_div = 8  # random sample one point every self.n_div points

    def get_points(self, pc_input, local_sigma=None):
        """
        sample point local and global
        """
        batch_size, dim = pc_input.shape
        n_local = batch_size // self.n_div
        # n_global = n_local

        device = pc_input.device
        dtype = pc_input.dtype

        if local_sigma is None:
            local_sigma = self.local_sigma
        else:
            local_sigma = local_sigma.unsqueeze(-1)

        idx_start = torch.randint(high=self.n_div, size=(1,))
        # sample local from Normal Distribution
        sample_local = (
            local_sigma * (torch.randn((n_local, dim), device=device, dtype=dtype) * 2 - 1)
            + pc_input[idx_start :: self.n_div]
        )
        # sample local from Uniform Distribution
        sample_global = self.global_sigma * (torch.rand((n_local, dim), device=device, dtype=dtype) * 2 - 1)
        sample = torch.cat([sample_local, sample_global], dim=0)
        return sample


if __name__ == "__main__":
    # 4090: cpu:0.001945288835268002 s, cuda:3.414950740989297e-05 s
    import time
    import numpy as np

    x = torch.rand(512000, 3, device="cuda")
    # x = torch.rand(512000, 3, device="cpu")
    sampler = NormPointSampler(2.0, 0.05)

    repeat = 200
    for i in range(repeat):
        t0 = time.perf_counter()
        pts = sampler.get_points(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

    l_t = []
    for i in range(repeat):
        t0 = time.perf_counter()
        pts = sampler.get_points(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        l_t.append(t1 - t0)
    print(pts.shape)
    print(np.array(l_t).mean())
