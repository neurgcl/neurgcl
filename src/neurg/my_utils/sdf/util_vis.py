import numpy as np
import skimage
import torch
import trimesh


def sdf_to_mesh(sdf_func, offset, scale, nbData=128, device="cuda"):
    domain_min = -1
    domain_max = 1
    n_pt = nbData + 1
    domain = torch.linspace(domain_min, domain_max, n_pt)
    grid_x, grid_y, grid_z = torch.meshgrid(domain, domain, domain)
    grid_x, grid_y, grid_z = grid_x.reshape(-1, 1), grid_y.reshape(-1, 1), grid_z.reshape(-1, 1)
    p = torch.cat([grid_x, grid_y, grid_z], dim=1).float().to(device)

    p = p * scale + offset
    # split data to deal with memory issues
    d = []
    batch_sz = 65536
    for i in range(0, len(p), batch_sz):
        p_s = p[i : i + batch_sz]
        d_s = sdf_func(p_s)
        d.append(d_s)
    d = torch.cat(d, dim=0)
    d = d / scale

    verts, faces, normals, values = skimage.measure.marching_cubes(
        d.view(n_pt, n_pt, n_pt).detach().cpu().numpy(),
        level=0.0,
        spacing=np.array([(domain_max - domain_min) / (nbData)] * 3),
    )
    verts = (verts - [1, 1, 1]) * scale + offset.cpu().numpy()
    mesh = trimesh.Trimesh(verts, faces)
    return mesh
