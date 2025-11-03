import copy
import multiprocessing
import os
from pathlib import Path
import random
from functools import partial

import _pickle as pickle
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from tqdm import tqdm
from trimesh.primitives import Box
from copy import deepcopy
from neurg.my_utils.config import PATH_ROOT
from neurg.my_utils.util_file import load_json, write_json
from neurg.my_utils.util_geometry import bounds2bbox, np_add_random
from neurg.my_utils.util_log import get_time_str
from neurg.my_utils.sdf.util_sdf import cal_sdf, cal_sdf_raw
from neurg.my_utils.util_path import SPath
from neurg.utils.data import DatasetCfg

N_REPEAT_SAMPLE = 5000

N_INSIDE = 20 * N_REPEAT_SAMPLE  # d<0
N_OUTSIDE = 30 * N_REPEAT_SAMPLE  # d>0 in bbox * 1.4

N_ZERO = 10 * N_REPEAT_SAMPLE  # on surface
N_NEAR = 20 * N_REPEAT_SAMPLE  # near surface (+- 0.05)

# N_BBOX_PADFIXED = 10 * N_REPEAT_SAMPLE  # bbox + 0.4
N_BBOX_LARGEST = 20 * N_REPEAT_SAMPLE  # Largest bbox(*1.4, +0.4)


def sample_bbox_insur(mesh, bbox: Box, N_INSIDE):
    # print(f"sample_bbox_insur: {N_INSIDE}")
    l_pts = []
    l_sdfs = []
    rest_len = N_INSIDE

    while rest_len > 0:
        pts_sample = bbox.sample_volume(max(100, int(rest_len * 1.1)))

        sdfs = cal_sdf(mesh, pts_sample)
        sel_in = sdfs < 0

        tmp_pts = pts_sample[sel_in][:rest_len]
        l_pts.append(tmp_pts)
        l_sdfs.append(sdfs[sel_in][:rest_len])

        rest_len -= len(tmp_pts)

    # print(f"sample_bbox_insur: {N_INSIDE} done.")
    return np.concatenate(l_pts), np.concatenate(l_sdfs)


def sample_bbox_outsur(mesh, bbox: Box, N_OUTSIDE):
    # print(f"sample_bbox_outsur: {N_OUTSIDE}")
    l_pts = []
    l_sdfs = []
    rest_len = N_OUTSIDE

    while rest_len > 0:
        pts_sample = bbox.sample_volume(max(100, int(rest_len * 1.1)))

        sdfs = cal_sdf(mesh, pts_sample)
        sel_out = sdfs > 0

        tmp_pts = pts_sample[sel_out][:rest_len]
        l_pts.append(tmp_pts)
        l_sdfs.append(sdfs[sel_out][:rest_len])

        rest_len -= len(tmp_pts)

    # print(f"sample_bbox_outsur: {N_OUTSIDE} done.")
    return np.concatenate(l_pts), np.concatenate(l_sdfs)


def sample_bbox_inout(mesh, bbox: Box, bbox_pad: Box, N_INSIDE, N_OUTSIDE):
    # print(f"sample_bbox_outsur: {N_OUTSIDE}")
    l_pts_in = []
    l_sdfs_in = []
    l_pts_out = []
    l_sdfs_out = []

    rest_len_in = N_INSIDE
    rest_len_out = N_OUTSIDE
    rest_len = rest_len_in + rest_len_out

    while rest_len > 0:
        if rest_len_out > 0:
            pts_sample = bbox_pad.sample_volume(max(100, int(rest_len * 1.1)))
        elif rest_len_in > 0:
            pts_sample = bbox.sample_volume(max(100, int(rest_len * 1.1)))

        sdfs = cal_sdf(mesh, pts_sample)
        if rest_len_in > 0:
            sel_in = sdfs < 0
            tmp_pts_in = pts_sample[sel_in][:rest_len_in]
            l_pts_in.append(deepcopy(tmp_pts_in))
            l_sdfs_in.append(deepcopy(sdfs[sel_in][:rest_len_in]))
            rest_len_in -= len(tmp_pts_in)

        if rest_len_out > 0:
            sel_out = sdfs > 0
            tmp_pts_out = pts_sample[sel_out][:rest_len_out]
            l_pts_out.append(deepcopy(tmp_pts_out))
            l_sdfs_out.append(deepcopy(sdfs[sel_out][:rest_len_out]))
            rest_len_out -= len(tmp_pts_out)

        rest_len = rest_len_in + rest_len_out
    # print(f"sample_bbox_outsur: {N_OUTSIDE} done.")
    return (
        np.c_[np.concatenate(l_pts_in), np.concatenate(l_sdfs_in)],
        np.c_[np.concatenate(l_pts_out), np.concatenate(l_sdfs_out)],
    )


def sample_near_sur(mesh, xyz_random):
    pts_near_sur_raw, _ = trimesh.sample.sample_surface(mesh, N_NEAR)
    pts_near_sur = np_add_random(pts_near_sur_raw, xyz_random)
    sdfs = cal_sdf_raw(mesh, pts_near_sur)
    return np.array(pts_near_sur), np.array(sdfs)


def sample_one_link(link_file, out_dir, file_idx=0, seed=None):
    seed = seed or random.randint(0, 2**32 - 1)
    np.random.seed(seed)

    link_name = os.path.splitext(os.path.split(link_file)[-1])[0]
    print(f"--f {link_name}, seed={seed}: {link_file}")
    # print(f"Current link: {link_name}")

    mesh = trimesh.load(link_file)

    def show(*args, **kwargs):
        print(f"[{link_name}]", *args, **kwargs)

    # is the current mesh watertight?
    show(f"mesh.is_watertight: {mesh.is_watertight}")

    # bounds = mesh.bounds
    # print(f"bounds: \n{bounds}")

    bbox_raw = mesh.bounding_box
    # print(bbox_raw.bounds)
    bbox_raw = Box(extents=bbox_raw.extents, transform=bbox_raw.transform)
    assert (bbox_raw.bounds == mesh.bounding_box.bounds).all()
    # print(bbox_raw.bounds)
    bbox_pad_ratio = copy.deepcopy(bbox_raw)
    bbox_pad_ratio.primitive.extents *= 1.4
    # print(bbox_pad_ratio.bounds)

    bbox_pad_fixed = copy.deepcopy(bbox_raw)
    bbox_pad_fixed.primitive.extents += 0.4
    # print(bbox_pad_fixed.bounds)

    np_bbox_large = np.array(
        [
            np.minimum(bbox_pad_ratio.bounds[0], bbox_pad_fixed.bounds[0]),
            np.maximum(bbox_pad_ratio.bounds[1], bbox_pad_fixed.bounds[1]),
        ]
    )
    bbox_largest = bounds2bbox(np_bbox_large)

    bbox_center = bbox_raw.centroid

    dic_data = {}
    dic_data['bbox_largest'] = bbox_largest.bounds

    show(f"Start sampling...")

    # Sample ------------------------------------------------------
    # sample balanced
    if file_idx == 0:
        show(f"Start sampling pts_inside,pts_outside ...")

    dic_data['pts_inside'], dic_data['pts_outside'] = sample_bbox_inout(
        mesh, bbox_raw, bbox_pad_ratio, N_INSIDE, N_OUTSIDE
    )

    # largest bbox
    if file_idx == 0:
        show(f"Start sampling pts_bbox_largest ...")
    pts_bbox_largest = bbox_largest.sample_volume(N_BBOX_LARGEST)
    sdf_bbox_largest = cal_sdf(mesh, pts_bbox_largest)
    dic_data['pts_bbox_largest'] = np.c_[pts_bbox_largest, sdf_bbox_largest]
    del pts_bbox_largest, sdf_bbox_largest

    # on surface
    if file_idx == 0:
        show(f"Start sampling pts_on_sur ...")
    pts_on_sur, face_index = trimesh.sample.sample_surface(mesh, N_ZERO)
    sdf_on_sur = np.zeros((pts_on_sur.shape[0]))
    dic_data['pts_on_sur'] = np.c_[pts_on_sur, sdf_on_sur]
    del pts_on_sur, sdf_on_sur
    dic_data['pts_on_sur_idx'] = face_index

    # near surface
    if file_idx == 0:
        show(f"Start sampling pts_near_sur ...")
    dic_data['pts_near_sur'] = np.c_[sample_near_sur(mesh, 0.05)]

    # Output ------------------------------------------------------
    if file_idx == 0:
        show(f"Prepare output ...")

    dir_link = os.path.join(out_dir, f'{link_name}')
    os.makedirs(dir_link, exist_ok=True)

    with open(os.path.join(dir_link, f'{link_name}.pkl'), 'wb') as out_file:
        pickle.dump(dic_data, out_file)

    json_info = {k: dic_data[k] for k in dic_data.keys() if not k.startswith('pts_')}
    for k in json_info:
        v = json_info[k]
        if isinstance(v, np.ndarray):
            json_info[k] = v.tolist()
        if isinstance(v, Path):
            json_info[k] = str(v)
    write_json(os.path.join(dir_link, f'info.json'), json_info)

    l_pts = [dic_data[k] for k in dic_data.keys() if k != 'pts_on_sur_idx' and k.startswith('pts_')]
    del dic_data

    pts_all = np.concatenate(l_pts, axis=0)
    del l_pts
    d_final = pts_all

    plt.hist(d_final[:, -1])
    plt.title(f"sdf hist - {link_name}")
    plt.xlabel("dist")
    plt.savefig(os.path.join(dir_link, f"sdf_hist.png"))
    plt.close()

    show(f"Done.")


if __name__ == '__main__':
    import argparse
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, default=None)
    parser.add_argument('--mesh_path', type=str, default=PATH_ROOT / 'data/raw/ros_mesh/collision/hand.stl')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--cpus', type=int, default=1)
    args = parser.parse_args()
    version = get_time_str()

    mesh_dir = args.mesh_dir
    if mesh_dir is not None:
        # args.cpus = min(4, args.cpus)
        use_multiprocess = args.cpus > 1
        version = get_time_str()

        cfg = load_json(mesh_dir+"/config.json")
        link_files = [os.path.join(mesh_dir, fname) for fname in cfg["link_meshs"]]

        out_dir = args.out_dir or PATH_ROOT / f'data/dataset/raw/links_{version}'
        out_dir = SPath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        # save info -----------------------------------------------
        cfg_path = out_dir / 'config.json'
        ds_cfg = DatasetCfg(version=version, mesh_dir=mesh_dir, link_meshs=cfg["link_meshs"], N_REPEAT_SAMPLE=N_REPEAT_SAMPLE)
        ds_cfg.write_json(cfg_path)
        print(f"save config to {cfg_path}")

        t0 = time.time()
        if use_multiprocess:
            import multiprocessing

            from tqdm.contrib.concurrent import process_map

            max_workers = min(args.cpus, multiprocessing.cpu_count() // 2 - 1)
            print(f"max_workers: {max_workers}")
            r = process_map(partial(sample_one_link, out_dir=out_dir), link_files, max_workers=max_workers)
        else:
            for i in tqdm(range(0, len(link_files))):
                sample_one_link(link_files[i], out_dir=out_dir)
        t1 = time.time()
        print(f"Time: {t1 - t0:.3f}s")

    else:
        out_dir = args.out_dir or (PATH_ROOT / f'data/dataset/raw/single_{version}')

        t0 = time.time()
        sample_one_link(args.mesh_path, out_dir=out_dir)
        t1 = time.time()
        print(f"Time: {t1 - t0:.3f}s")
