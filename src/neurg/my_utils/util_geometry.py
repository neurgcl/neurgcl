import numpy as np
import trimesh
from trimesh import primitives


def pad_bbox(bbox: np.ndarray, pad: float):
    bbox_padded = bbox.copy()
    bbox_padded[0] = bbox_padded[0] - pad
    bbox_padded[1] = bbox_padded[1] + pad
    return bbox_padded


def pad_bbox_by_ratio(bbox: np.ndarray, ratio: float):
    extents = bbox[1] - bbox[0]
    pad = extents*ratio
    bbox_padded = bbox.copy()
    bbox_padded[0] = bbox_padded[0] - pad
    bbox_padded[1] = bbox_padded[1] + pad
    return bbox_padded


def sample_volume_bbox(bbox: np.ndarray, n_pts: int):
    extents = bbox[1] - bbox[0]
    samples = np.random.random((n_pts, 3))
    samples = bbox[0]+samples*extents
    return samples


def sample_pt_sdfs_in_bbox(mesh, bbox, count):
    pts_sample = bbox.sample_volume(count)
    sdfs = -trimesh.proximity.signed_distance(mesh, pts_sample)
    ret = np.c_[pts_sample, sdfs]
    return ret


def bounds2bbox(bounds):
    transform = np.eye(4)
    # translate to center of axis aligned bounds
    transform[:3, 3] = bounds.mean(axis=0)

    aabb = primitives.Box(transform=transform,
                          extents=bounds[1]-bounds[0],
                          mutable=False)
    return aabb


def np_add_random(array: np.ndarray, extents):
    samples = np.random.random(array.shape)*2 - 1
    samples *= extents
    out = array + samples
    return out
