import numpy as np
import trimesh

from trimesh.constants import tol
from trimesh.proximity import closest_point, points_to_barycentric


def signed_distance_repeat(mesh, points, repeat=8):
    """
    Find the signed distance from a mesh to a list of points.

    * Points OUTSIDE the mesh will have NEGATIVE distance
    * Points within tol.merge of the surface will have POSITIVE distance
    * Points INSIDE the mesh will have POSITIVE distance

    Parameters
    -----------
    mesh : trimesh.Trimesh
      Mesh to query.
    points : (n, 3) float
      Points in space

    Returns
    ----------
    signed_distance : (n,) float
      Signed distance from point to mesh
    """
    # make sure we have a numpy array
    points = np.asanyarray(points, dtype=np.float64)

    # find the closest point on the mesh to the queried points
    closest, distance, triangle_id = closest_point(mesh, points)

    # we only care about nonzero distances
    nonzero = distance > tol.merge

    if not nonzero.any():
        return distance

    # For closest points that project directly in to the triangle, compute sign from
    # triangle normal Project each point in to the closest triangle plane
    nonzero = np.where(nonzero)[0]
    normals = mesh.face_normals[triangle_id]
    projection = (
        points[nonzero]
        - (
            normals[nonzero].T
            * np.einsum("ij,ij->i", points[nonzero] - closest[nonzero], normals[nonzero])
        ).T
    )

    # Determine if the projection lies within the closest triangle
    barycentric = points_to_barycentric(mesh.triangles[triangle_id[nonzero]], projection)
    ontriangle = ~(
        ((barycentric < -tol.merge) | (barycentric > 1 + tol.merge)).any(axis=1)
    )
    # Where projection does lie in the triangle, compare vector to projection to the
    # triangle normal to compute sign
    sign = np.sign(
        np.einsum(
            "ij,ij->i",
            normals[nonzero[ontriangle]],
            points[nonzero[ontriangle]] - projection[ontriangle],
        )
    )
    distance[nonzero[ontriangle]] *= -1.0 * sign

    # For all other triangles, resort to raycasting against the entire mesh
    pts = points[nonzero[~ontriangle]]
    inside = mesh.ray.contains_points(pts)
    # signs = (inside.astype(int) * 2) - 1.0

    signs = inside.astype(int)
    for _ in range(repeat):
        signs += mesh.ray.contains_points(pts).astype(int)
    signs = signs*2-1.0

    for j_repeat in range(3):
        flag_warn = np.abs(signs) < (repeat+j_repeat)/2+1
        has_warn = np.sum(flag_warn)
        if has_warn:
            signs_warn = np.zeros(np.sum(flag_warn), dtype=int)
            for _ in range(repeat):
                signs_warn += mesh.ray.contains_points(pts[signs_warn]).astype(int)
            signs_warn = signs_warn*2-1.0
            signs[flag_warn] += signs_warn
        else:
            break
    sign = np.sign(signs).astype(int)

    # apply sign to previously computed distance
    distance[nonzero[~ontriangle]] *= sign

    return distance

def cal_sdf(mesh, points, maxsz=8192, repeat=8):
    n_pts = len(points)
    step_num = n_pts // maxsz

    if n_pts % maxsz > 0:
        step_num += 1

    l_sdf = []
    for i in range(step_num):
        sdfs = -signed_distance_repeat(mesh, points[i * maxsz : (i + 1) * maxsz],repeat=repeat)
        l_sdf.append(sdfs)
    return np.concatenate(l_sdf)


def cal_sdf_raw(mesh, points, maxsz=16384):
    n_pts = len(points)
    step_num = n_pts // maxsz

    if n_pts % maxsz > 0:
        step_num += 1

    l_sdf = []
    for i in range(step_num):
        if i * maxsz>len(points):
            break
        idx2 = min((i + 1) * maxsz, len(points))
        sdfs = -trimesh.proximity.signed_distance(mesh, points[i * maxsz : idx2])
        l_sdf.append(sdfs)

    return np.concatenate(l_sdf)


def cal_sdf_grad_via_trimesh(mesh:trimesh.Trimesh, pts, thresh_surface=1e-5):
    """
    Args:
        mesh
        pts (n,3)
    Returns:
        grad (n,3) :
            sdf>thresh:             (x - x_closest) / |x - x_closest|
            -thresh<=sdf<=thresh:   surface normal
            sdf<-thresh:          - (x - x_closest) / |x - x_closest|
    """
    np_x = pts.astype(np.float64)
    closest, distance, triangle_id = trimesh.proximity.closest_point(mesh, np_x)

    # sdf = -trimesh.proximity.signed_distance(mesh, np_x)
    sdf = cal_sdf_raw(mesh, np_x)

    idx_non_surface = np.abs(sdf) > thresh_surface
    rel_pos = np_x[idx_non_surface] - closest[idx_non_surface]
    grads_gt_non_surface = rel_pos / np.linalg.norm(rel_pos, axis=-1)[:, np.newaxis]

    idx_surface = ~idx_non_surface
    grads_gt_surface = mesh.face_normals[triangle_id[idx_surface]]

    flag_inside = (sdf < 0) & (idx_non_surface)
    grads_all = np.zeros((len(np_x), 3))
    grads_all[idx_non_surface] = grads_gt_non_surface
    grads_all[flag_inside] = -grads_all[flag_inside]
    grads_all[idx_surface] = grads_gt_surface
    return grads_all