import open3d as o3d
import numpy as np
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from typing import Optional, Callable, Any, Tuple

def pairwise_distance(
    x: torch.Tensor, y: torch.Tensor, normalized: bool = False, channel_first: bool = False
) -> torch.Tensor:
    r"""Pairwise distance of two (batched) point clouds.

    Args:
        x (Tensor): (*, N, C) or (*, C, N)
        y (Tensor): (*, M, C) or (*, C, M)
        normalized (bool=False): if the points are normalized, we have "x2 + y2 = 1", so "d2 = 2 - 2xy".
        channel_first (bool=False): if True, the points shape is (*, C, N).

    Returns:
        dist: torch.Tensor (*, N, M)
    """
    if channel_first:
        channel_dim = -2
        xy = torch.matmul(x.transpose(-1, -2), y)  # [(*, C, N) -> (*, N, C)] x (*, C, M)
    else:
        channel_dim = -1
        xy = torch.matmul(x, y.transpose(-1, -2))  # (*, N, C) x [(*, M, C) -> (*, C, M)]
    if normalized:
        sq_distances = 2.0 - 2.0 * xy
    else:
        x2 = torch.sum(x ** 2, dim=channel_dim).unsqueeze(-1)  # (*, N, C) or (*, C, N) -> (*, N) -> (*, N, 1)
        y2 = torch.sum(y ** 2, dim=channel_dim).unsqueeze(-2)  # (*, M, C) or (*, C, M) -> (*, M) -> (*, 1, M)
        sq_distances = x2 - 2 * xy + y2
    sq_distances = sq_distances.clamp(min=0.0)
    return sq_distances


def apply_transform(points: torch.Tensor, transform: torch.Tensor, normals: Optional[torch.Tensor] = None):
    r"""Rigid transform to points and normals (optional).

    Given a point cloud P(3, N), normals V(3, N) and a transform matrix T in the form of
      | R t |
      | 0 1 |,
    the output point cloud Q = RP + t, V' = RV.

    In the implementation, P and V are (N, 3), so R should be transposed: Q = PR^T + t, V' = VR^T.

    There are two cases supported:
    1. points and normals are (*, 3), transform is (4, 4), the output points are (*, 3).
       In this case, the transform is applied to all points.
    2. points and normals are (B, N, 3), transform is (B, 4, 4), the output points are (B, N, 3).
       In this case, the transform is applied batch-wise. The points can be broadcast if B=1.

    Args:
        points (Tensor): (*, 3) or (B, N, 3)
        normals (optional[Tensor]=None): same shape as points.
        transform (Tensor): (4, 4) or (B, 4, 4)

    Returns:
        points (Tensor): same shape as points.
        normals (Tensor): same shape as points.
    """
    if normals is not None:
        assert points.shape == normals.shape
    if transform.ndim == 2:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        points_shape = points.shape
        points = points.reshape(-1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        points = points.reshape(*points_shape)
        if normals is not None:
            normals = normals.reshape(-1, 3)
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
            normals = normals.reshape(*points_shape)
    elif transform.ndim == 3 and points.ndim == 3:
        rotation = transform[:, :3, :3]  # (B, 3, 3)
        translation = transform[:, None, :3, 3]  # (B, 1, 3)
        points = torch.matmul(points, rotation.transpose(-1, -2)) + translation
        if normals is not None:
            normals = torch.matmul(normals, rotation.transpose(-1, -2))
    else:
        raise ValueError(
            'Incompatible shapes between points {} and transform {}.'.format(
                tuple(points.shape), tuple(transform.shape)
            )
        )
    if normals is not None:
        return points, normals
    else:
        return points


def get_rotation_translation_from_transform(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Get rotation matrix and translation vector from rigid transform matrix.

    Args:
        transform (array): (4, 4)

    Returns:
        rotation (array): (3, 3)
        translation (array): (3,)
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return rotation, translation


def compute_relative_rotation_error(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotation (array): ground truth rotation matrix (3, 3)
        est_rotation (array): estimated rotation matrix (3, 3)

    Returns:
        rre (float): relative rotation error.
    """
    x = 0.5 * (np.trace(np.matmul(est_rotation.T, gt_rotation)) - 1.0)
    x = np.clip(x, -1.0, 1.0)
    x = np.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def compute_relative_translation_error(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute the isotropic Relative Translation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translation (array): ground truth translation vector (3,)
        est_translation (array): estimated translation vector (3,)

    Returns:
        rte (float): relative translation error.
    """
    return np.linalg.norm(gt_translation - est_translation)


def compute_registration_error(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transform (array): ground truth transformation matrix (4, 4)
        est_transform (array): estimated transformation matrix (4, 4)

    Returns:
        rre (float): relative rotation error.
        rte (float): relative translation error.
    """
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    rre = compute_relative_rotation_error(gt_rotation, est_rotation)
    rte = compute_relative_translation_error(gt_translation, est_translation)
    return rre, rte


def compute_rotation_mse_and_mae(gt_rotation: np.ndarray, est_rotation: np.ndarray):
    r"""Compute anisotropic rotation error (MSE and MAE)."""
    gt_euler_angles = Rotation.from_dcm(gt_rotation).as_euler('xyz', degrees=True)  # (3,)
    est_euler_angles = Rotation.from_dcm(est_rotation).as_euler('xyz', degrees=True)  # (3,)
    mse = np.mean((gt_euler_angles - est_euler_angles) ** 2)
    mae = np.mean(np.abs(gt_euler_angles - est_euler_angles))
    return mse, mae


def compute_translation_mse_and_mae(gt_translation: np.ndarray, est_translation: np.ndarray):
    r"""Compute anisotropic translation error (MSE and MAE)."""
    mse = np.mean((gt_translation - est_translation) ** 2)
    mae = np.mean(np.abs(gt_translation - est_translation))
    return mse, mae


def compute_transform_mse_and_mae(gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute anisotropic rotation and translation error (MSE and MAE)."""
    gt_rotation, gt_translation = get_rotation_translation_from_transform(gt_transform)
    est_rotation, est_translation = get_rotation_translation_from_transform(est_transform)
    r_mse, r_mae = compute_rotation_mse_and_mae(gt_rotation, est_rotation)
    t_mse, t_mae = compute_translation_mse_and_mae(gt_translation, est_translation)
    return r_mse, r_mae, t_mse, t_mae


def compute_registration_rmse(src_points: np.ndarray, gt_transform: np.ndarray, est_transform: np.ndarray):
    r"""Compute re-alignment error (approximated RMSE in 3DMatch).

    Used in Rotated 3DMatch.

    Args:
        src_points (array): source point cloud. (N, 3)
        gt_transform (array): ground-truth transformation. (4, 4)
        est_transform (array): estimated transformation. (4, 4)

    Returns:
        error (float): root mean square error.
    """
    gt_points = apply_transform(src_points, gt_transform)
    est_points = apply_transform(src_points, est_transform)
    error = np.linalg.norm(gt_points - est_points, axis=1).mean()
    return error


def modified_chamfer_distance(raw_points, ref_points, src_points, gt_transform, transform, reduction='mean'):
    r"""Compute the modified chamfer distance.

    Args:
        raw_points (Tensor): (B, N_raw, 3)
        ref_points (Tensor): (B, N_ref, 3)
        src_points (Tensor): (B, N_src, 3)
        gt_transform (Tensor): (B, 4, 4)
        transform (Tensor): (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        chamfer_distance
    """
    assert reduction in ['mean', 'sum', 'none']

    # P_t -> Q_raw
    aligned_src_points = apply_transform(src_points, transform)  # (B, N_src, 3)
    sq_dist_mat_p_q = pairwise_distance(aligned_src_points, raw_points)  # (B, N_src, N_raw)
    nn_sq_distances_p_q = sq_dist_mat_p_q.min(dim=-1)[0]  # (B, N_src)
    chamfer_distance_p_q = torch.sqrt(nn_sq_distances_p_q).mean(dim=-1)  # (B)

    # Q -> P_raw
    composed_transform = torch.matmul(transform, torch.inverse(gt_transform))  # (B, 4, 4)
    aligned_raw_points = apply_transform(raw_points, composed_transform)  # (B, N_raw, 3)
    sq_dist_mat_q_p = pairwise_distance(ref_points, aligned_raw_points)  # (B, N_ref, N_raw)
    nn_sq_distances_q_p = sq_dist_mat_q_p.min(dim=-1)[0]  # (B, N_ref)
    chamfer_distance_q_p = torch.sqrt(nn_sq_distances_q_p).mean(dim=-1)  # (B)

    # sum up
    chamfer_distance = chamfer_distance_p_q + chamfer_distance_q_p  # (B)

    if reduction == 'mean':
        chamfer_distance = chamfer_distance.mean()
    elif reduction == 'sum':
        chamfer_distance = chamfer_distance.sum()
    return chamfer_distance


def relative_rotation_error(gt_rotations, rotations):
    r"""Isotropic Relative Rotation Error.

    RRE = acos((trace(R^T \cdot \bar{R}) - 1) / 2)

    Args:
        gt_rotations (Tensor): ground truth rotation matrix (*, 3, 3)
        rotations (Tensor): estimated rotation matrix (*, 3, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    mat = torch.matmul(rotations.transpose(-1, -2), gt_rotations)
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    x = 0.5 * (trace - 1.0)
    x = x.clamp(min=-1.0, max=1.0)
    x = torch.arccos(x)
    rre = 180.0 * x / np.pi
    return rre


def relative_translation_error(gt_translations, translations):
    r"""Isotropic Relative Rotation Error.

    RTE = \lVert t - \bar{t} \rVert_2

    Args:
        gt_translations (Tensor): ground truth translation vector (*, 3)
        translations (Tensor): estimated translation vector (*, 3)

    Returns:
        rre (Tensor): relative rotation errors (*)
    """
    rte = torch.linalg.norm(gt_translations - translations, dim=-1)
    return rte


def isotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the isotropic Relative Rotation Error and Relative Translation Error.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (*, 4, 4)
        transforms (Tensor): estimated transformation matrix (*, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        rre (Tensor): relative rotation error.
        rte (Tensor): relative translation error.
    """
    assert reduction in ['mean', 'sum', 'none']

    gt_rotations, gt_translations = get_rotation_translation_from_transform(gt_transforms)
    rotations, translations = get_rotation_translation_from_transform(transforms)

    rre = relative_rotation_error(gt_rotations, rotations)  # (*)
    rte = relative_translation_error(gt_translations, translations)  # (*)

    if reduction == 'mean':
        rre = rre.mean()
        rte = rte.mean()
    elif reduction == 'sum':
        rre = rre.sum()
        rte = rte.sum()

    return rre, rte


def anisotropic_transform_error(gt_transforms, transforms, reduction='mean'):
    r"""Compute the anisotropic Relative Rotation Error and Relative Translation Error.

    This function calls numpy-based implementation to achieve batch-wise computation and thus is non-differentiable.

    Args:
        gt_transforms (Tensor): ground truth transformation matrix (B, 4, 4)
        transforms (Tensor): estimated transformation matrix (B, 4, 4)
        reduction (str='mean'): reduction method, 'mean', 'sum' or 'none'

    Returns:
        r_mse (Tensor): rotation mse.
        r_mae (Tensor): rotation mae.
        t_mse (Tensor): translation mse.
        t_mae (Tensor): translation mae.
    """
    assert reduction in ['mean', 'sum', 'none']

    batch_size = gt_transforms.shape[0]
    gt_transforms_array = gt_transforms.detach().cpu().numpy()
    transforms_array = transforms.detach().cpu().numpy()

    all_r_mse = []
    all_r_mae = []
    all_t_mse = []
    all_t_mae = []
    for i in range(batch_size):
        r_mse, r_mae, t_mse, t_mae = compute_transform_mse_and_mae(gt_transforms_array[i], transforms_array[i])
        all_r_mse.append(r_mse)
        all_r_mae.append(r_mae)
        all_t_mse.append(t_mse)
        all_t_mae.append(t_mae)
    r_mse = torch.as_tensor(all_r_mse).to(gt_transforms)
    r_mae = torch.as_tensor(all_r_mae).to(gt_transforms)
    t_mse = torch.as_tensor(all_t_mse).to(gt_transforms)
    t_mae = torch.as_tensor(all_t_mae).to(gt_transforms)

    if reduction == 'mean':
        r_mse = r_mse.mean()
        r_mae = r_mae.mean()
        t_mse = t_mse.mean()
        t_mae = t_mae.mean()
    elif reduction == 'sum':
        r_mse = r_mse.sum()
        r_mae = r_mae.sum()
        t_mse = t_mse.sum()
        t_mae = t_mae.sum()

    return r_mse, r_mae, t_mse, t_mae
