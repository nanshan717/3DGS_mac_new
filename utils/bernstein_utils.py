#
# Bernstein surface helpers for BR-GS.
#

import torch


def bernstein_basis(num_control_points, samples, device=None, dtype=torch.float32):
    """Return Bernstein basis values with shape [samples, num_control_points]."""
    if num_control_points < 2:
        raise ValueError("Bernstein surface needs at least two control points per axis.")

    degree = num_control_points - 1
    u = torch.linspace(0.0, 1.0, samples, device=device, dtype=dtype).clamp(0.0, 1.0)
    i = torch.arange(num_control_points, device=device, dtype=dtype)
    n = torch.tensor(float(degree), device=device, dtype=dtype)

    log_coeff = (
        torch.lgamma(n + 1.0)
        - torch.lgamma(i + 1.0)
        - torch.lgamma(n - i + 1.0)
    )
    coeff = torch.exp(log_coeff)
    return coeff[None, :] * (u[:, None] ** i[None, :]) * ((1.0 - u[:, None]) ** (n - i[None, :]))


def evaluate_bernstein_surface(control_points, samples_u=32, samples_v=32):
    """Evaluate a tensor-product Bernstein surface as [samples_u * samples_v, 3]."""
    if control_points.ndim != 3 or control_points.shape[-1] != 3:
        raise ValueError("control_points must have shape [num_u, num_v, 3].")

    basis_u = bernstein_basis(
        control_points.shape[0],
        samples_u,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    basis_v = bernstein_basis(
        control_points.shape[1],
        samples_v,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    surface = torch.einsum("ui,vj,ijc->uvc", basis_u, basis_v, control_points)
    return surface.reshape(-1, 3)


def _subsample_points(points, max_points):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    # Deterministic sampling keeps BSR loss stable across repeated debug runs.
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        max_points,
        device=points.device,
        dtype=torch.long,
    )
    return points[indices]


def bernstein_surface_distance_loss(
    points,
    control_points,
    point_mask=None,
    samples_u=32,
    samples_v=32,
    max_points=4096,
    chunk_size=2048,
):
    """Mean squared nearest distance from selected points to a sampled Bernstein surface."""
    if point_mask is not None:
        points = points[point_mask]
    finite_mask = torch.isfinite(points).all(dim=-1)
    points = points[finite_mask]
    points = _subsample_points(points, max_points)

    debug = {
        "num_bsr_points": int(points.shape[0]),
        "num_surface_points": int(samples_u * samples_v),
        "mean_distance": 0.0,
    }

    if points.shape[0] == 0:
        zero = control_points.sum() * 0.0
        return zero, debug

    surface_points = evaluate_bernstein_surface(control_points, samples_u, samples_v)
    min_dist_sq_chunks = []
    for chunk in torch.split(points, max(1, chunk_size), dim=0):
        dist_sq = torch.cdist(chunk, surface_points).square()
        min_dist_sq_chunks.append(dist_sq.min(dim=1).values)

    min_dist_sq = torch.cat(min_dist_sq_chunks, dim=0)
    loss = min_dist_sq.mean()
    debug["mean_distance"] = float(torch.sqrt(min_dist_sq.detach()).mean().item())
    return loss, debug
