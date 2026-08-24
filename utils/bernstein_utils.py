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
    """Evaluate one or more tensor-product surfaces as a flat point set."""
    if control_points.ndim == 3:
        control_points = control_points[None, None]
    if control_points.ndim != 5 or control_points.shape[-1] != 3:
        raise ValueError("control_points must have shape [U,V,3] or [Pu,Pv,U,V,3].")

    basis_u = bernstein_basis(
        control_points.shape[-3],
        samples_u,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    basis_v = bernstein_basis(
        control_points.shape[-2],
        samples_v,
        device=control_points.device,
        dtype=control_points.dtype,
    )
    surface = torch.einsum("ui,vj,abijc->abuvc", basis_u, basis_v, control_points)
    return surface.reshape(-1, 3)


def bernstein_control_regularization(control_points):
    """Second-difference energy and C0 seam continuity for a patch grid."""
    if control_points.ndim == 3:
        control_points = control_points[None, None]
    smooth_terms = []
    if control_points.shape[-3] >= 3:
        smooth_terms.append((control_points[..., 2:, :, :] - 2 * control_points[..., 1:-1, :, :] + control_points[..., :-2, :, :]).square().mean())
    if control_points.shape[-2] >= 3:
        smooth_terms.append((control_points[..., :, 2:, :] - 2 * control_points[..., :, 1:-1, :] + control_points[..., :, :-2, :]).square().mean())
    zero = control_points.sum() * 0.0
    smoothness = sum(smooth_terms, zero) / max(len(smooth_terms), 1)

    seam_terms = []
    if control_points.shape[0] > 1:
        seam_terms.append((control_points[:-1, :, -1, :, :] - control_points[1:, :, 0, :, :]).square().mean())
    if control_points.shape[1] > 1:
        seam_terms.append((control_points[:, :-1, :, -1, :] - control_points[:, 1:, :, 0, :]).square().mean())
    continuity = sum(seam_terms, zero) / max(len(seam_terms), 1)
    return smoothness, continuity


def _subsample_aligned(max_points, *tensors):
    first = tensors[0]
    if max_points <= 0 or first.shape[0] <= max_points:
        return tensors
    # Deterministic sampling keeps BSR loss stable across repeated debug runs.
    indices = torch.linspace(
        0,
        first.shape[0] - 1,
        max_points,
        device=first.device,
        dtype=torch.long,
    )
    return tuple(tensor[indices] if tensor is not None else None for tensor in tensors)


def _spatial_subsample(max_points, points, weights, opacities, frame_origin, frame_u, frame_v):
    if max_points <= 0 or points.shape[0] <= max_points:
        return points, weights, opacities
    with torch.no_grad():
        rel = points.detach() - frame_origin
        u, v = rel @ frame_u, rel @ frame_v
        u = (u - u.min()) / (u.max() - u.min()).clamp_min(1e-8)
        v = (v - v.min()) / (v.max() - v.min()).clamp_min(1e-8)
        # Sorting a fine 2-D cell key and sampling uniformly through that order
        # is deterministic, cheap, and avoids dependence on PLY/densification order.
        cells = 4096
        key = torch.floor(u * (cells - 1)).long() * cells + torch.floor(v * (cells - 1)).long()
        order = torch.argsort(key)
        take = torch.linspace(0, order.shape[0] - 1, max_points, device=points.device).long()
        indices = order[take]
    return points[indices], weights[indices], opacities[indices] if opacities is not None else None


def _density_weights(points, k=8):
    """Return [N] confidence weights that down-weight sparse isolated samples."""
    if k <= 0 or points.shape[0] <= 2:
        return torch.ones(points.shape[0], device=points.device, dtype=points.dtype)

    with torch.no_grad():
        detached = points.detach()
        dist = torch.cdist(detached, detached)
        knn = dist.topk(k=min(k + 1, detached.shape[0]), largest=False).values[:, 1:]
        mean_knn = knn.mean(dim=1)
        median_knn = torch.quantile(mean_knn, 0.5).clamp_min(1e-6)
        density = (median_knn / mean_knn.clamp_min(1e-6)).clamp(0.0, 2.0) * 0.5
        return density.to(dtype=points.dtype)


def _weighted_mean(values, weights):
    weights = weights.clamp_min(0.0)
    denom = weights.sum().clamp_min(1e-8)
    return (values * weights).sum() / denom


def bernstein_surface_distance_loss(
    points,
    control_points,
    point_mask=None,
    point_weights=None,
    opacities=None,
    samples_u=32,
    samples_v=32,
    max_points=4096,
    chunk_size=2048,
    robust_delta=0.1,
    density_k=8,
    density_blend=0.5,
    floater_lambda=0.0,
    floater_margin=0.0,
    support_scale=1.0,
    coverage_lambda=0.0,
    control_smoothness_lambda=0.0,
    patch_continuity_lambda=0.0,
    spatial_sampling=False,
    frame_origin=None,
    frame_u=None,
    frame_v=None,
    floater_points=None,
    floater_weights=None,
    floater_opacities=None,
):
    """Robust weighted nearest-distance loss from Gaussians to a sampled Bernstein surface."""
    if point_weights is None:
        point_weights = torch.ones(points.shape[0], device=points.device, dtype=points.dtype)
    if point_mask is not None:
        points = points[point_mask]
        point_weights = point_weights[point_mask]
        if opacities is not None:
            opacities = opacities[point_mask]

    finite_mask = torch.isfinite(points).all(dim=-1)
    points = points[finite_mask]
    point_weights = point_weights[finite_mask]
    if opacities is not None:
        opacities = opacities[finite_mask]

    positive_mask = point_weights > 0.0
    points = points[positive_mask]
    point_weights = point_weights[positive_mask]
    if opacities is not None:
        opacities = opacities[positive_mask]

    if spatial_sampling and all(x is not None for x in (frame_origin, frame_u, frame_v)):
        points, point_weights, opacities = _spatial_subsample(
            max_points, points, point_weights, opacities, frame_origin, frame_u, frame_v
        )
    else:
        points, point_weights, opacities = _subsample_aligned(max_points, points, point_weights, opacities)

    debug = {
        "num_bsr_points": int(points.shape[0]),
        "num_surface_points": int(samples_u * samples_v),
        "mean_distance": 0.0,
        "surface_loss": 0.0,
        "floater_loss": 0.0,
        "mean_weight": 0.0,
        "mean_density_weight": 0.0,
        "coverage_loss": 0.0,
        "control_smoothness_loss": 0.0,
        "patch_continuity_loss": 0.0,
    }

    if points.shape[0] == 0:
        zero = control_points.sum() * 0.0
        return zero, debug

    density_blend = float(max(0.0, min(1.0, density_blend)))
    density_weight = _density_weights(points, density_k)
    weights = point_weights * ((1.0 - density_blend) + density_blend * density_weight)

    surface_points = evaluate_bernstein_surface(control_points, samples_u, samples_v)
    min_dist_sq_chunks = []
    for chunk in torch.split(points, max(1, chunk_size), dim=0):
        dist_sq = torch.cdist(chunk, surface_points).square()
        min_dist_sq_chunks.append(dist_sq.min(dim=1).values)

    scale_sq = max(float(support_scale) ** 2, 1e-12)
    min_dist_sq = torch.cat(min_dist_sq_chunks, dim=0) / scale_sq
    distances = torch.sqrt(min_dist_sq + 1e-12)
    robust_delta = float(robust_delta)
    if robust_delta > 0.0:
        delta = torch.tensor(robust_delta, device=points.device, dtype=points.dtype)
        robust_values = torch.where(
            distances <= delta,
            0.5 * distances.square() / delta,
            distances - 0.5 * delta,
        )
    else:
        # robust_delta <= 0 restores the v1 squared-distance objective.
        robust_values = min_dist_sq

    surface_loss = _weighted_mean(robust_values, weights)
    floater_loss = surface_loss * 0.0
    if floater_lambda > 0.0:
        fp = floater_points if floater_points is not None else points
        fw = floater_weights if floater_weights is not None else point_weights
        fo = floater_opacities if floater_opacities is not None else opacities
        if fo is not None:
            valid = torch.isfinite(fp).all(dim=-1) & (fw > 0)
            fp, fw, fo = fp[valid], fw[valid], fo[valid]
            if fp.shape[0] == 0:
                fo = None
        if fo is not None:
            if spatial_sampling and all(x is not None for x in (frame_origin, frame_u, frame_v)):
                fp, fw, fo = _spatial_subsample(max_points, fp, fw, fo, frame_origin, frame_u, frame_v)
            else:
                fp, fw, fo = _subsample_aligned(max_points, fp, fw, fo)
            floater_dist_chunks = []
            for chunk in torch.split(fp, max(1, chunk_size), dim=0):
                floater_dist_chunks.append(torch.cdist(chunk, surface_points).square().min(dim=1).values)
            floater_distances = torch.sqrt(torch.cat(floater_dist_chunks) / scale_sq + 1e-12)
            opacity_values = fo.squeeze(-1) if fo.ndim > 1 else fo
        else:
            floater_distances = distances
            opacity_values = None
        if opacity_values is not None:
            with torch.no_grad():
                detached_distances = floater_distances.detach()
                if floater_margin > 0.0:
                    margin = torch.tensor(float(floater_margin), device=fp.device, dtype=fp.dtype)
                else:
                    median = torch.quantile(detached_distances, 0.5)
                    mad = torch.quantile(torch.abs(detached_distances - median), 0.5)
                    margin = (median + 2.0 * mad).clamp_min(1e-6)
                softness = (0.25 * margin).clamp_min(1e-6)
                far_confidence = torch.sigmoid((detached_distances - margin) / softness)
            floater_loss = _weighted_mean(opacity_values * far_confidence, fw)

    coverage_loss = surface_loss * 0.0
    if coverage_lambda > 0.0:
        coverage_chunks = []
        for chunk in torch.split(surface_points, max(1, chunk_size), dim=0):
            coverage_chunks.append(torch.cdist(chunk, points).square().min(dim=1).values)
        coverage_loss = torch.cat(coverage_chunks).mean() / scale_sq

    control_smoothness, patch_continuity = bernstein_control_regularization(control_points)
    control_smoothness = control_smoothness / scale_sq
    patch_continuity = patch_continuity / scale_sq
    loss = (
        surface_loss
        + float(coverage_lambda) * coverage_loss
        + float(control_smoothness_lambda) * control_smoothness
        + float(patch_continuity_lambda) * patch_continuity
        + float(floater_lambda) * floater_loss
    )
    debug["mean_distance"] = float(_weighted_mean(distances.detach(), weights.detach()).item())
    debug["surface_loss"] = float(surface_loss.detach().item())
    debug["floater_loss"] = float(floater_loss.detach().item())
    debug["mean_weight"] = float(point_weights.detach().mean().item())
    debug["mean_density_weight"] = float(density_weight.detach().mean().item())
    debug["coverage_loss"] = float(coverage_loss.detach().item())
    debug["control_smoothness_loss"] = float(control_smoothness.detach().item())
    debug["patch_continuity_loss"] = float(patch_continuity.detach().item())
    return loss, debug
