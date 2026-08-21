import math

import torch
import torch.nn.functional as F

from stable_worldmodel.wm.loss import VISRegLoss


def _reference_visreg(
    z: torch.Tensor,
    *,
    num_projections: int,
    lambda_scale: float,
    lambda_shape: float,
    lambda_center: float,
) -> torch.Tensor:
    _, batch_size, embedding_dim = z.shape

    mean = z.mean(dim=1, keepdim=True)
    center_loss = mean.pow(2).mean()

    centered = z - mean
    std = centered.norm(dim=1).div(math.sqrt(batch_size)).clamp_min(1e-6)
    scale_loss = (std - 1.0).pow(2).mean()

    normalized = centered / std.detach().unsqueeze(1)
    projections = F.normalize(
        torch.randn(
            embedding_dim,
            num_projections,
            device=z.device,
            dtype=z.dtype,
        ),
        dim=0,
    )
    projected = (normalized @ projections).sort(dim=1).values
    quantiles = torch.linspace(
        1,
        batch_size,
        batch_size,
        device=z.device,
        dtype=torch.float32,
    ) / (batch_size + 1)
    target = (
        torch.erfinv(2 * quantiles - 1)
        .mul_(math.sqrt(2))
        .view(1, batch_size, 1)
    )
    shape_loss = (projected - target).pow(2).mean()

    return (
        lambda_scale * scale_loss
        + lambda_shape * shape_loss
        + lambda_center * center_loss
    )


def test_visreg_matches_pinned_upstream_formula() -> None:
    generator = torch.Generator().manual_seed(17)
    embeddings = torch.randn(
        3,
        11,
        7,
        generator=generator,
        dtype=torch.float64,
    )
    kwargs = {
        'num_projections': 13,
        'lambda_scale': 1.7,
        'lambda_shape': 0.6,
        'lambda_center': 0.2,
    }
    loss_fn = VISRegLoss(**kwargs)

    torch.manual_seed(29)
    actual = loss_fn(embeddings)
    torch.manual_seed(29)
    expected = _reference_visreg(embeddings, **kwargs)

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_visreg_keeps_quantile_target_in_fp32_under_bf16() -> None:
    loss_fn = VISRegLoss(num_projections=7)
    embeddings = torch.randn(2, 17, 8, dtype=torch.bfloat16)

    target = loss_fn._get_target(embeddings.shape[1], embeddings.device)
    torch.manual_seed(41)
    loss = loss_fn(embeddings)

    assert target.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_visreg_has_finite_nonzero_gradient_near_collapse() -> None:
    generator = torch.Generator().manual_seed(31)
    embeddings = (
        1e-4
        * torch.randn(
            4,
            16,
            8,
            generator=generator,
            dtype=torch.float64,
        )
    ).requires_grad_()
    loss_fn = VISRegLoss(num_projections=17)

    torch.manual_seed(37)
    loss = loss_fn(embeddings)
    loss.backward()

    assert torch.isfinite(loss)
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert embeddings.grad.norm() > 0
