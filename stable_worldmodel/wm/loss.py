import math

import torch
import torch.nn.functional as F
from einops import einsum


class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Warning: This version only support single-gpu.
    Reference: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer('t', t)
        self.register_buffer('phi', window)
        self.register_buffer('weights', weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        statistic = self._projected_statistic(proj, A)
        return statistic.mean()  # average over projections and time

    def _projected_statistic(self, proj, projections):
        """Return the Epps-Pulley statistic for each leading slice."""

        # compute the epps-pulley statistic
        x_t = (proj @ projections).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(
            -3
        ).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic


class ConditionalSIGReg(SIGReg):
    """SIGReg on condition-matched representation contrasts.

    Vanilla SIGReg constrains every time-indexed marginal population
    ``z[t]``.  That population can remain Gaussian even when two futures
    belonging to the same visible query/action condition are mapped to the
    same representation.  This module replaces selected marginal slices by
    their condition-matched Haar high-pass contrasts

    ``(z[t, i] - z[t, j]) / sqrt(2)``.

    For two independent standard Gaussian representations, this contrast is
    itself standard Gaussian.  Pair collapse instead creates a zero
    population, which the same SIGReg statistic detects directly.  This
    deliberately tests the conditional high-pass population alone at active
    time indices: a complete invertible Haar transform is insufficient,
    because its low-pass population can compensate for a collapsed
    high-pass direction in the marginal sketch.

    The implementation uses the same random feature projections,
    characteristic-function statistic, Gaussian target, and single scalar
    loss as SIGReg.  No rule or class labels are consumed: callers provide
    disjoint pairs that can be constructed from visible conditioning
    variables.  With no pairs, the implementation calls :class:`SIGReg`
    exactly.

    Args:
        knots: Number of characteristic-function integration knots.
        num_proj: Number of random feature projections.
        randomize_pair_orientation: Multiply every contrast row by an
            independent Rademacher sign.  This makes the expected objective
            invariant to the arbitrary ordering inside each pair.
    """

    def __init__(
        self,
        knots=17,
        num_proj=1024,
        randomize_pair_orientation=True,
    ):
        super().__init__(knots=knots, num_proj=num_proj)
        self.randomize_pair_orientation = randomize_pair_orientation

    @staticmethod
    def _validate_pairs(proj, pairs, active):
        if proj.dim() != 3:
            raise ValueError(
                "ConditionalSIGReg expects proj with shape (T, B, D), "
                f"got {tuple(proj.shape)}"
            )
        if pairs.dim() != 2 or pairs.size(-1) != 2:
            raise ValueError(
                "pairs must have shape (P, 2), "
                f"got {tuple(pairs.shape)}"
            )
        if pairs.dtype != torch.long:
            raise TypeError(f"pairs must use torch.long, got {pairs.dtype}")
        if pairs.numel() and (
            int(pairs.min()) < 0 or int(pairs.max()) >= proj.size(1)
        ):
            raise ValueError(
                "pairs contain an index outside the representation batch"
            )
        flattened = pairs.flatten()
        if torch.unique(flattened).numel() != flattened.numel():
            raise ValueError("pairs must be disjoint")
        expected_active_shape = (proj.size(0), pairs.size(0))
        if tuple(active.shape) != expected_active_shape:
            raise ValueError(
                "active must have shape (T, P), "
                f"expected {expected_active_shape}, got {tuple(active.shape)}"
            )
        if active.dtype != torch.bool:
            raise TypeError(f"active must use torch.bool, got {active.dtype}")

    def forward(self, proj, pairs=None, active=None):
        """
        Args:
            proj: Representations with shape ``(T, B, D)``.
            pairs: Disjoint condition-matched indices with shape ``(P, 2)``.
            active: Boolean mask with shape ``(T, P)``.  At a time index with
                any active pairs, SIGReg is evaluated on the selected pair
                contrasts instead of the unconditional batch marginal.
        """

        if pairs is None:
            if active is not None:
                raise ValueError("active requires pairs")
            return super().forward(proj)

        pairs = pairs.to(device=proj.device)
        if active is None:
            active = torch.ones(
                proj.size(0),
                pairs.size(0),
                dtype=torch.bool,
                device=proj.device,
            )
        else:
            active = active.to(device=proj.device)
        self._validate_pairs(proj, pairs, active)
        if pairs.numel() == 0 or not bool(active.any()):
            return super().forward(proj)

        projections = torch.randn(
            proj.size(-1), self.num_proj, device=proj.device
        )
        projections = projections.div_(projections.norm(p=2, dim=0))
        rows = []
        inverse_sqrt_two = 1.0 / math.sqrt(2.0)
        for time_index in range(proj.size(0)):
            selected = active[time_index]
            if not bool(selected.any()):
                population = proj[time_index]
            else:
                selected_pairs = pairs[selected]
                population = (
                    proj[time_index, selected_pairs[:, 0]]
                    - proj[time_index, selected_pairs[:, 1]]
                ) * inverse_sqrt_two
                if self.randomize_pair_orientation:
                    signs = torch.empty(
                        population.size(0),
                        1,
                        device=population.device,
                        dtype=population.dtype,
                    )
                    signs.bernoulli_(0.5).mul_(2.0).sub_(1.0)
                    population = population * signs
            rows.append(
                self._projected_statistic(
                    population.unsqueeze(0), projections
                ).mean()
            )
        return torch.stack(rows).mean()


class VISRegLoss(torch.nn.Module):
    """VISReg sliced Gaussianity regularizer.

    This implementation is vendored from ``stable-pretraining`` commit
    ``7830274c5b92637da7b1a433766494df0d5dbe85``. See
    ``THIRD_PARTY_NOTICES.md`` for source and license details.

    For each leading slice, the batch of embeddings is regularized toward an
    isotropic standard Gaussian through three terms:

    - center: penalize non-zero feature means;
    - scale: penalize per-feature standard deviations away from one;
    - shape: match sorted random one-dimensional projections to theoretical
      standard-normal quantiles.

    Args:
        num_projections: Number of random one-dimensional projections.
        lambda_scale: Weight on the unit-scale term.
        lambda_shape: Weight on the sliced-quantile shape term.
        lambda_center: Weight on the zero-mean term.
    """

    def __init__(
        self,
        num_projections: int = 256,
        lambda_scale: float = 1.0,
        lambda_shape: float = 1.0,
        lambda_center: float = 1.0,
    ):
        super().__init__()
        self.K = num_projections
        self.lambda_scale = lambda_scale
        self.lambda_shape = lambda_shape
        self.lambda_center = lambda_center
        self._cached_B = -1
        self._cached_target = None

    def _get_target(self, B: int, device, dtype) -> torch.Tensor:
        """Return theoretical standard-normal quantiles for ``B`` samples."""

        if self._cached_B != B:
            q = torch.linspace(
                1,
                B,
                B,
                device=device,
                dtype=torch.float32,
            ) / (B + 1)
            self._cached_target = torch.erfinv(2 * q - 1).mul_(math.sqrt(2))
            self._cached_B = B
        return self._cached_target.to(device=device, dtype=dtype)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Compute VISReg for embeddings shaped ``(V, B, D)``."""

        _, B, D = z.shape

        mu = z.mean(dim=1, keepdim=True)
        center_loss = mu.pow(2).mean()

        z_centered = z - mu
        std = z_centered.norm(dim=1).div(math.sqrt(B)) + 1e-6
        scale_loss = (std - 1.0).pow(2).mean()

        z_norm = z_centered / std.detach().unsqueeze(1)
        W = F.normalize(
            torch.randn(D, self.K, device=z.device, dtype=z.dtype),
            dim=0,
        )
        p_sorted = (z_norm @ W).sort(dim=1).values
        target = self._get_target(B, z.device, z.dtype).view(1, B, 1)
        shape_loss = (p_sorted - target).pow(2).mean()

        return (
            self.lambda_scale * scale_loss
            + self.lambda_shape * shape_loss
            + self.lambda_center * center_loss
        )


class VCReg(torch.nn.Module):
    """Variance-Covariance Regularizer"""

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def _std_loss(self, z):
        z = z.transpose(0, 1)  # (T, B, D)
        std = (z.var(dim=1) + self.eps).sqrt()  # (T, D)
        std_loss = torch.mean(F.relu(1 - std), dim=-1)  # (T,)
        return std_loss

    def _cov_loss(self, z):
        B, T, D = z.shape
        z = z.transpose(0, 1)  # (T, B, D)
        cov = einsum(z, z, 't b i, t b j -> t i j') / (B - 1)  # (T, D, D)
        diag = einsum(cov, 't i i -> t i').pow(2).sum(dim=-1)  # (T,)
        cov_loss = (cov.pow(2).sum(dim=[-1, -2]) - diag).div(D**2 - D)  # (T,)
        return cov_loss

    def forward(self, z):
        """
        z: (..., D)
        """

        if z.dim() == 2:
            D = z.size(-1)
            z = z.view(-1, D)

        z = z - z.mean(
            dim=0, keepdim=True
        )  # mean for each dim across batch samples

        return {
            'std_loss': self._std_loss(z).mean(),
            'std_t_loss': self._std_loss(z.transpose(0, 1)).mean(),
            'cov_loss': self._cov_loss(z).mean(),
            'cov_t_loss': self._cov_loss(z.transpose(0, 1)).mean(),
        }


class PLDMLoss(torch.nn.Module):
    """VCReg anti-collapse + Temporal Alignment + Inverse Dynamics Modeling losses
    reference: https://arxiv.org/abs/2502.14819
    """

    def __init__(self):
        super().__init__()
        self.vc_reg = VCReg()

    def forward(self, z, a_pred=None, a_target=None):
        """
        z: (B, T, D)
        a_pred: (B, T-1, A)
        a_target: (B, T-1, A)
        """

        output = {}
        if a_pred is not None and a_target is not None:
            output['idm_loss'] = F.mse_loss(a_pred, a_target)

        output['temp_align_loss'] = F.mse_loss(z[:, :-1], z[:, 1:])  # detach?
        output.update(self.vc_reg(z))

        return output


class TemporalStraighteningLoss(torch.nn.Module):
    """Temporal Straightening Loss Module (Mean Pairwise Negative Cosine Similarity)
    reference: https://arxiv.org/abs/2603.12231
    """

    def __init__(self):
        super().__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        v = x[:, 1:] - x[:, :-1]  # velocities
        sim = self.cos_sim(v[:, :-1], v[:, 1:])
        return -sim.mean()


__all__ = [
    'ConditionalSIGReg',
    'PLDMLoss',
    'SIGReg',
    'TemporalStraighteningLoss',
    'VCReg',
    'VISRegLoss',
]
