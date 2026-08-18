import math

import pytest
import torch

from stable_worldmodel.wm.loss import (
    ConditionalSIGReg,
    DynamicsResponseSIGReg,
    GroupBalancedSIGReg,
    JointTemporalCovarianceSIGReg,
    ScaleCalibratedConditionalSIGReg,
    SIGReg,
    TemporallyCenteredSIGReg,
)


def _adjacent_pairs(pair_count: int) -> torch.Tensor:
    return torch.arange(2 * pair_count).view(pair_count, 2)


def test_sigreg_explicit_full_tail_is_exactly_native():
    embeddings = torch.randn(3, 32, 24)

    torch.manual_seed(11)
    expected = SIGReg(knots=9, num_proj=64)(embeddings)
    torch.manual_seed(11)
    observed = SIGReg(
        knots=9,
        num_proj=64,
        tail_fraction=1.0,
    )(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_tail_sigreg_matches_manual_per_time_top_fraction():
    embeddings = torch.randn(3, 32, 24)
    tail_fraction = 0.1
    regularizer = SIGReg(
        knots=9,
        num_proj=64,
        tail_fraction=tail_fraction,
    )

    torch.manual_seed(13)
    projections = torch.randn(24, 64)
    projections = projections / projections.norm(p=2, dim=0)
    statistic = regularizer._projected_statistic(
        embeddings,
        projections,
    )
    expected = statistic.topk(
        math.ceil(tail_fraction * statistic.size(-1)),
        dim=-1,
        sorted=False,
    ).values.mean()

    torch.manual_seed(13)
    observed = regularizer(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_tail_sigreg_is_at_least_native_mean_on_same_projections():
    embeddings = torch.randn(3, 32, 24)

    torch.manual_seed(15)
    native = SIGReg(knots=9, num_proj=64)(embeddings)
    torch.manual_seed(15)
    tail = SIGReg(
        knots=9,
        num_proj=64,
        tail_fraction=0.1,
    )(embeddings)

    assert float(tail) >= float(native)


@pytest.mark.parametrize(
    "tail_fraction",
    [0.0, -0.1, 1.1, float("nan"), float("inf")],
)
def test_sigreg_rejects_invalid_tail_fraction(tail_fraction):
    with pytest.raises(ValueError, match="tail_fraction"):
        SIGReg(tail_fraction=tail_fraction)


def test_asymmetric_sigreg_matches_manual_characteristic_statistic():
    embeddings = torch.randn(3, 32, 24)
    overdispersion_weight = 0.25
    regularizer = SIGReg(
        knots=9,
        num_proj=64,
        overdispersion_weight=overdispersion_weight,
    )

    torch.manual_seed(16)
    projections = torch.randn(24, 64)
    projections = projections / projections.norm(p=2, dim=0)
    arguments = (embeddings @ projections).unsqueeze(-1) * regularizer.t
    cosine_residual = (
        arguments.cos().mean(dim=-3) - regularizer.phi
    )
    sine_mean = arguments.sin().mean(dim=-3)
    expected = (
        (
            torch.relu(cosine_residual).square()
            + overdispersion_weight
            * torch.relu(-cosine_residual).square()
            + sine_mean.square()
        )
        @ regularizer.weights
        * embeddings.size(-2)
    ).mean()

    torch.manual_seed(16)
    observed = regularizer(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_one_sided_sigreg_removes_only_negative_cosine_residuals():
    embeddings = 2.5 * torch.randn(3, 128, 24)

    torch.manual_seed(18)
    native = SIGReg(knots=9, num_proj=64)(embeddings)
    torch.manual_seed(18)
    one_sided = SIGReg(
        knots=9,
        num_proj=64,
        overdispersion_weight=0.0,
    )(embeddings)

    assert float(one_sided) <= float(native)


@pytest.mark.parametrize(
    "overdispersion_weight",
    [-0.1, 1.1, float("nan"), float("inf")],
)
def test_sigreg_rejects_invalid_overdispersion_weight(
    overdispersion_weight,
):
    with pytest.raises(ValueError, match="overdispersion_weight"):
        SIGReg(overdispersion_weight=overdispersion_weight)


def test_temporally_centered_sigreg_matches_manual_residuals():
    embeddings = torch.randn(4, 32, 24)
    residuals = embeddings - embeddings.mean(dim=0, keepdim=True)

    torch.manual_seed(19)
    expected = SIGReg(knots=9, num_proj=64)(residuals)
    torch.manual_seed(19)
    observed = TemporallyCenteredSIGReg(
        knots=9,
        num_proj=64,
    )(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_temporally_centered_sigreg_is_trajectory_translation_invariant():
    embeddings = torch.randn(4, 32, 24)
    trajectory_centers = torch.randn(1, 32, 24)
    regularizer = TemporallyCenteredSIGReg(knots=9, num_proj=64)

    torch.manual_seed(21)
    expected = regularizer(embeddings)
    torch.manual_seed(21)
    observed = regularizer(embeddings + trajectory_centers)

    torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-5)


def test_temporally_centered_sigreg_backpropagates_through_centering():
    embeddings = torch.randn(4, 32, 24, requires_grad=True)

    torch.manual_seed(25)
    loss = TemporallyCenteredSIGReg(
        knots=9,
        num_proj=64,
    )(embeddings)
    gradient = torch.autograd.grad(loss, embeddings)[0]

    assert float(gradient.norm()) > 0.0
    torch.testing.assert_close(
        gradient.sum(dim=0),
        torch.zeros_like(gradient[0]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.randn(32, 24),
        torch.randn(1, 32, 24),
    ],
)
def test_temporally_centered_sigreg_rejects_invalid_sequences(embeddings):
    with pytest.raises(ValueError, match="TemporallyCenteredSIGReg"):
        TemporallyCenteredSIGReg()(embeddings)


def test_joint_temporal_covariance_sigreg_matches_manual_joint_whitening():
    embeddings = torch.randn(4, 32, 24)
    rho = 0.125
    basis = torch.tensor(
        [
            [0.5, 0.5, 0.5, 0.5],
            [2**-0.5, -(2**-0.5), 0.0, 0.0],
            [6**-0.5, 6**-0.5, -2 * 6**-0.5, 0.0],
            [12**-0.5, 12**-0.5, 12**-0.5, -3 * 12**-0.5],
        ]
    )
    modes = torch.einsum("st,tbd->sbd", basis, embeddings)
    variances = torch.tensor([4 - 3 * rho, rho, rho, rho])
    modes = modes / variances.sqrt().view(4, 1, 1)
    joint = modes.permute(1, 0, 2).reshape(1, 32, 4 * 24)

    torch.manual_seed(31)
    expected = SIGReg(knots=9, num_proj=64)(joint)
    torch.manual_seed(31)
    observed = JointTemporalCovarianceSIGReg(
        knots=9,
        num_proj=64,
        rho=rho,
    )(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_joint_temporal_covariance_default_rho_and_basis_are_exact():
    regularizer = JointTemporalCovarianceSIGReg(knots=9, num_proj=64)
    basis = regularizer._helmert_basis(
        4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert regularizer.resolved_rho(4) == pytest.approx(1 / 16)
    torch.testing.assert_close(
        basis @ basis.T,
        torch.eye(4, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        basis[0],
        torch.full((4,), 0.5, dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )


def test_joint_temporal_covariance_adds_no_learned_parameters():
    regularizer = JointTemporalCovarianceSIGReg(knots=9, num_proj=64)

    assert list(regularizer.named_parameters()) == []
    assert set(regularizer.state_dict()) == {"t", "phi", "weights"}


def test_joint_temporal_covariance_accepts_native_sigreg_defaults():
    embeddings = torch.randn(4, 32, 24)
    regularizer = JointTemporalCovarianceSIGReg(
        knots=9,
        num_proj=64,
        tail_fraction=1.0,
        overdispersion_weight=1.0,
    )

    torch.manual_seed(41)
    expected = JointTemporalCovarianceSIGReg(
        knots=9,
        num_proj=64,
    )(embeddings)
    torch.manual_seed(41)
    observed = regularizer(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_joint_temporal_covariance_sigreg_backpropagates_all_time_modes():
    embeddings = torch.randn(4, 32, 24, requires_grad=True)

    torch.manual_seed(37)
    loss = JointTemporalCovarianceSIGReg(
        knots=9,
        num_proj=64,
    )(embeddings)
    gradient = torch.autograd.grad(loss, embeddings)[0]

    assert torch.all(gradient.flatten(1).norm(dim=1) > 0)
    assert float(gradient.mean(dim=0).norm()) > 0.0


@pytest.mark.parametrize("rho", [0.0, -0.1, 1.1, float("nan")])
def test_joint_temporal_covariance_sigreg_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match="rho"):
        JointTemporalCovarianceSIGReg(rho=rho)


@pytest.mark.parametrize(
    "embeddings",
    [
        torch.randn(32, 24),
        torch.randn(1, 32, 24),
    ],
)
def test_joint_temporal_covariance_sigreg_rejects_invalid_sequences(
    embeddings,
):
    with pytest.raises(ValueError, match="JointTemporalCovarianceSIGReg"):
        JointTemporalCovarianceSIGReg()(embeddings)


def test_without_pairs_is_exactly_native_sigreg():
    embeddings = torch.randn(3, 32, 24)
    native = SIGReg(knots=9, num_proj=64)
    conditional = ConditionalSIGReg(knots=9, num_proj=64)

    torch.manual_seed(17)
    expected = native(embeddings)
    torch.manual_seed(17)
    observed = conditional(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_independent_gaussian_pair_contrasts_match_null_scale():
    pair_count = 512
    embeddings = torch.randn(1, 2 * pair_count, 64)
    loss = ConditionalSIGReg(knots=17, num_proj=256)(
        embeddings,
        pairs=_adjacent_pairs(pair_count),
        active=torch.ones(1, pair_count, dtype=torch.bool),
    )

    assert 0.75 < float(loss) < 1.35


def test_conditional_collapse_is_visible_when_marginals_still_vary():
    pair_count = 256
    values = torch.randn(pair_count, 48)
    collapsed = torch.stack([values, values], dim=1).reshape(
        1, 2 * pair_count, 48
    )
    pairs = _adjacent_pairs(pair_count)
    active = torch.ones(1, pair_count, dtype=torch.bool)

    torch.manual_seed(23)
    marginal_loss = SIGReg(knots=17, num_proj=256)(collapsed)
    torch.manual_seed(23)
    conditional_loss = ConditionalSIGReg(
        knots=17,
        num_proj=256,
    )(collapsed, pairs=pairs, active=active)

    assert float(conditional_loss) > 2.0 * float(marginal_loss)


def test_small_pair_contrasts_receive_an_expanding_descent_direction():
    pair_count = 128
    base = torch.randn(pair_count, 32)
    direction = torch.randn(pair_count, 32)
    direction = torch.nn.functional.normalize(direction, dim=-1)
    epsilon = 0.05
    values = torch.stack(
        [base + epsilon * direction, base - epsilon * direction],
        dim=1,
    )
    embeddings = values.reshape(1, 2 * pair_count, 32).requires_grad_()
    pairs = _adjacent_pairs(pair_count)
    active = torch.ones(1, pair_count, dtype=torch.bool)

    torch.manual_seed(29)
    regularizer = ConditionalSIGReg(knots=17, num_proj=512)(
        embeddings,
        pairs=pairs,
        active=active,
    )
    pair_distance = (
        embeddings[:, pairs[:, 0]] - embeddings[:, pairs[:, 1]]
    ).square().mean()
    distance_gradient = torch.autograd.grad(
        pair_distance, embeddings, retain_graph=True
    )[0]
    loss_gradient = torch.autograd.grad(regularizer, embeddings)[0]
    predicted_distance_change = -torch.sum(
        distance_gradient * loss_gradient
    )

    assert float(predicted_distance_change) > 0.0


def test_include_unpaired_restores_gradient_coverage():
    embeddings = torch.randn(1, 12, 24, requires_grad=True)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.ones(1, 2, dtype=torch.bool)

    torch.manual_seed(31)
    high_pass_only = ConditionalSIGReg(
        knots=9,
        num_proj=64,
        include_unpaired=False,
    )(embeddings, pairs=pairs, active=active)
    high_pass_gradient = torch.autograd.grad(
        high_pass_only,
        embeddings,
        retain_graph=True,
    )[0]
    torch.manual_seed(31)
    mixed_population = ConditionalSIGReg(
        knots=9,
        num_proj=64,
        include_unpaired=True,
    )(embeddings, pairs=pairs, active=active)
    mixed_gradient = torch.autograd.grad(
        mixed_population,
        embeddings,
    )[0]

    assert torch.count_nonzero(high_pass_gradient[:, :8]) == 0
    assert float(mixed_gradient[:, :8].norm()) > 0.0


def test_include_unpaired_is_exact_when_every_row_is_paired():
    embeddings = torch.randn(2, 16, 20)
    pairs = _adjacent_pairs(8)
    active = torch.ones(2, 8, dtype=torch.bool)

    torch.manual_seed(37)
    expected = ConditionalSIGReg(
        knots=9,
        num_proj=64,
        include_unpaired=False,
    )(embeddings, pairs=pairs, active=active)
    torch.manual_seed(37)
    observed = ConditionalSIGReg(
        knots=9,
        num_proj=64,
        include_unpaired=True,
    )(embeddings, pairs=pairs, active=active)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_complete_haar_matches_sigreg_on_manual_transform():
    embeddings = torch.randn(2, 12, 20)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.ones(2, 2, dtype=torch.bool)
    inverse_sqrt_two = 2.0**-0.5
    manually_transformed = torch.cat(
        [
            embeddings[:, :8],
            (
                embeddings[:, pairs[:, 0]]
                + embeddings[:, pairs[:, 1]]
            )
            * inverse_sqrt_two,
            (
                embeddings[:, pairs[:, 0]]
                - embeddings[:, pairs[:, 1]]
            )
            * inverse_sqrt_two,
        ],
        dim=1,
    )

    torch.manual_seed(41)
    expected = SIGReg(knots=9, num_proj=64)(manually_transformed)
    torch.manual_seed(41)
    observed = ConditionalSIGReg(
        knots=9,
        num_proj=64,
        randomize_pair_orientation=False,
        complete_haar_population=True,
    )(embeddings, pairs=pairs, active=active)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_complete_haar_and_include_unpaired_are_mutually_exclusive():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ConditionalSIGReg(
            include_unpaired=True,
            complete_haar_population=True,
        )


def test_group_balanced_without_pairs_is_exactly_native_sigreg():
    embeddings = torch.randn(3, 32, 24)
    native = SIGReg(knots=9, num_proj=64)
    balanced = GroupBalancedSIGReg(knots=9, num_proj=64)

    torch.manual_seed(43)
    expected = native(embeddings)
    torch.manual_seed(43)
    observed = balanced(embeddings)

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_group_balanced_matches_separately_scored_manual_populations():
    embeddings = torch.randn(2, 12, 20)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.tensor([[False, False], [True, True]])
    regularizer = GroupBalancedSIGReg(
        knots=9,
        num_proj=64,
        randomize_pair_orientation=False,
    )

    torch.manual_seed(47)
    projections = torch.randn(20, 64)
    projections = projections / projections.norm(p=2, dim=0)
    marginal_t0 = regularizer._projected_statistic(
        embeddings[0].unsqueeze(0),
        projections,
    ).mean()
    marginal_t1 = regularizer._projected_statistic(
        embeddings[1].unsqueeze(0),
        projections,
    ).mean()
    contrasts = (
        embeddings[1, pairs[:, 0]] - embeddings[1, pairs[:, 1]]
    ) / (2.0**0.5)
    contrast_t1 = regularizer._projected_statistic(
        contrasts.unsqueeze(0),
        projections,
    ).mean()
    expected = torch.stack(
        [marginal_t0, 0.5 * (marginal_t1 + contrast_t1)]
    ).mean()

    torch.manual_seed(47)
    observed = regularizer(
        embeddings,
        pairs=pairs,
        active=active,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_group_balanced_keeps_unpaired_gradient_coverage():
    embeddings = torch.randn(1, 12, 24, requires_grad=True)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.ones(1, 2, dtype=torch.bool)

    torch.manual_seed(53)
    loss = GroupBalancedSIGReg(knots=9, num_proj=64)(
        embeddings,
        pairs=pairs,
        active=active,
    )
    gradient = torch.autograd.grad(loss, embeddings)[0]

    assert float(gradient[:, :8].norm()) > 0.0
    assert float(gradient[:, 8:].norm()) > 0.0


def test_scale_calibrated_population_matches_manual_sigreg():
    embeddings = torch.randn(2, 12, 20)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.tensor([[False, False], [True, True]])
    scales = torch.tensor([1.0, 0.25])
    contrasts = (
        embeddings[1, pairs[:, 0]] - embeddings[1, pairs[:, 1]]
    ) / (2.0**0.5 * scales[1])
    regularizer = ScaleCalibratedConditionalSIGReg(
        knots=9,
        num_proj=64,
        randomize_pair_orientation=False,
    )
    torch.manual_seed(59)
    projections = torch.randn(20, 64)
    projections = projections / projections.norm(p=2, dim=0)
    expected = torch.stack(
        [
            regularizer._projected_statistic(
                embeddings[0].unsqueeze(0),
                projections,
            ).mean(),
            regularizer._projected_statistic(
                torch.cat([embeddings[1, :8], contrasts], dim=0).unsqueeze(0),
                projections,
            ).mean(),
        ]
    ).mean()
    torch.manual_seed(59)
    observed = regularizer(
        embeddings,
        pairs=pairs,
        active=active,
        contrast_scales=scales,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_scale_calibrated_population_routes_both_gradient_groups():
    embeddings = torch.randn(1, 12, 24, requires_grad=True)
    pairs = torch.tensor([[8, 9], [10, 11]])
    active = torch.ones(1, 2, dtype=torch.bool)

    torch.manual_seed(61)
    loss = ScaleCalibratedConditionalSIGReg(
        knots=9,
        num_proj=64,
    )(
        embeddings,
        pairs=pairs,
        active=active,
        contrast_scales=torch.tensor([0.2]),
    )
    gradient = torch.autograd.grad(loss, embeddings)[0]

    assert float(gradient[:, :8].norm()) > 0.0
    assert float(gradient[:, 8:].norm()) > 0.0


@pytest.mark.parametrize(
    ("scales", "error"),
    [
        (None, "required"),
        (torch.tensor([1.0, 1.0]), "shape"),
        (torch.tensor([0.0]), "positive"),
        (torch.tensor([float("nan")]), "finite"),
    ],
)
def test_scale_calibrated_invalid_scale_fails_closed(scales, error):
    embeddings = torch.randn(1, 4, 8)
    pairs = torch.tensor([[2, 3]])
    active = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match=error):
        ScaleCalibratedConditionalSIGReg(knots=5, num_proj=8)(
            embeddings,
            pairs=pairs,
            active=active,
            contrast_scales=scales,
        )


def test_dynamics_response_excludes_irreducible_prediction_pairs():
    targets = torch.randn(2, 12, 20, requires_grad=True)
    predictions = torch.randn(2, 12, 20, requires_grad=True)
    pairs = torch.tensor([[8, 9], [10, 11]])
    target_active = torch.tensor([[True, True], [True, True]])
    prediction_active = torch.tensor(
        [[False, False], [True, True]]
    )

    torch.manual_seed(67)
    loss = DynamicsResponseSIGReg(
        knots=9,
        num_proj=64,
        randomize_pair_orientation=False,
    )(
        targets,
        predictions,
        pairs=pairs,
        target_active=target_active,
        prediction_active=prediction_active,
        contrast_scales=torch.tensor([0.2, 0.4]),
    )
    target_gradient, prediction_gradient = torch.autograd.grad(
        loss,
        (targets, predictions),
    )

    assert float(target_gradient[:, :8].norm()) > 0.0
    assert float(prediction_gradient[:, :8].norm()) > 0.0
    assert float(target_gradient[:, 8:].norm()) > 0.0
    assert torch.count_nonzero(prediction_gradient[0, 8:]) == 0
    assert float(prediction_gradient[1, 8:].norm()) > 0.0


def test_dynamics_response_without_pairs_is_native_joint_sigreg():
    targets = torch.randn(3, 16, 20)
    predictions = torch.randn(3, 16, 20)

    torch.manual_seed(71)
    expected = SIGReg(knots=9, num_proj=64)(
        torch.cat([targets, predictions], dim=1)
    )
    torch.manual_seed(71)
    observed = DynamicsResponseSIGReg(knots=9, num_proj=64)(
        targets,
        predictions,
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_dynamics_response_rejects_prediction_only_contrast():
    targets = torch.randn(1, 4, 8)
    predictions = torch.randn(1, 4, 8)
    pairs = torch.tensor([[2, 3]])
    with pytest.raises(ValueError, match="also be target-active"):
        DynamicsResponseSIGReg(knots=5, num_proj=8)(
            targets,
            predictions,
            pairs=pairs,
            target_active=torch.zeros(1, 1, dtype=torch.bool),
            prediction_active=torch.ones(1, 1, dtype=torch.bool),
            contrast_scales=torch.ones(1),
        )


@pytest.mark.parametrize(
    ("pairs", "active", "error"),
    [
        (
            torch.tensor([[0, 1], [1, 2]]),
            torch.ones(2, 2, dtype=torch.bool),
            "disjoint",
        ),
        (
            torch.tensor([[0, 5]]),
            torch.ones(2, 1, dtype=torch.bool),
            "outside",
        ),
        (
            torch.tensor([[0, 1]], dtype=torch.int32),
            torch.ones(2, 1, dtype=torch.bool),
            "torch.long",
        ),
        (
            torch.empty(0, 2, dtype=torch.long),
            torch.ones(2, 1, dtype=torch.bool),
            "active must have shape",
        ),
    ],
)
def test_invalid_pair_contract_fails_closed(pairs, active, error):
    embeddings = torch.randn(2, 4, 8)
    with pytest.raises((TypeError, ValueError), match=error):
        ConditionalSIGReg(knots=5, num_proj=8)(
            embeddings,
            pairs=pairs,
            active=active,
        )
