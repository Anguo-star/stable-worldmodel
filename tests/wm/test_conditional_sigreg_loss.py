import pytest
import torch

from stable_worldmodel.wm.loss import ConditionalSIGReg, SIGReg


def _adjacent_pairs(pair_count: int) -> torch.Tensor:
    return torch.arange(2 * pair_count).view(pair_count, 2)


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
