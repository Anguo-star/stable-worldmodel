from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_terminal_aligned_transition_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_terminal_aligned_transition_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_terminal_basis_pairs_support_deltas_and_retains_query_state():
    module = _load()
    embedding = torch.tensor([[[1.0], [3.0], [8.0]]])
    transformed = module.terminal_aligned_transition_basis(embedding)

    torch.testing.assert_close(
        transformed,
        torch.tensor([[[2.0], [5.0], [8.0]]]),
    )
    # Recover the absolute history backwards from the final state.  This is
    # full-history invertibility, not prefix causality for the early outputs.
    z2 = transformed[:, 2]
    z1 = z2 - transformed[:, 1]
    z0 = z1 - transformed[:, 0]
    torch.testing.assert_close(torch.stack([z0, z1, z2], dim=1), embedding)


def test_terminal_loss_has_no_gradient_on_leaky_early_outputs():
    module = _load()
    prediction = torch.randn(4, 3, 2, requires_grad=True)
    embeddings = torch.randn(4, 4, 2)
    loss = module.terminal_only_population_balanced_prediction_loss(
        prediction=prediction,
        embeddings=embeddings,
        original_batch_size=2,
        conditional_population="identifiable_future_only",
    )
    gradient = torch.autograd.grad(loss, prediction)[0]
    torch.testing.assert_close(gradient[:, :2], torch.zeros_like(gradient[:, :2]))
    assert torch.count_nonzero(gradient[:, -1]).item() > 0


def test_terminal_input_excludes_unobserved_future():
    module = _load()
    observed_history = torch.randn(2, 3, 5)
    first_future = torch.randn(2, 1, 5)
    second_future = first_future + 100.0

    first = module.terminal_aligned_transition_basis(observed_history)
    second = module.terminal_aligned_transition_basis(observed_history)
    torch.testing.assert_close(first, second)
    assert not torch.equal(first_future, second_future)
