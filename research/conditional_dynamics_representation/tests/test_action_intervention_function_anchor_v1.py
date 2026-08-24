import pytest
import torch

from research.conditional_dynamics_representation.scripts import (
    action_intervention_function_anchor_v1 as anchor,
)
from research.conditional_dynamics_representation.scripts import (
    run_pusht_motion_damping_action_intervention_anchor_v1 as runner,
)


def test_exact_source_response_is_zero_and_stationary() -> None:
    teacher_base = torch.randn(5, 7, requires_grad=True)
    teacher_counterfactual = torch.randn(5, 7, requires_grad=True)
    student_base = teacher_base.detach().clone().requires_grad_(True)
    student_counterfactual = (
        teacher_counterfactual.detach().clone().requires_grad_(True)
    )
    result = anchor.normalized_action_response_anchor(
        student_base=student_base,
        student_counterfactual=student_counterfactual,
        teacher_base=teacher_base,
        teacher_counterfactual=teacher_counterfactual,
    )
    assert float(result["loss"]) == 0.0
    result["loss"].backward()
    assert torch.count_nonzero(student_base.grad).item() == 0
    assert torch.count_nonzero(student_counterfactual.grad).item() == 0
    assert teacher_base.grad is None
    assert teacher_counterfactual.grad is None


def test_anchor_is_translation_invariant_and_corrective() -> None:
    teacher_base = torch.zeros(4, 3)
    teacher_counterfactual = torch.ones(4, 3)
    student_base = torch.full((4, 3), 5.0, requires_grad=True)
    student_counterfactual = torch.full(
        (4, 3), 5.25, requires_grad=True
    )
    result = anchor.normalized_action_response_anchor(
        student_base=student_base,
        student_counterfactual=student_counterfactual,
        teacher_base=teacher_base,
        teacher_counterfactual=teacher_counterfactual,
    )
    assert float(result["loss"]) == pytest.approx(0.75**2)
    result["loss"].backward()
    assert bool((student_base.grad > 0).all())
    assert bool((student_counterfactual.grad < 0).all())


def test_cycle_changes_only_final_query_action() -> None:
    actions = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    counterfactual, source_indices = runner.cyclic_final_query_action(actions)
    assert torch.equal(counterfactual[:, :-1], actions[:, :-1])
    assert torch.equal(counterfactual[:, -1], actions[source_indices, -1])
    assert bool((source_indices != torch.arange(4)).all())


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        runner.cyclic_final_query_action(torch.zeros(1, 3, 2))
    with pytest.raises(ValueError):
        anchor.normalized_action_response_anchor(
            student_base=torch.zeros(2, 3),
            student_counterfactual=torch.zeros(2, 4),
            teacher_base=torch.zeros(2, 3),
            teacher_counterfactual=torch.zeros(2, 3),
        )
