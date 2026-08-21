from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_pusht_motion_damping_anchored_context_transfer_v1.py"
)


def _load():
    name = "_test_pusht_motion_damping_anchored_context_transfer_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def test_transfer_preserves_source_transitions_and_destination_query():
    module = _load()
    history = torch.tensor(
        [
            [[0.0], [1.0], [3.0]],
            [[10.0], [12.0], [15.0]],
            [[100.0], [104.0], [109.0]],
            [[200.0], [206.0], [213.0]],
        ]
    )
    actions = torch.tensor(
        [
            [[0.0], [1.0], [2.0]],
            [[10.0], [11.0], [12.0]],
            [[20.0], [21.0], [22.0]],
            [[30.0], [31.0], [32.0]],
        ]
    )
    transferred, transferred_actions, source = (
        module.anchored_context_transfer(history, actions)
    )

    assert source.tolist() == [2, 3, 0, 1]
    torch.testing.assert_close(transferred[:, -1], history[:, -1])
    torch.testing.assert_close(
        transferred[:, 1:] - transferred[:, :-1],
        (history.index_select(0, source)[:, 1:]
         - history.index_select(0, source)[:, :-1]),
    )
    torch.testing.assert_close(
        transferred_actions[:, :-1],
        actions.index_select(0, source)[:, :-1],
    )
    torch.testing.assert_close(
        transferred_actions[:, -1], actions[:, -1]
    )


def test_source_mapping_is_involution_and_rejects_incomplete_twins():
    module = _load()
    source = module.same_condition_opposite_query_indices(12)
    torch.testing.assert_close(source.index_select(0, source), torch.arange(12))
    try:
        module.same_condition_opposite_query_indices(6)
    except ValueError as error:
        assert "four-row" in str(error)
    else:
        raise AssertionError("incomplete twin groups were accepted")


def test_prefix_basis_removes_absolute_translation_from_transfer():
    module = _load()
    from research.conditional_dynamics_representation.scripts import (
        run_pusht_motion_damping_terminal_aligned_transition_v1 as terminal,
    )

    history = torch.arange(4 * 3 * 2, dtype=torch.float32).reshape(4, 3, 2)
    actions = torch.zeros_like(history)
    transferred, _, source = module.anchored_context_transfer(history, actions)
    represented = terminal.terminal_aligned_transition_basis(transferred)
    source_history = history.index_select(0, source)
    expected = torch.cat(
        [source_history[:, 1:] - source_history[:, :-1], history[:, -1:]],
        dim=1,
    )
    torch.testing.assert_close(represented, expected)
