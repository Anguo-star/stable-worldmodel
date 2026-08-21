from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_action_delay_causal_transition_basis_v1.py"
)


def _load():
    name = "_test_action_delay_causal_transition_basis_v1"
    specification = importlib.util.spec_from_file_location(name, SOURCE)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


class _LegacyLeWM:
    def __init__(self, marker=None, **kwargs):
        self.marker = marker
        self.seen = None

    def predict(self, embedding, action_embedding):
        self.seen = embedding
        return embedding + action_embedding


def test_runtime_overlay_is_parameter_free_and_causal():
    module = _load()
    audit = module._install_runtime_basis(_LegacyLeWM)
    assert audit == {"native_support": False, "patched": True}

    model = _LegacyLeWM(temporal_input_basis="causal_transition")
    embedding = torch.randn(2, 4, 3)
    action = torch.randn_like(embedding)
    model.predict(embedding, action)

    expected = torch.cat(
        [embedding[:, :1], embedding[:, 1:] - embedding[:, :-1]], dim=1
    )
    torch.testing.assert_close(model.seen, expected)
    torch.testing.assert_close(expected.cumsum(dim=1), embedding)

    changed = embedding.clone()
    changed[:, 3:] += 10
    changed_basis = module._causal_transition_basis(changed)
    torch.testing.assert_close(changed_basis[:, :3], expected[:, :3])
