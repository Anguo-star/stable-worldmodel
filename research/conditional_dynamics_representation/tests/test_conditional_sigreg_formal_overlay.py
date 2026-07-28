from __future__ import annotations

import ast
import importlib
import importlib.util
import pickle
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/run_conditional_sigreg_formal.py"
)
SPEC = importlib.util.spec_from_file_location(
    "conditional_sigreg_formal_overlay",
    SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
OVERLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OVERLAY)


def test_overlay_does_not_import_torch_at_module_scope():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module)

    assert "torch" not in top_level_imports


def test_contextworld_trainer_module_is_spawn_importable(
    tmp_path,
    monkeypatch,
):
    contextworld_repo = tmp_path / "ContextWorld"
    scripts = contextworld_repo / "scripts"
    scripts.mkdir(parents=True)
    trainer = scripts / "train_tworoom_step1.py"
    trainer.write_text(
        "class PassageReleaseGatedDataset:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n",
        encoding="utf-8",
    )
    monkeypatch.delitem(
        sys.modules,
        "scripts.train_tworoom_step1",
        raising=False,
    )
    monkeypatch.delitem(sys.modules, "scripts", raising=False)
    importlib.invalidate_caches()

    module = OVERLAY._load_contextworld_train(contextworld_repo)
    payload = pickle.dumps(module.PassageReleaseGatedDataset("visible"))
    del sys.modules["scripts.train_tworoom_step1"]
    importlib.invalidate_caches()
    restored = pickle.loads(payload)

    assert module.__name__ == "scripts.train_tworoom_step1"
    assert restored.value == "visible"
    assert type(restored).__module__ == "scripts.train_tworoom_step1"


def test_swanlab_id_validation_runs_before_training():
    valid = Namespace(
        logger_backend="swanlab",
        swanlab_id="paired_native_s3072",
        run_name="unused",
    )
    OVERLAY._validate_external_logger_identity(valid)

    too_long = Namespace(
        logger_backend="swanlab",
        swanlab_id="x" * 65,
        run_name="unused",
    )
    with pytest.raises(ValueError, match="observed_length=65"):
        OVERLAY._validate_external_logger_identity(too_long)

    forbidden = Namespace(
        logger_backend="swanlab",
        swanlab_id="paired/native",
        run_name="unused",
    )
    with pytest.raises(ValueError, match="forbidden"):
        OVERLAY._validate_external_logger_identity(forbidden)


class _Samples:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def __getitems__(self, indices):
        return [self.values[int(index)] for index in indices]


class _Logical:
    def __init__(self, original, passage):
        self.names = ["original", "passage_mixed"]
        self.groups = [original, passage]
        self.normalized_weights = {
            "original": 0.5,
            "passage_mixed": 0.5,
        }
        self.column_names = ["pixels", "action"]

    def __len__(self):
        return 32

    def __getitem__(self, index):
        group = self.groups[index % 2]
        return group[(index // 2) % len(group)]

    def __getitems__(self, indices):
        return [self[int(index)] for index in indices]


def _passage_sample(pair: int, rule: int):
    pixels = torch.zeros(4, 3, 3, 3)
    pixels[1] = float(rule + 1)
    pixels[3] = float((pair + 1) * (rule + 1))
    return {
        "source": "passage",
        "pair": pair,
        "rule": rule,
        "pixels": pixels,
        "action": torch.full((4, 2), float(pair)),
    }


def test_full_batch_is_exact_replay50_with_adjacent_passage_pairs():
    original = _Samples(
        [{"source": "original", "index": index} for index in range(20)]
    )
    passage = _Samples(
        [
            _passage_sample(pair, rule)
            for pair in range(10)
            for rule in (0, 1)
        ]
    )
    dataset = OVERLAY.VisibleConditionReplay50Dataset(
        _Logical(original, passage),
        batch_size=8,
    )

    batch = dataset.__getitems__([11, 2, 7, 4, 1, 6, 9, 14])

    assert [sample["source"] for sample in batch] == [
        "original",
        "original",
        "original",
        "original",
        "passage",
        "passage",
        "passage",
        "passage",
    ]
    assert batch[4]["pair"] == batch[5]["pair"]
    assert batch[6]["pair"] == batch[7]["pair"]
    assert batch[4]["rule"] != batch[5]["rule"]
    assert batch[6]["rule"] != batch[7]["rule"]
    assert batch[4]["pair"] != batch[6]["pair"]
    assert all(sample[OVERLAY.PAIR_BATCH_MARKER] for sample in batch)


def test_non_full_read_preserves_native_logical_mapping():
    original = _Samples(
        [{"source": "original", "index": index} for index in range(20)]
    )
    passage = _Samples(
        [
            _passage_sample(pair, rule)
            for pair in range(10)
            for rule in (0, 1)
        ]
    )
    logical = _Logical(original, passage)
    dataset = OVERLAY.VisibleConditionReplay50Dataset(
        logical,
        batch_size=8,
    )

    expected = logical.__getitems__([0, 1, 2])
    observed = dataset.__getitems__([0, 1, 2])

    assert all(left is right for left, right in zip(observed, expected))


def test_pair_metadata_uses_only_visible_equality_and_difference():
    original_pixels = torch.randn(4, 4, 3, 3, 3)
    original_actions = torch.randn(4, 4, 2)
    pair_pixels = []
    pair_actions = []
    for pair in range(2):
        left = _passage_sample(pair, 0)
        right = _passage_sample(pair, 1)
        pair_pixels.extend([left["pixels"], right["pixels"]])
        pair_actions.extend([left["action"], right["action"]])
    pixels = torch.cat(
        [original_pixels, torch.stack(pair_pixels)],
        dim=0,
    )
    actions = torch.cat(
        [original_actions, torch.stack(pair_actions)],
        dim=0,
    )
    actions[4:, 0, 0] = float("nan")

    metadata = OVERLAY.conditional_pair_metadata(pixels, actions)

    assert metadata is not None
    pairs, active = metadata
    torch.testing.assert_close(
        pairs,
        torch.tensor([[4, 5], [6, 7]]),
    )
    torch.testing.assert_close(
        active,
        torch.tensor(
            [
                [False, False],
                [True, True],
                [False, False],
                [True, True],
            ]
        ),
    )


def test_unpaired_validation_layout_falls_back_to_native_sigreg():
    pixels = torch.randn(8, 4, 3, 3, 3)
    actions = torch.randn(8, 4, 2)

    assert OVERLAY.conditional_pair_metadata(pixels, actions) is None
