#!/usr/bin/env python3
"""Train canonical history response on a real history x action grid.

This is the no-teacher counterpart of action-function anchoring.  Hidden
training rows form real simulator four-tuples: two damping histories crossed
with the observed zero query action and one nonzero query action.  Native MSE
sees every real future, while the existing canonical response auxiliary is
applied to each adjacent history pair.  The deployed LeWM is unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

import torch


sys.modules.setdefault("flash_attn", None)

THIS_SOURCE = Path(__file__).resolve()
REPO_ROOT = THIS_SOURCE.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.conditional_dynamics_representation.scripts import (  # noqa: E402
    run_pusht_motion_damping_canonical_response_only_freeze_v1 as base,
)


CANDIDATE = "pusht_motion_damping_cartesian_action_pair_v1"
OVERLAY_SHA256 = "d613a955ce9d50d7fcd32147928dae2b3c925ed2f6f96efd9dd5cea97c7c635e"
OVERLAY_TEMPLATE_COUNT = 256
OVERLAY_CONDITION_PAIR_COUNT = 512
ROWS_PER_TEMPLATE = 4
ALLOW_ACTION_FUTURE_DEGENERACY = False
EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT = 0
ROWS_PER_TWIN = 8
OPTIMIZER_STEPS = 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CartesianTwinBatchStream:
    """Yield forward/reverse twins with both real action branches."""

    def __init__(
        self,
        pair_count: int,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if pair_count <= 0 or pair_count % 4:
            raise ValueError("condition pair_count must be positive / 4")
        template_count = pair_count // 2
        if template_count % 2:
            raise ValueError("templates must contain forward/reverse twins")
        if batch_size <= 0 or batch_size % ROWS_PER_TWIN:
            raise ValueError("hidden batch must contain complete 8-row twins")
        self.template_count = template_count
        self.twin_count = template_count // 2
        self.twins_per_batch = batch_size // ROWS_PER_TWIN
        if self.twin_count % self.twins_per_batch:
            raise ValueError("twin_count must divide evenly across batches")
        self.generator = torch.Generator().manual_seed(int(seed))

    def __iter__(self) -> Iterator[torch.Tensor]:
        while True:
            order = torch.randperm(self.twin_count, generator=self.generator)
            for start in range(0, self.twin_count, self.twins_per_batch):
                twins = order[start : start + self.twins_per_batch]
                templates = torch.stack(
                    [2 * twins, 2 * twins + 1], dim=1
                ).reshape(-1)
                offsets = torch.arange(ROWS_PER_TEMPLATE)
                yield (
                    ROWS_PER_TEMPLATE * templates[:, None] + offsets[None, :]
                ).reshape(-1)


def validate_cartesian_grid(
    pixels: torch.Tensor,
    raw_action_blocks: torch.Tensor,
    *,
    template_count: int,
) -> dict[str, Any]:
    expected_rows = ROWS_PER_TEMPLATE * int(template_count)
    if pixels.dtype != torch.uint8 or pixels.shape != (
        expected_rows,
        4,
        3,
        224,
        224,
    ):
        raise ValueError(f"unexpected cartesian pixels {tuple(pixels.shape)}")
    if not raw_action_blocks.is_floating_point() or raw_action_blocks.shape != (
        expected_rows,
        4,
        5,
        2,
    ):
        raise ValueError(
            f"unexpected cartesian actions {tuple(raw_action_blocks.shape)}"
        )
    if not bool(torch.isfinite(raw_action_blocks).all()):
        raise FloatingPointError("cartesian actions contain nonfinite values")
    checks = {
        "history_query_exact_across_action_branches": True,
        "query_exact_across_hidden_modes": True,
        "support_actions_exact_across_action_branches": True,
        "query_actions_differ_across_action_branches": True,
        "actions_exact_across_hidden_modes": True,
        "history_differs_across_hidden_modes": True,
        "future_differs_across_hidden_modes": True,
        "future_differs_across_action_branches": True,
    }
    degenerate_action_future_count = 0
    for template_index in range(template_count):
        start = ROWS_PER_TEMPLATE * template_index
        a0_low, a0_high, a1_low, a1_high = range(start, start + 4)
        for left, right in ((a0_low, a1_low), (a0_high, a1_high)):
            checks["history_query_exact_across_action_branches"] &= bool(
                torch.equal(pixels[left, :3], pixels[right, :3])
            )
            checks["support_actions_exact_across_action_branches"] &= bool(
                torch.equal(
                    raw_action_blocks[left, :2], raw_action_blocks[right, :2]
                )
                and torch.equal(
                    raw_action_blocks[left, 3:], raw_action_blocks[right, 3:]
                )
            )
            checks["query_actions_differ_across_action_branches"] &= bool(
                not torch.equal(
                    raw_action_blocks[left, 2], raw_action_blocks[right, 2]
                )
            )
            checks["future_differs_across_action_branches"] &= bool(
                not torch.equal(pixels[left, 3], pixels[right, 3])
            )
            degenerate_action_future_count += int(
                torch.equal(pixels[left, 3], pixels[right, 3])
            )
        for left, right in ((a0_low, a0_high), (a1_low, a1_high)):
            checks["query_exact_across_hidden_modes"] &= bool(
                torch.equal(pixels[left, 2], pixels[right, 2])
            )
            checks["actions_exact_across_hidden_modes"] &= bool(
                torch.equal(raw_action_blocks[left], raw_action_blocks[right])
            )
            checks["history_differs_across_hidden_modes"] &= bool(
                not torch.equal(pixels[left, 1], pixels[right, 1])
            )
            checks["future_differs_across_hidden_modes"] &= bool(
                not torch.equal(pixels[left, 3], pixels[right, 3])
            )
    if ALLOW_ACTION_FUTURE_DEGENERACY:
        checks.pop("future_differs_across_action_branches")
        checks[
            "action_future_separation_or_registered_native_only_rows"
        ] = (
            degenerate_action_future_count
            == EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT
        )
        checks["degenerate_action_future_count_exact"] = (
            degenerate_action_future_count
            == EXPECTED_DEGENERATE_ACTION_FUTURE_COUNT
        )
    if not all(checks.values()):
        raise RuntimeError(f"cartesian grid contract failed: {checks}")
    return checks


def _parse_overlay(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cartesian-overlay", type=Path, required=True)
    args, _ = parser.parse_known_args(list(argv))
    overlay = args.cartesian_overlay.expanduser().resolve()
    if not overlay.is_file() or _sha256(overlay) != OVERLAY_SHA256:
        raise RuntimeError("cartesian overlay identity changed")
    return overlay


def _without_overlay_argument(argv: Sequence[str]) -> list[str]:
    result: list[str] = []
    index = 0
    values = list(argv)
    while index < len(values):
        if values[index] == "--cartesian-overlay":
            if index + 1 >= len(values):
                raise ValueError("--cartesian-overlay is missing its value")
            index += 2
            continue
        result.append(values[index])
        index += 1
    return result


def _install_cartesian_data(
    trainer: Any,
    *,
    overlay: Path,
) -> dict[str, Any]:
    native_release_loader = trainer.load_motion_damping_icl_release
    native_training_split = trainer.trainer._training_split
    state: dict[str, Any] = {
        "release_calls": 0,
        "materialization_calls": 0,
        "grid_checks": None,
    }

    def load_release(path: Path) -> dict[str, Any]:
        release = copy.deepcopy(native_release_loader(path))
        release["data"]["pair_counts"]["train"] = (
            OVERLAY_CONDITION_PAIR_COUNT
        )
        followup = release["training"]["learnability_followup"]
        followup.update(
            {
                "candidate": CANDIDATE,
                "fixed_checkpoint_step": OPTIMIZER_STEPS,
                "training_data_overlay": "real_history_x_action_four_tuples",
            }
        )
        state["release_calls"] += 1
        return release

    def training_split(
        _path: Path,
        *,
        expected_pairs: int,
        action_stats: dict[str, Any],
    ) -> Any:
        if expected_pairs != OVERLAY_CONDITION_PAIR_COUNT:
            raise RuntimeError("runtime train pair count did not bind overlay")
        payload = torch.load(overlay, map_location="cpu", weights_only=True)
        if (
            int(payload["template_count"]) != OVERLAY_TEMPLATE_COUNT
            or int(payload["condition_pair_count"])
            != OVERLAY_CONDITION_PAIR_COUNT
            or payload["row_order"]
            != "template_then_action_branch_then_damping_mode"
        ):
            raise RuntimeError("cartesian overlay metadata changed")
        pixels = payload["pixels"]
        raw_actions = payload["raw_action_blocks"]
        state["grid_checks"] = validate_cartesian_grid(
            pixels,
            raw_actions,
            template_count=OVERLAY_TEMPLATE_COUNT,
        )
        actions = trainer.trainer.pilot.normalize_action_blocks(
            raw_actions.reshape(
                raw_actions.size(0), raw_actions.size(1), -1
            ).float(),
            action_stats,
        )
        split = trainer.trainer.pilot.MaterializedSplit(
            pixels=pixels,
            action=actions,
            pair_count=OVERLAY_CONDITION_PAIR_COUNT,
        )
        state["materialization_calls"] += 1
        return split

    trainer.load_motion_damping_icl_release = load_release
    trainer.trainer._training_split = training_split
    trainer.CompleteTwinPairedBatchStream = CartesianTwinBatchStream
    state["native_training_split_replaced"] = (
        native_training_split is not training_split
    )
    return state


def _rewrite_report(
    output: Path,
    *,
    freeze_state: dict[str, Any],
    cartesian_state: dict[str, Any],
    overlay: Path,
) -> Path:
    report = base._rewrite_report(output, state=freeze_state)
    payload = json.loads(report.read_text(encoding="utf-8"))
    result = payload["result"]
    inherited = result.pop(
        "motion_canonical_response_only_freeze_contract"
    )
    grouping = result["batch"]["motion_damping_twin_grouping"]
    grouping.update(
        {
            "condition_rows_per_group": ROWS_PER_TWIN,
            "condition_pairs_per_group": 4,
            "action_branches_per_condition": 2,
            "real_future_cartesian_grid": True,
        }
    )
    checks = inherited["checks"]
    checks.update(
        {
            "overlay_sha_exact": _sha256(overlay) == OVERLAY_SHA256,
            "release_loaded_once": cartesian_state["release_calls"] == 1,
            "overlay_materialized_once": (
                cartesian_state["materialization_calls"] == 1
            ),
            "all_cartesian_grid_checks": bool(
                cartesian_state["grid_checks"]
                and all(cartesian_state["grid_checks"].values())
            ),
            "eight_row_twin_grouping": (
                grouping["condition_rows_per_group"] == ROWS_PER_TWIN
                and grouping["condition_pairs_per_group"] == 4
            ),
        }
    )
    if not all(checks.values()):
        raise RuntimeError(f"cartesian terminal contract failed: {checks}")
    inherited["checks"] = checks
    inherited["objective"] = (
        "native_mse_on_real_2x2_history_action_grid + 0.09*SIGReg + "
        "0.09*(canonical_history_response + canonical_assignment_0p5)"
    )
    inherited["cartesian_training_overlay"] = {
        "path": str(overlay),
        "sha256": OVERLAY_SHA256,
        "template_count": OVERLAY_TEMPLATE_COUNT,
        "condition_pair_count": OVERLAY_CONDITION_PAIR_COUNT,
        "rows_per_forward_reverse_twin": ROWS_PER_TWIN,
        "action_branches": ["observed_zero", "query_velocity_unit"],
        "all_futures_from_simulator": True,
        "training_only_frozen_teacher": False,
        "hidden_labels_at_model_boundary": False,
        "pair_metadata_at_model_boundary": False,
    }
    result["motion_cartesian_action_pair_contract"] = inherited
    payload["provenance"]["method"] = {
        "candidate": CANDIDATE,
        "source": str(THIS_SOURCE),
        "source_sha256": _sha256(THIS_SOURCE),
    }
    report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    effective = list(sys.argv[1:] if argv is None else argv)
    args = base._validate_args(effective)
    overlay = _parse_overlay(effective)
    trainer = base.causal._load_trainer()
    base.native_twin._install_runtime(trainer)
    cartesian_state = _install_cartesian_data(trainer, overlay=overlay)
    freeze_state = base._install_freeze_only(trainer)
    previous = list(sys.argv)
    try:
        trainer_args = _without_overlay_argument(effective)
        sys.argv = [str(THIS_SOURCE), *base.residual._trainer_argv(trainer_args)]
        trainer.main()
    finally:
        sys.argv = previous
    if not args.dry_run:
        output = args.output.expanduser().resolve()
        report = _rewrite_report(
            output,
            freeze_state=freeze_state,
            cartesian_state=cartesian_state,
            overlay=overlay,
        )
        sidecar = output / "motion_cartesian_action_pair_v1.json"
        if sidecar.exists():
            raise FileExistsError(sidecar)
        sidecar.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "candidate": CANDIDATE,
                    "source": str(THIS_SOURCE),
                    "source_sha256": _sha256(THIS_SOURCE),
                    "source_checkpoint_sha256": (
                        base.SOURCE_CHECKPOINT_SHA256
                    ),
                    "cartesian_overlay_sha256": OVERLAY_SHA256,
                    "fresh_optimizer_steps": OPTIMIZER_STEPS,
                    "training_report": str(report),
                    "training_report_sha256": _sha256(report),
                    "training_only_frozen_teacher": False,
                    "learned_parameters_added_to_saved_model": 0,
                    "model_modules_added_to_saved_model": 0,
                    "inference_compute_added": 0,
                    "single_seed_discovery": True,
                    "public_test_opened": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
