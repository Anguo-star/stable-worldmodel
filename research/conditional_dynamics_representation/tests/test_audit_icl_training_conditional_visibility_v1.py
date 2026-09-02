#!/usr/bin/env python3
"""Unit tests for audit_icl_training_conditional_visibility_v1.

All tests use small synthetic data — no real Training tables are read.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the audit module without executing main()
# ---------------------------------------------------------------------------
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_icl_training_conditional_visibility_v1.py"
)


def _load_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("audit_icl_v1", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()


# ---------------------------------------------------------------------------
# 1. Weighted conditional variance — binary identity
# ---------------------------------------------------------------------------
class TestWeightedConditionalVariance:
    """C_u for binary uniform must equal ||Δy||^2 / 4."""

    def test_binary_identity(self):
        rng = np.random.default_rng(0)
        N = 20
        # Two conditions, equal weight 0.5
        y0 = rng.normal(size=(N, 2))
        y1 = rng.normal(size=(N, 2))
        x = rng.normal(size=(N, 2))
        w = np.array([0.5, 0.5])

        C_expected = np.sum((y1 - y0) ** 2, axis=1) / 4.0

        C_got = _mod.weighted_conditional_variance(
            np.stack([y0, y1], axis=1),  # shape (N, 2, D)
            x,
            w,
        )
        np.testing.assert_allclose(C_got, C_expected, rtol=1e-12)

    def test_multi_condition_weighted(self):
        # 3 conditions with weights [0.2, 0.3, 0.5]
        rng = np.random.default_rng(1)
        N = 10
        D = 3
        Y = rng.normal(size=(N, 3, D))   # (N, C, D)
        x = rng.normal(size=(N, D))
        w = np.array([0.2, 0.3, 0.5])

        m = np.einsum("c,ncd->nd", w, Y - x[:, None, :])
        diffs = (Y - x[:, None, :]) - m[:, None, :]
        C_expected = np.einsum("c,ncd,ncd->n", w, diffs, diffs)

        C_got = _mod.weighted_conditional_variance(Y, x, w)
        np.testing.assert_allclose(C_got, C_expected, rtol=1e-12)

    def test_single_condition_zero(self):
        # One condition → m equals the shift, C = 0
        rng = np.random.default_rng(2)
        Y = rng.normal(size=(5, 1, 2))
        x = rng.normal(size=(5, 2))
        C = _mod.weighted_conditional_variance(Y, x, np.array([1.0]))
        np.testing.assert_allclose(C, 0.0, atol=1e-12)

    def test_history_binary_identity_averaged_over_time(self):
        rng = np.random.default_rng(21)
        h0 = rng.normal(size=(7, 3, 2))
        h1 = rng.normal(size=(7, 3, 2))
        got = _mod.weighted_history_conditional_variance(
            np.stack([h0, h1], axis=1), np.array([0.5, 0.5])
        )
        expected = np.mean(np.sum((h1 - h0) ** 2, axis=2) / 4.0, axis=1)
        np.testing.assert_allclose(got, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Ratio-of-means
# ---------------------------------------------------------------------------
class TestRatioOfMeans:
    def test_basic(self):
        C = np.array([1.0, 2.0, 3.0])
        B = np.array([1.0, 1.0, 1.0])
        w = np.ones(3) / 3.0
        rho = _mod.ratio_of_means(C, B, w)
        # E[C] = 2, E[B] = 1, rho = 2/3
        assert math.isclose(rho, 2.0 / 3.0, rel_tol=1e-12)

    def test_not_per_sample_mean(self):
        # ratio of means ≠ mean of per-sample ratios in general
        C = np.array([1.0, 9.0])
        B = np.array([1.0, 1.0])
        w = np.ones(2) / 2.0
        rho = _mod.ratio_of_means(C, B, w)
        # E[C]=5, E[B]=1, rho=5/6 ≈ 0.833
        assert math.isclose(rho, 5.0 / 6.0, rel_tol=1e-12)
        # mean of per-sample: (1/2+9/10)/2 = (0.5+0.9)/2 = 0.7
        assert rho != 0.7


# ---------------------------------------------------------------------------
# 3. Leave-cluster-out KNN
# ---------------------------------------------------------------------------
class TestLeaveClusterOut:
    def test_no_self_cluster_in_neighbors(self):
        rng = np.random.default_rng(3)
        N = 30
        D = 4
        desc = rng.normal(size=(N, D))
        # 10 clusters of 3 each
        cluster_ids = np.repeat(np.arange(10), 3)

        neighbors, distances = _mod.leave_cluster_out_knn(
            desc, cluster_ids, max_k=5, chunk_size=10
        )
        for i in range(N):
            own_cluster = cluster_ids[i]
            neighbor_clusters = cluster_ids[neighbors[i]]
            assert not np.any(neighbor_clusters == own_cluster), (
                f"row {i}: own cluster {own_cluster} appears in neighbors"
            )

    def test_distances_sorted_ascending(self):
        rng = np.random.default_rng(4)
        N = 20
        desc = rng.normal(size=(N, 3))
        cluster_ids = np.repeat(np.arange(10), 2)
        _, distances = _mod.leave_cluster_out_knn(
            desc, cluster_ids, max_k=4, chunk_size=20
        )
        for row in distances:
            assert np.all(np.diff(row) >= 0), "distances not sorted"

    def test_background_variance_nonneg(self):
        rng = np.random.default_rng(5)
        N = 24
        desc = rng.normal(size=(N, 4))
        means = rng.normal(size=(N, 2))
        cluster_ids = np.repeat(np.arange(8), 3)
        neighbors, _ = _mod.leave_cluster_out_knn(
            desc, cluster_ids, max_k=8, chunk_size=24
        )
        bvk = _mod.background_variance_by_k(means, neighbors, [4, 8])
        for k, vals in bvk.items():
            assert np.all(vals >= 0.0), f"B_{k} has negative values"


# ---------------------------------------------------------------------------
# 4. Deterministic sampling
# ---------------------------------------------------------------------------
class TestDeterministicSampling:
    def test_same_seed_same_result(self):
        a = _mod.deterministic_balanced_sample(
            group_ids=np.repeat(np.arange(4), 8),
            total=16,
            seed=20260901,
        )
        b = _mod.deterministic_balanced_sample(
            group_ids=np.repeat(np.arange(4), 8),
            total=16,
            seed=20260901,
        )
        np.testing.assert_array_equal(a, b)

    def test_balanced_groups(self):
        group_ids = np.repeat(np.arange(8), 16)
        selected = _mod.deterministic_balanced_sample(
            group_ids=group_ids,
            total=32,
            seed=20260901,
        )
        counts = np.bincount(group_ids[selected], minlength=8)
        assert np.all(counts == counts[0]), "groups not balanced"

    def test_different_seed_different(self):
        g = np.repeat(np.arange(4), 8)
        a = _mod.deterministic_balanced_sample(g, 16, seed=1)
        b = _mod.deterministic_balanced_sample(g, 16, seed=2)
        assert not np.array_equal(a, b)


# ---------------------------------------------------------------------------
# 5. Training-only path guard
# ---------------------------------------------------------------------------
class TestPathGuard:
    def test_rejects_development(self):
        with pytest.raises(ValueError, match="training-only"):
            _mod.training_only_guard(
                Path("/some/synthesis/pusht_action_strength_development_v1/train.lance")
            )

    def test_rejects_public(self):
        with pytest.raises(ValueError, match="training-only"):
            _mod.training_only_guard(Path("/data/public/train.lance"))

    def test_rejects_test(self):
        with pytest.raises(ValueError, match="training-only"):
            _mod.training_only_guard(Path("/data/test/something.json"))

    def test_rejects_validation(self):
        with pytest.raises(ValueError, match="training-only"):
            _mod.training_only_guard(Path("/data/validation/train.lance"))

    def test_rejects_val(self):
        with pytest.raises(ValueError, match="training-only"):
            _mod.training_only_guard(Path("/splits/val/train.lance"))

    def test_allows_release(self):
        # 'release' in path name is allowed
        _mod.training_only_guard(
            Path("/ContextWorld/artifacts/synthesis/pusht_motion_damping_h3_release_v4/train.lance")
        )

    def test_allows_train(self):
        _mod.training_only_guard(Path("/data/train.lance"))


# ---------------------------------------------------------------------------
# 6. Exclusive output (no overwrite)
# ---------------------------------------------------------------------------
class TestExclusiveOutput:
    def test_raises_if_exists(self, tmp_path):
        existing = tmp_path / "out"
        existing.mkdir()
        with pytest.raises(FileExistsError):
            _mod.exclusive_mkdir(existing)

    def test_creates_if_absent(self, tmp_path):
        target = tmp_path / "newdir"
        _mod.exclusive_mkdir(target)
        assert target.is_dir()


# ---------------------------------------------------------------------------
# 7. Speed evidence label
# ---------------------------------------------------------------------------
class TestSpeedEvidenceLabel:
    def test_label_is_counterfactual(self):
        assert _mod.SPEED_EVIDENCE_TYPE == "simulator_counterfactual_on_training_queries"

    def test_not_observed_matched_twin(self):
        assert "observed_matched_twin" not in _mod.SPEED_EVIDENCE_TYPE
        assert "observed" not in _mod.SPEED_EVIDENCE_TYPE

    def test_manifest_speed_uses_frozen_factors_field(self):
        row = {"factors": {"agent.speed": 4.3}, "scenario_id": "train-only"}
        assert _mod._manifest_speed(row) == 4.3


class TestTwoRoomCollision:
    def test_border_clamp_accounts_for_agent_radius(self):
        got = _mod._tworoom_collision(
            np.array([50.0, 50.0]), np.array([-10.0, 300.0])
        )
        np.testing.assert_allclose(got, [21.0, 203.0])

    def test_wall_blocks_outside_door(self):
        got = _mod._tworoom_collision(
            np.array([95.0, 100.0]), np.array([105.0, 100.0])
        )
        np.testing.assert_allclose(got, [99.5, 100.0])

    def test_wall_allows_crossing_through_door(self):
        got = _mod._tworoom_collision(
            np.array([99.0, 49.0]), np.array([105.0, 49.0])
        )
        np.testing.assert_allclose(got, [105.0, 49.0])


# ---------------------------------------------------------------------------
# 8. Boundary assertions: no pixel / model / optimizer
# ---------------------------------------------------------------------------
class TestBoundaries:
    def test_boundary_constants(self):
        assert _mod.PIXELS_DECODED is False
        assert _mod.MODEL_LOADED is False
        assert _mod.OPTIMIZER_STEPS == 0

    def test_pixel_status_string(self):
        assert _mod.PIXEL_TARGET_SEPARATION_STATUS == "pending_separate_raw_pixel_audit"

    def test_history_inferability_status(self):
        assert _mod.HISTORY_INFERABILITY_STATUS == (
            "pending_separate_full_observation_or_raw_pixel_audit"
        )

    def test_latent_status_string(self):
        assert _mod.LATENT_STATUS == "pending_gpu_forward"

    def test_binding_cause_status(self):
        assert _mod.BINDING_CAUSE_STATUS == "not_claimed"

    def test_claim_scope(self):
        assert "frozen_training_only" in _mod.CLAIM_SCOPE


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
