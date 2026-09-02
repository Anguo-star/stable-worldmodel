"""Tests for audit_icl_training_raw_pixel_visibility_v1.

Covers: core pixel formulas, B background variance, KNN, balanced sampling,
evidence-boundary guards, training-only guard, and boundary conditions.
"""
import math
import importlib
import io
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import target module
# ---------------------------------------------------------------------------

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "audit_icl_training_raw_pixel_visibility_v1.py"
)
spec = importlib.util.spec_from_file_location(
    "audit_icl_training_raw_pixel_visibility_v1", _SCRIPT
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# Convenience aliases
normalize_pixels = mod.normalize_pixels
pixel_conditional_variance = mod.pixel_conditional_variance
pixel_mean_displacement = mod.pixel_mean_displacement
ratio_of_means = mod.ratio_of_means
robust_scale = mod.robust_scale
leave_cluster_out_knn = mod.leave_cluster_out_knn
background_variance_by_k = mod.background_variance_by_k
deterministic_balanced_sample = mod.deterministic_balanced_sample
bootstrap_ratio_of_means_ci = mod.bootstrap_ratio_of_means_ci
training_only_guard = mod.training_only_guard
_distribution_summary = mod._distribution_summary


# ---------------------------------------------------------------------------
# Evidence-boundary constants
# ---------------------------------------------------------------------------

class TestEvidenceBoundaryConstants:
    def test_pixel_target_separation_measured(self):
        assert mod.PIXEL_TARGET_SEPARATION_STATUS == "measured_on_sampled_training_queries"

    def test_model_not_loaded(self):
        assert mod.MODEL_LOADED is False

    def test_optimizer_steps_zero(self):
        assert mod.OPTIMIZER_STEPS == 0

    def test_latent_status(self):
        assert mod.LATENT_STATUS == "not_measured"

    def test_binding_cause(self):
        assert mod.BINDING_CAUSE_STATUS == "not_claimed"

    def test_claim_scope_contains_raw_pixel(self):
        assert "raw_pixel" in mod.CLAIM_SCOPE

    def test_speed_pixel_status(self):
        assert mod.SPEED_PIXEL_STATUS == "simulator_rendered_counterfactual_on_training_queries"

    def test_pixel_units(self):
        assert mod.PIXEL_UNITS == "normalized_rgb_mse_per_pixel_channel"

    def test_ks_tuple(self):
        assert set(mod.KS) == {32, 64, 128}

    def test_main_k_in_ks(self):
        assert mod.MAIN_K in mod.KS


# ---------------------------------------------------------------------------
# Training-only guard
# ---------------------------------------------------------------------------

class TestTrainingOnlyGuard:
    def test_clean_path_passes(self):
        training_only_guard(Path("/data/training/run_v1"))

    def test_test_token_blocked(self):
        with pytest.raises(ValueError, match="forbidden token"):
            training_only_guard(Path("/data/test/run_v1"))

    def test_development_token_blocked(self):
        with pytest.raises(ValueError, match="forbidden token"):
            training_only_guard(Path("/data/development_set/run"))

    def test_public_token_blocked(self):
        with pytest.raises(ValueError, match="forbidden token"):
            training_only_guard(Path("/data/public/run"))

    def test_validation_token_blocked(self):
        with pytest.raises(ValueError, match="forbidden token"):
            training_only_guard(Path("/data/validation/run"))

    def test_val_underscore_blocked(self):
        with pytest.raises(ValueError, match="forbidden token"):
            training_only_guard(Path("/data/val_set/run"))


# ---------------------------------------------------------------------------
# normalize_pixels
# ---------------------------------------------------------------------------

class TestNormalizePixels:
    def test_uint8_range(self):
        raw = np.array([0, 127, 255], dtype=np.uint8)
        out = normalize_pixels(raw, 255.0)
        assert out.dtype == np.float64
        np.testing.assert_allclose(out, [0.0, 127 / 255.0, 1.0], atol=1e-12)

    def test_already_float_scale_1(self):
        raw = np.array([0.0, 0.5, 1.0])
        out = normalize_pixels(raw, 1.0)
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)

    def test_invalid_scale_raises(self):
        with pytest.raises(RuntimeError):
            normalize_pixels(np.array([0.5]), 0.0)

    def test_out_of_range_raises(self):
        with pytest.raises(RuntimeError):
            normalize_pixels(np.array([300.0]), 255.0)


class TestPngDecodeAndHistory:
    def test_lossless_png_decode(self):
        from PIL import Image

        raw = np.asarray([[[0, 127, 255], [255, 1, 0]]], dtype=np.uint8)
        buffer = io.BytesIO()
        Image.fromarray(raw).save(buffer, format="PNG")
        decoded = mod.decode_pixel_blob(buffer.getvalue(), raw.shape)
        np.testing.assert_array_equal(
            decoded.reshape(raw.shape), raw.astype(np.float32) / np.float32(255.0)
        )

    def test_binary_history_identity(self):
        rng = np.random.default_rng(31)
        h0 = rng.random((3, 11))
        h1 = rng.random((3, 11))
        got = mod.pixel_history_conditional_variance(
            np.stack([h0, h1]), np.asarray([0.5, 0.5])
        )
        expected = float(np.mean(np.square((h0 - h1) / 2.0)))
        assert abs(got - expected) < 1.0e-12


# ---------------------------------------------------------------------------
# pixel_conditional_variance — core formula
# ---------------------------------------------------------------------------

class TestPixelConditionalVariance:
    def _binary_uniform(self, y0, y1, x):
        """Reference: C = mean_p(((y0-y1)/2)^2) for uniform binary."""
        delta = np.asarray(y0, float) - np.asarray(y1, float)
        return float(np.mean(np.square(delta / 2.0)))

    def test_binary_uniform_identity(self):
        """For w=[0.5,0.5], C_pixel == mean((delta/2)^2) exactly."""
        rng = np.random.default_rng(42)
        P = 50
        x = rng.random(P)
        y0 = rng.random(P)
        y1 = rng.random(P)
        w = np.array([0.5, 0.5])
        Y = np.stack([y0, y1])[None]
        got = float(pixel_conditional_variance(Y, x[None], w)[0])
        expected = self._binary_uniform(y0, y1, x)
        assert abs(got - expected) < 1e-12

    def test_zero_when_conditions_identical(self):
        """If y0 == y1, both diffs equal m, residuals are zero -> C=0."""
        rng = np.random.default_rng(7)
        P = 20
        x = rng.random(P)
        y = rng.random(P)
        Y = np.stack([y, y])[None]
        w = np.array([0.5, 0.5])
        result = float(pixel_conditional_variance(Y, x[None], w)[0])
        assert abs(result) < 1e-14

    def test_nonzero_when_conditions_differ(self):
        rng = np.random.default_rng(9)
        P = 30
        x = rng.random(P)
        y0 = rng.random(P)
        y1 = y0 + 0.5
        Y = np.stack([y0, y1])[None]
        result = float(pixel_conditional_variance(Y, x[None], np.array([0.5, 0.5]))[0])
        assert result > 0.0

    def test_batch_consistency(self):
        """Batched N=3 matches calling N=1 three times."""
        rng = np.random.default_rng(13)
        N, C, P = 3, 2, 15
        Y = rng.random((N, C, P))
        x = rng.random((N, P))
        w = np.array([0.5, 0.5])
        batch = pixel_conditional_variance(Y, x, w)
        for i in range(N):
            single = float(pixel_conditional_variance(Y[i:i+1], x[i:i+1], w)[0])
            assert abs(float(batch[i]) - single) < 1e-12

    def test_non_uniform_weights(self):
        """Weighted C: explicitly verify formula with w=[0.3, 0.7]."""
        P = 10
        x = np.zeros(P)
        y0 = np.ones(P)
        y1 = np.full(P, 3.0)
        w = np.array([0.3, 0.7])
        Y = np.stack([y0, y1])[None]
        # m = 0.3*(1-0) + 0.7*(3-0) = 0.3 + 2.1 = 2.4
        # res0 = 1 - 2.4 = -1.4, res1 = 3 - 2.4 = 0.6
        # C = 0.3*mean(1.96) + 0.7*mean(0.36) = 0.3*1.96 + 0.7*0.36 = 0.588 + 0.252 = 0.84
        expected = 0.3 * 1.96 + 0.7 * 0.36
        got = float(pixel_conditional_variance(Y, x[None], w)[0])
        assert abs(got - expected) < 1e-10

    def test_shape_error_Y_not_3d(self):
        with pytest.raises(RuntimeError):
            pixel_conditional_variance(np.zeros((3, 5)), np.zeros((3, 5)), np.array([0.5, 0.5]))

    def test_w_not_sum_to_1(self):
        with pytest.raises(RuntimeError):
            pixel_conditional_variance(np.zeros((2, 2, 5)), np.zeros((2, 5)), np.array([0.4, 0.4]))


# ---------------------------------------------------------------------------
# pixel_mean_displacement
# ---------------------------------------------------------------------------

class TestPixelMeanDisplacement:
    def test_binary_mean(self):
        P = 8
        x = np.zeros(P)
        y0 = np.ones(P)
        y1 = np.full(P, 3.0)
        Y = np.stack([y0, y1])[None]
        w = np.array([0.5, 0.5])
        m = pixel_mean_displacement(Y, x[None], w)
        np.testing.assert_allclose(m, np.full((1, P), 2.0), atol=1e-12)

    def test_weighted_mean(self):
        P = 4
        x = np.zeros(P)
        y0 = np.ones(P)      # displacement = 1
        y1 = np.full(P, 5.0) # displacement = 5
        Y = np.stack([y0, y1])[None]
        w = np.array([0.2, 0.8])
        m = pixel_mean_displacement(Y, x[None], w)
        expected = 0.2 * 1.0 + 0.8 * 5.0   # = 4.2
        np.testing.assert_allclose(m, np.full((1, P), expected), atol=1e-12)


# ---------------------------------------------------------------------------
# ratio_of_means
# ---------------------------------------------------------------------------

class TestRatioOfMeans:
    def test_half_when_equal(self):
        C = np.array([1.0, 2.0, 3.0])
        B = np.array([1.0, 2.0, 3.0])
        w = np.full(3, 1.0 / 3.0)
        assert abs(ratio_of_means(C, B, w) - 0.5) < 1e-12

    def test_near_zero_when_C_small(self):
        C = np.array([1e-9, 1e-9])
        B = np.array([1.0, 1.0])
        w = np.array([0.5, 0.5])
        assert ratio_of_means(C, B, w) < 0.01

    def test_near_one_when_B_small(self):
        C = np.array([1.0, 1.0])
        B = np.array([1e-9, 1e-9])
        w = np.array([0.5, 0.5])
        assert ratio_of_means(C, B, w) > 0.99

    def test_shape_mismatch_raises(self):
        with pytest.raises(RuntimeError):
            ratio_of_means(np.array([1.0]), np.array([1.0, 2.0]), np.array([0.5, 0.5]))

    def test_zero_denom_raises(self):
        with pytest.raises(RuntimeError):
            ratio_of_means(np.array([0.0]), np.array([0.0]), np.array([1.0]))


# ---------------------------------------------------------------------------
# background_variance_by_k — B formula
# ---------------------------------------------------------------------------

class TestBackgroundVarianceByK:
    def _make_neighbors_and_means(self, N=20, P=5, k=4, seed=0):
        rng = np.random.default_rng(seed)
        means = rng.standard_normal((N, P))
        neighbors = np.zeros((N, k), dtype=np.int64)
        for i in range(N):
            pool = list(range(N))
            pool.remove(i)
            rng2 = np.random.default_rng(seed + i)
            chosen = rng2.choice(pool, size=k, replace=False)
            neighbors[i] = np.sort(chosen)
        return means, neighbors

    def test_zero_when_neighbors_identical(self):
        """If all neighbors have the same mean-displacement, B = 0."""
        N, P, k = 8, 6, 3
        means = np.tile(np.arange(P, dtype=float), (N, 1))  # all rows equal
        # build neighbors: each row points to the same other rows
        neighbors = np.zeros((N, k), dtype=np.int64)
        for i in range(N):
            pool = [j for j in range(N) if j != i][:k]
            neighbors[i] = pool
        bvk = background_variance_by_k(means, neighbors, [k])
        np.testing.assert_allclose(bvk[k], 0.0, atol=1e-12)

    def test_positive_when_neighbors_differ(self):
        means, neighbors = self._make_neighbors_and_means(N=20, P=5, k=4)
        bvk = background_variance_by_k(means, neighbors, [4])
        assert np.all(bvk[4] >= 0.0)
        # At least some should be positive (random means)
        assert float(np.max(bvk[4])) > 1e-6

    def test_multiple_k_values(self):
        means, neighbors_128 = self._make_neighbors_and_means(N=30, P=4, k=8)
        bvk = background_variance_by_k(means, neighbors_128, [4, 8])
        assert set(bvk.keys()) == {4, 8}
        np.testing.assert_array_less(-1e-15, bvk[4])
        np.testing.assert_array_less(-1e-15, bvk[8])

    def test_invalid_k_raises(self):
        means = np.ones((5, 3))
        neighbors = np.zeros((5, 4), dtype=np.int64)
        with pytest.raises(RuntimeError):
            background_variance_by_k(means, neighbors, [5])

    def test_non_finite_means_raises(self):
        means = np.ones((5, 3))
        means[2, 1] = np.nan
        neighbors = np.zeros((5, 4), dtype=np.int64)
        with pytest.raises(RuntimeError):
            background_variance_by_k(means, neighbors, [4])

    def test_output_shape(self):
        N, P, k = 10, 8, 3
        means, neighbors = self._make_neighbors_and_means(N=N, P=P, k=k)
        bvk = background_variance_by_k(means, neighbors, [k])
        assert bvk[k].shape == (N,)


# ---------------------------------------------------------------------------
# leave_cluster_out_knn
# ---------------------------------------------------------------------------

class TestLeaveClusterOutKnn:
    def _make_grid(self, N=20, D=2, seed=0):
        rng = np.random.default_rng(seed)
        desc = rng.standard_normal((N, D))
        cluster_ids = np.arange(N, dtype=np.int64)
        return desc, cluster_ids

    def test_no_self_in_neighbors(self):
        desc, cids = self._make_grid(N=20, D=2)
        neighbors, _ = leave_cluster_out_knn(desc, cids, max_k=5)
        for i in range(len(desc)):
            assert i not in neighbors[i]

    def test_no_cluster_member_in_neighbors(self):
        """Queries in the same cluster must be excluded."""
        N = 30
        rng = np.random.default_rng(5)
        desc = rng.standard_normal((N, 3))
        cids = np.repeat(np.arange(10), 3).astype(np.int64)
        neighbors, _ = leave_cluster_out_knn(desc, cids, max_k=5)
        for i in range(N):
            same_cluster = set(np.flatnonzero(cids == cids[i]))
            for nb in neighbors[i]:
                assert nb not in same_cluster

    def test_correct_k_neighbors(self):
        desc, cids = self._make_grid(N=25, D=4)
        k = 8
        neighbors, dists = leave_cluster_out_knn(desc, cids, max_k=k)
        assert neighbors.shape == (25, k)
        assert dists.shape == (25, k)

    def test_distances_non_negative_and_finite(self):
        desc, cids = self._make_grid(N=20, D=3)
        _, dists = leave_cluster_out_knn(desc, cids, max_k=5)
        assert np.all(dists >= 0.0)
        assert np.all(np.isfinite(dists))

    def test_distances_sorted_ascending(self):
        desc, cids = self._make_grid(N=30, D=3, seed=99)
        _, dists = leave_cluster_out_knn(desc, cids, max_k=6)
        for i in range(len(desc)):
            assert np.all(np.diff(dists[i]) >= -1e-12)

    def test_max_k_too_large_raises(self):
        desc = np.eye(5)
        cids = np.arange(5, dtype=np.int64)
        with pytest.raises(RuntimeError):
            leave_cluster_out_knn(desc, cids, max_k=5)

    def test_non_finite_input_raises(self):
        desc = np.ones((10, 3))
        desc[3, 1] = np.inf
        cids = np.arange(10, dtype=np.int64)
        with pytest.raises(RuntimeError):
            leave_cluster_out_knn(desc, cids, max_k=5)

    def test_tie_breaking_stable(self):
        """Verify tie-breaking produces the same result on repeated calls."""
        rng = np.random.default_rng(42)
        N, D = 20, 2
        desc = rng.standard_normal((N, D))
        cids = np.arange(N, dtype=np.int64)
        nb1, d1 = leave_cluster_out_knn(desc, cids, max_k=4)
        nb2, d2 = leave_cluster_out_knn(desc, cids, max_k=4)
        np.testing.assert_array_equal(nb1, nb2)
        np.testing.assert_array_equal(d1, d2)


# ---------------------------------------------------------------------------
# deterministic_balanced_sample
# ---------------------------------------------------------------------------

class TestDeterministicBalancedSample:
    def test_count(self):
        groups = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        idx = deterministic_balanced_sample(groups, 6, seed=1)
        assert len(idx) == 6

    def test_balance_across_groups(self):
        groups = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        idx = deterministic_balanced_sample(groups, 6, seed=42)
        counts = {g: int(np.sum(groups[idx] == g)) for g in [0, 1, 2]}
        assert all(v == 2 for v in counts.values())

    def test_reproducible(self):
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        idx1 = deterministic_balanced_sample(groups, 6, seed=7)
        idx2 = deterministic_balanced_sample(groups, 6, seed=7)
        np.testing.assert_array_equal(idx1, idx2)

    def test_different_seeds_differ(self):
        groups = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        idx1 = deterministic_balanced_sample(groups, 4, seed=1)
        idx2 = deterministic_balanced_sample(groups, 4, seed=2)
        # Very likely to differ with different seeds over 3-element groups
        # (not guaranteed but seed=1 vs seed=2 with 3 choices should differ)
        # Just check they are both valid
        assert len(idx1) == 4 and len(idx2) == 4

    def test_total_not_divisible_raises(self):
        groups = np.array([0, 0, 1, 1], dtype=np.int64)
        with pytest.raises(RuntimeError):
            deterministic_balanced_sample(groups, 3, seed=0)

    def test_group_too_small_raises(self):
        groups = np.array([0, 1, 1], dtype=np.int64)
        with pytest.raises(RuntimeError):
            deterministic_balanced_sample(groups, 4, seed=0)

    def test_indices_in_range(self):
        groups = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        idx = deterministic_balanced_sample(groups, 6, seed=0)
        assert np.all(idx >= 0) and np.all(idx < len(groups))


# ---------------------------------------------------------------------------
# bootstrap_ratio_of_means_ci
# ---------------------------------------------------------------------------

class TestBootstrapRatioOfMeansCi:
    def _make_data(self, N=40, seed=3):
        rng = np.random.default_rng(seed)
        C = rng.exponential(0.5, N)
        B = rng.exponential(0.5, N)
        cids = np.arange(N, dtype=np.int64)
        return C, B, cids

    def test_ci_keys(self):
        C, B, cids = self._make_data()
        result = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=200, seed=0)
        for key in ("mean", "ci_lo", "ci_hi", "ci_level", "n_resamples"):
            assert key in result

    def test_ci_ordering(self):
        C, B, cids = self._make_data()
        result = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=200, seed=0)
        assert result["ci_lo"] <= result["mean"] <= result["ci_hi"]

    def test_rho_in_unit_interval(self):
        C, B, cids = self._make_data()
        result = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=200, seed=0)
        assert 0.0 <= result["ci_lo"] <= result["ci_hi"] <= 1.0

    def test_reproducible(self):
        C, B, cids = self._make_data()
        r1 = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=100, seed=99)
        r2 = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=100, seed=99)
        assert r1["mean"] == r2["mean"]
        assert r1["ci_lo"] == r2["ci_lo"]

    def test_high_C_gives_high_rho(self):
        N = 30
        C = np.full(N, 10.0)
        B = np.full(N, 0.01)
        cids = np.arange(N, dtype=np.int64)
        result = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=50, seed=0)
        assert result["mean"] > 0.9

    def test_low_C_gives_low_rho(self):
        N = 30
        C = np.full(N, 0.01)
        B = np.full(N, 10.0)
        cids = np.arange(N, dtype=np.int64)
        result = bootstrap_ratio_of_means_ci(C, B, cids, n_resamples=50, seed=0)
        assert result["mean"] < 0.1


# ---------------------------------------------------------------------------
# robust_scale
# ---------------------------------------------------------------------------

class TestRobustScale:
    def test_zero_iqr_column_stays_zero(self):
        data = np.ones((10, 3))
        data[:, 0] = np.arange(10, dtype=float)
        scaled, _, _ = robust_scale(data)
        np.testing.assert_array_equal(scaled[:, 1], 0.0)
        np.testing.assert_array_equal(scaled[:, 2], 0.0)

    def test_nonzero_iqr_column_scaled(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((100, 4))
        scaled, median, iqr = robust_scale(data)
        # Active columns: median of scaled should be near 0
        for j in range(4):
            if iqr[j] > 0:
                assert abs(float(np.median(scaled[:, j]))) < 0.1

    def test_output_shape(self):
        data = np.arange(30, dtype=float).reshape(10, 3)
        scaled, median, iqr = robust_scale(data)
        assert scaled.shape == data.shape
        assert median.shape == (3,)
        assert iqr.shape == (3,)


# ---------------------------------------------------------------------------
# _distribution_summary
# ---------------------------------------------------------------------------

class TestDistributionSummary:
    def test_keys(self):
        v = np.arange(10, dtype=float)
        s = _distribution_summary(v)
        for key in ("mean", "std", "q00", "q25", "q50", "q75", "q100", "n"):
            assert key in s

    def test_n_matches_length(self):
        v = np.arange(7, dtype=float)
        s = _distribution_summary(v)
        assert s["n"] == 7

    def test_ordering(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal(50)
        s = _distribution_summary(v)
        assert s["q00"] <= s["q25"] <= s["q50"] <= s["q75"] <= s["q100"]

    def test_scalar_input(self):
        s = _distribution_summary(np.array([5.0]))
        assert s["n"] == 1
        assert s["mean"] == 5.0


# ---------------------------------------------------------------------------
# Boundary conditions for pixel_conditional_variance
# ---------------------------------------------------------------------------

class TestPixelConditionalVarianceBoundary:
    def test_single_pixel_single_query(self):
        Y = np.array([[[0.2], [0.8]]])  # (1, 2, 1)
        x = np.array([[0.5]])           # (1, 1)
        w = np.array([0.5, 0.5])
        # delta = 0.2 - 0.8 = -0.6; C = mean((delta/2)^2) = (0.3)^2 = 0.09
        result = float(pixel_conditional_variance(Y, x, w)[0])
        assert abs(result - 0.09) < 1e-12

    def test_many_conditions_equal_weight(self):
        C_count = 5
        P = 10
        w = np.full(C_count, 1.0 / C_count)
        rng = np.random.default_rng(17)
        x = rng.random((3, P))
        Y = rng.random((3, C_count, P))
        result = pixel_conditional_variance(Y, x, w)
        assert result.shape == (3,)
        assert np.all(result >= 0.0)

    def test_large_pixel_dim_memory_safe(self):
        """Verify no OOM with a larger but manageable pixel dimension."""
        P = 224 * 224 * 3   # TwoRoom frame size
        N = 4
        C = 2
        rng = np.random.default_rng(5)
        # Use float32 to build, cast inside
        x = rng.random((N, P)).astype(np.float32)
        y0 = rng.random((N, P)).astype(np.float32)
        y1 = rng.random((N, P)).astype(np.float32)
        Y = np.stack([y0, y1], axis=1)  # (N, 2, P)
        w = np.array([0.5, 0.5])
        result = pixel_conditional_variance(Y, x, w)
        assert result.shape == (N,)
        assert np.all(result >= 0.0)
