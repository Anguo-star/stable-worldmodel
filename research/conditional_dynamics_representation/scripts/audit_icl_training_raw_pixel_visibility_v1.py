#!/usr/bin/env python3
"""Training-only raw-pixel conditional visibility CPU auditor (v1).

Measures whether physical conditional energy is detectable in frozen Training
pools in raw RGB pixel space across four conditional-dynamics tasks using
per-pixel-channel MSE, leave-cluster-out KNN, and cluster bootstrap. Speed is
reported separately as simulator-rendered counterfactual evidence because its
Training pool has no observed same-query cross-speed twins.

Evidence boundary (immutable):
  pixel_target_separation_status = measured_on_sampled_training_queries
  latent_status                  = not_measured
  binding_cause_status           = not_claimed
  claim_scope                    = frozen_training_only_raw_pixel_descriptive_upstream_visibility

KNN descriptors are the same physical-state descriptors used in v1;
pixel C / B / rho are computed in pixel space only.
No learned model is loaded and no optimizer step is taken. Full runs decode
only the selected lossless Training PNG frames; check-only runs decode none.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import io
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Evidence boundary — immutable module-level constants (read by tests)
# ---------------------------------------------------------------------------
MODEL_LOADED: bool = False
OPTIMIZER_STEPS: int = 0
PIXEL_TARGET_SEPARATION_STATUS: str = "measured_on_sampled_training_queries"
LATENT_STATUS: str = "not_measured"
BINDING_CAUSE_STATUS: str = "not_claimed"
CLAIM_SCOPE: str = "sampled_training_only_descriptive_raw_pixel_visibility"
SPEED_PIXEL_STATUS: str = "simulator_rendered_counterfactual_on_training_queries"
PIXEL_UNITS: str = "normalized_rgb_mse_per_pixel_channel"

KS: tuple[int, ...] = (32, 64, 128)
MAIN_K: int = 64
NEIGHBOR_CHUNK_SIZE: int = 128
BOOTSTRAP_RESAMPLES: int = 2048
BOOTSTRAP_SEED: int = 20260901
SAMPLE_PER_TASK: int = 256

_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {"development", "public", "test", "validation", "val"}
)


# ---------------------------------------------------------------------------
# Guards and utilities
# ---------------------------------------------------------------------------

def training_only_guard(path: Path) -> None:
    """Raise ValueError if any path token matches a forbidden split name."""
    tokens: set[str] = set()
    for part in path.parts:
        tokens.update(part.replace("-", "_").lower().split("_"))
        tokens.add(part.lower())
    bad = tokens & _FORBIDDEN_TOKENS
    if bad:
        raise ValueError(
            f"training-only guard: forbidden token(s) {sorted(bad)} in path {path}"
        )


def exclusive_mkdir(path: Path) -> None:
    """Create directory; raise FileExistsError if it already exists."""
    path.mkdir(parents=True, exist_ok=False)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _finite(name: str, value: "np.ndarray") -> None:
    if not bool(np.isfinite(value).all()):
        raise RuntimeError(f"{name} contains non-finite values")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_storage_digest(rows: Iterable[dict[str, Any]]) -> str:
    """Hash manifest-declared storage identities without rereading pixel payloads."""
    digest = hashlib.sha256()
    normalized: list[tuple[str, str]] = []
    for row in rows:
        identity = str(row.get("scenario_id", row.get("pair_id", "")))
        storage = str(row.get("storage_sha256", row.get("fingerprint", "")))
        normalized.append((identity, storage))
    for identity, storage in sorted(normalized):
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
        digest.update(storage.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_dump(path: Path, obj: Any) -> None:
    with path.open("x", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Pixel normalisation  (read by tests)
# ---------------------------------------------------------------------------

def normalize_pixels(raw: "np.ndarray", pixel_scale: float) -> "np.ndarray":
    """Divide stored pixel values by pixel_scale to reach [0, 1] per channel.

    raw may be integer or float; output is float64 in [0, 1].
    """
    _require(pixel_scale > 0.0, "pixel_scale must be positive")
    result = np.asarray(raw, dtype=np.float64) / float(pixel_scale)
    _require(
        bool((result >= 0.0).all() and (result <= 1.0 + 1e-6).all()),
        f"normalized pixels out of [0,1]: min={float(result.min()):.4f} max={float(result.max()):.4f}",
    )
    return result


# ---------------------------------------------------------------------------
# Core math — pixel conditional variance and background variance
# ---------------------------------------------------------------------------

def pixel_conditional_variance(
    Y_pix: "np.ndarray",   # (N, C, P) — P = H*W*C_channels, already in [0,1]
    x_pix: "np.ndarray",   # (N, P)
    w: "np.ndarray",       # (C,) — must sum to 1
) -> "np.ndarray":
    """Per-query pixel conditional variance using per-pixel-channel MSE.

    C_pixel_u = sum_c w_c * mean_p((y_uc_p - x_u_p - m_u_p)^2)

    where m_u_p = sum_c w_c (y_uc_p - x_u_p) is the weighted mean displacement
    per pixel-channel and the outer mean is over pixel-channels P.
    """
    Y = np.asarray(Y_pix, dtype=np.float64)
    x = np.asarray(x_pix, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    _require(Y.ndim == 3, "Y_pix must be (N, C, P)")
    _require(x.ndim == 2 and x.shape == (Y.shape[0], Y.shape[2]), "x_pix shape mismatch")
    _require(w.ndim == 1 and len(w) == Y.shape[1], "w shape mismatch")
    _require(math.isclose(float(w.sum()), 1.0, abs_tol=1e-9), "w must sum to 1")
    diffs = Y - x[:, None, :]                          # (N, C, P)
    m = np.einsum("c,ncp->np", w, diffs)               # (N, P) — weighted mean displacement
    residuals = diffs - m[:, None, :]                  # (N, C, P)
    sq_mean_p = np.mean(np.square(residuals), axis=2)  # (N, C)  — mean over pixels
    return np.einsum("c,nc->n", w, sq_mean_p)          # (N,)


def pixel_mean_displacement(
    Y_pix: "np.ndarray",  # (N, C, P)
    x_pix: "np.ndarray",  # (N, P)
    w: "np.ndarray",      # (C,)
) -> "np.ndarray":
    """Weighted mean displacement m_u = sum_c w_c (y_uc - x_u), shape (N, P)."""
    Y = np.asarray(Y_pix, dtype=np.float64)
    x = np.asarray(x_pix, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    diffs = Y - x[:, None, :]                # (N, C, P)
    return np.einsum("c,ncp->np", w, diffs)  # (N, P)


def ratio_of_means(
    C: "np.ndarray",
    B: "np.ndarray",
    w: "np.ndarray",
) -> float:
    """rho = E[C] / (E[C] + E[B]) — ratio of weighted means, not mean of ratios."""
    C = np.asarray(C, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    _require(C.shape == B.shape == w.shape, "ratio_of_means shape mismatch")
    w = w / w.sum()
    ec = float(np.dot(w, C))
    eb = float(np.dot(w, B))
    denom = ec + eb
    _require(denom > 0.0, "ratio_of_means denominator is zero")
    return ec / denom


def robust_scale(descriptors: "np.ndarray") -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Robust scale by Training pool median/IQR; columns with zero IQR are left zero."""
    desc = np.asarray(descriptors, dtype=np.float64)
    q = np.percentile(desc, [25.0, 50.0, 75.0], axis=0)
    median = q[1]
    iqr = q[2] - q[0]
    active = iqr > 0.0
    scaled = np.zeros_like(desc)
    scaled[:, active] = (desc[:, active] - median[active]) / iqr[active]
    return scaled, median, iqr


# ---------------------------------------------------------------------------
# Leave-cluster-out exact KNN  (read by tests)
# ---------------------------------------------------------------------------

def leave_cluster_out_knn(
    descriptors: "np.ndarray",   # (N, D) already scaled — physical descriptor space
    cluster_ids: "np.ndarray",   # (N,)  int — twin/query cluster
    *,
    max_k: int,
    chunk_size: int = NEIGHBOR_CHUNK_SIZE,
) -> tuple["np.ndarray", "np.ndarray"]:
    """Return (neighbors, distances) excluding every member of the query's cluster."""
    desc = np.asarray(descriptors, dtype=np.float64)
    cids = np.asarray(cluster_ids, dtype=np.int64)
    N = desc.shape[0]
    _require(desc.ndim == 2, "descriptors must be rank 2")
    _require(cids.shape == (N,), "cluster_ids length mismatch")
    _require(max_k > 0, "max_k must be positive")
    _finite("knn descriptors", desc)

    members: dict[int, "np.ndarray"] = {
        int(c): np.flatnonzero(cids == c) for c in np.unique(cids)
    }
    _require(
        max_k <= N - max(len(v) for v in members.values()),
        "max_k too large given cluster sizes",
    )

    neighbors = np.empty((N, max_k), dtype=np.int64)
    distances = np.empty((N, max_k), dtype=np.float64)

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)
        q = desc[start:stop]                           # (Q, D)
        diff = q[:, None, :] - desc[None, :, :]       # (Q, N, D)
        dsq = np.einsum("qnd,qnd->qn", diff, diff, optimize=False)  # (Q, N)
        for local, row in enumerate(range(start, stop)):
            dsq[local, members[int(cids[row])]] = np.inf
            part = np.argpartition(dsq[local], max_k - 1)[:max_k]
            cutoff = float(np.max(dsq[local, part]))
            strict = np.flatnonzero(dsq[local] < cutoff)
            tied = np.flatnonzero(dsq[local] == cutoff)
            tied = tied[np.argsort(tied, kind="stable")]
            needed = max_k - len(strict)
            selected = np.concatenate([strict, tied[:needed]])
            order = np.lexsort((selected, dsq[local, selected]))
            selected = selected[order]
            _require(len(selected) == max_k, "neighbor count wrong")
            neighbors[row] = selected
            distances[row] = np.sqrt(np.maximum(0.0, dsq[local, selected]))

    _finite("knn distances", distances)
    return neighbors, distances


def background_variance_by_k(
    mean_displacements: "np.ndarray",   # (N, P) per-query weighted mean pixel displacement
    neighbors: "np.ndarray",            # (N, max_k) neighbor indices
    ks: Iterable[int],
) -> dict[int, "np.ndarray"]:
    """Exact B_loc,k via a pixel-normalized Gram matrix, without (N,k,P)."""
    means = np.asarray(mean_displacements, dtype=np.float32)
    _finite("mean displacements", means)
    _require(means.ndim == 2 and means.shape[1] > 0, "mean displacements must be (N,P)")
    pixel_dim = means.shape[1]
    gram = (means @ means.T).astype(np.float64) / float(pixel_dim)
    norms = np.diag(gram)
    result: dict[int, "np.ndarray"] = {}
    for k in ks:
        k = int(k)
        _require(0 < k <= neighbors.shape[1], f"invalid k={k}")
        variance = np.empty(len(means), dtype=np.float64)
        for row, selected in enumerate(neighbors[:, :k]):
            first_moment = float(np.mean(norms[selected]))
            center_norm = float(np.mean(gram[np.ix_(selected, selected)]))
            variance[row] = max(0.0, first_moment - center_norm)
        _finite(f"B_{k}", variance)
        _require(np.all(variance >= 0.0), f"B_{k} has negative values")
        result[k] = variance
    return result


# ---------------------------------------------------------------------------
# Deterministic balanced sampling  (read by tests)
# ---------------------------------------------------------------------------

def deterministic_balanced_sample(
    group_ids: "np.ndarray",
    total: int,
    seed: int,
) -> "np.ndarray":
    """Return `total` indices, equally distributed across group_ids, fixed seed."""
    group_ids = np.asarray(group_ids, dtype=np.int64)
    unique = np.unique(group_ids)
    n_groups = len(unique)
    _require(total % n_groups == 0, "total must be divisible by number of groups")
    per_group = total // n_groups
    rng = np.random.default_rng(seed)
    selected: list["np.ndarray"] = []
    for g in unique:
        indices = np.flatnonzero(group_ids == g)
        _require(len(indices) >= per_group, f"group {g} has fewer than {per_group} members")
        chosen = rng.choice(indices, size=per_group, replace=False)
        selected.append(np.sort(chosen))
    return np.concatenate(selected)


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ratio_of_means_ci(
    C: "np.ndarray",
    B: "np.ndarray",
    cluster_ids: "np.ndarray",
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    ci_level: float = 0.95,
) -> dict[str, float]:
    """Resample whole clusters, compute ratio_of_means on each resample, return CI."""
    C = np.asarray(C, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    unique_clusters = np.unique(cluster_ids)
    n_clusters = len(unique_clusters)
    rng = np.random.default_rng(seed)

    members = {int(c): np.flatnonzero(cluster_ids == c) for c in unique_clusters}

    rhos: list[float] = []
    for _ in range(n_resamples):
        chosen = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([members[int(c)] for c in chosen])
        w = np.full(len(idx), 1.0 / len(idx))
        rhos.append(ratio_of_means(C[idx], B[idx], w))

    rhos_arr = np.sort(rhos)
    alpha = 1.0 - ci_level
    lo_idx = max(0, int(math.floor(alpha / 2 * n_resamples)))
    hi_idx = min(n_resamples - 1, int(math.ceil((1 - alpha / 2) * n_resamples)))
    return {
        "mean": float(np.mean(rhos_arr)),
        "ci_lo": float(rhos_arr[lo_idx]),
        "ci_hi": float(rhos_arr[hi_idx]),
        "ci_level": ci_level,
        "n_resamples": n_resamples,
    }


def _distribution_summary(values: "np.ndarray") -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).ravel()
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "q00": float(np.min(v)),
        "q25": float(np.percentile(v, 25)),
        "q50": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "q100": float(np.max(v)),
        "n": int(len(v)),
    }


# ---------------------------------------------------------------------------
# Config loader and path resolver
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        raise ImportError("PyYAML is required: pip install pyyaml")


def _resolve(base: Path, s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


# ---------------------------------------------------------------------------
# Lance column helpers (shared with v1)
# ---------------------------------------------------------------------------

def _fixed_list_col(table: Any, name: str, width: int) -> "np.ndarray":
    col = table[name].combine_chunks()
    vals = np.asarray(col.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
    return vals.reshape(len(col), width)


def _scalar_col(table: Any, name: str, dtype: Any) -> "np.ndarray":
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def decode_pixel_blob(value: bytes, image_shape: Iterable[int]) -> "np.ndarray":
    """Decode one lossless PNG payload to flattened RGB float32 in [0,1]."""
    from PIL import Image

    expected = tuple(int(v) for v in image_shape)
    with Image.open(io.BytesIO(bytes(value))) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    _require(rgb.shape == expected, f"pixel shape {rgb.shape} != {expected}")
    return rgb.reshape(-1).astype(np.float32) / np.float32(255.0)


def _pixel_blobs(table: Any, name: str) -> list[bytes]:
    return [bytes(value) for value in table[name].to_pylist()]


def pixel_history_conditional_variance(
    histories: "np.ndarray",  # (C,T,P), normalized RGB
    w: "np.ndarray",          # (C,)
) -> float:
    """Average cross-condition pixel variance over visible history frames."""
    values = np.asarray(histories, dtype=np.float64)
    weights = np.asarray(w, dtype=np.float64)
    _require(values.ndim == 3, "histories must be (C,T,P)")
    _require(weights.shape == (values.shape[0],), "history weights shape mismatch")
    _require(math.isclose(float(weights.sum()), 1.0, abs_tol=1e-9), "w must sum to 1")
    center = np.einsum("c,ctp->tp", weights, values)
    residual = values - center[None, :, :]
    return float(
        np.einsum("c,ctp->", weights, np.square(residual))
        / (values.shape[1] * values.shape[2])
    )


# ---------------------------------------------------------------------------
# Motion pixel auditor
# ---------------------------------------------------------------------------

def audit_motion_raw_pixel(
    cfg: dict[str, Any],
    repo_root: Path,
    check_only: bool,
    sample_per_task: int,
    seed: int,
) -> dict[str, Any]:
    import lance  # type: ignore
    import re as _re, math as _math

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance)
    training_only_guard(manifest_path)
    _require(train_lance.exists(), f"Motion train.lance missing: {train_lance}")
    _require(manifest_path.exists(), f"Motion manifest missing: {manifest_path}")

    pair_count = cfg.get("expected_pair_count", 8192)
    ep_steps = cfg.get("episode_steps", 20)
    query_step = cfg.get("query_step", 10)
    future_step = cfg.get("future_step", 15)
    hist_steps = cfg.get("history_steps", [0, 5])
    tslice = cfg.get("target_slice", [6, 8])
    pixel_column = cfg.get("pixel_column", "obs")
    image_shape = cfg.get("image_shape", [96, 96, 3])
    pixel_scale = float(cfg.get("pixel_scale", 255.0))
    pixel_dim = int(image_shape[0]) * int(image_shape[1]) * int(image_shape[2])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_m = manifest.get("splits", {}).get("train", {})
    manifest_pairs = train_m.get("pairs", [])
    _require(len(manifest_pairs) == pair_count,
             f"Motion manifest pair count {len(manifest_pairs)} != {pair_count}")

    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "pair_count": pair_count,
        "pixel_dim": pixel_dim,
        "image_shape": image_shape,
        "pixel_scale": pixel_scale,
        "check_only": check_only,
    }

    if check_only:
        return {"task": "motion", "status": "check_ok", "receipt": receipt}

    columns = ["episode_idx", "step_idx", "action", "physics_state",
               "goal_state", "pair_id", "hidden_mode", "split", pixel_column]
    table = lance.dataset(str(train_lance)).to_table(columns=columns)
    expected_rows = 2 * pair_count * ep_steps
    _require(table.num_rows == expected_rows,
             f"Motion row count {table.num_rows} != {expected_rows}")

    ep_idx = _scalar_col(table, "episode_idx", np.int64)
    st_idx = _scalar_col(table, "step_idx", np.int64)
    actions = _fixed_list_col(table, "action", 2)
    physics = _fixed_list_col(table, "physics_state", 12)
    goals = _fixed_list_col(table, "goal_state", 7)
    pixels = _pixel_col(table, pixel_column, pixel_dim)
    pair_ids_raw = np.asarray(table["pair_id"].to_pylist(), dtype=object)
    modes_raw = np.asarray(table["hidden_mode"].to_pylist(), dtype=object)

    order = np.lexsort((st_idx, ep_idx))
    ep_idx = ep_idx[order]; st_idx = st_idx[order]
    actions = actions[order]; physics = physics[order]
    goals = goals[order]; pixels = pixels[order]
    pair_ids_raw = pair_ids_raw[order]; modes_raw = modes_raw[order]

    n_eps = 2 * pair_count
    actions_e = actions.reshape(n_eps, ep_steps, 2)
    physics_e = physics.reshape(n_eps, ep_steps, 12)
    goals_e = goals.reshape(n_eps, ep_steps, 7)
    pixels_e = pixels.reshape(n_eps, ep_steps, pixel_dim)
    pair_ids_e = pair_ids_raw.reshape(n_eps, ep_steps)
    modes_e = modes_raw.reshape(n_eps, ep_steps)

    MODES = ("faster_decay", "no_extra_decay")
    episodes: dict[int, dict[str, int]] = {}
    for ep in range(n_eps):
        pid = str(pair_ids_e[ep, 0])
        m = _re.fullmatch(r"pmd-train-(\d{6})-(forward|reverse)", pid)
        _require(m is not None, f"unexpected pair_id: {pid}")
        pidx = int(m.group(1))
        mode = str(modes_e[ep, 0])
        _require(mode in MODES, f"unexpected mode: {mode}")
        by_mode = episodes.setdefault(pidx, {})
        _require(mode not in by_mode, f"duplicate pair/mode: {pidx}/{mode}")
        by_mode[mode] = ep

    # Build per-pair arrays for balanced sampling
    all_pair_indices = sorted(episodes.keys())
    _require(len(all_pair_indices) == pair_count, "pair count mismatch after grouping")
    group_ids = np.arange(pair_count, dtype=np.int64)
    selected = deterministic_balanced_sample(group_ids, sample_per_task, seed)

    C_pix_list: list[float] = []
    m_pix_list: list[np.ndarray] = []
    C_phys_list: list[float] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []

    w2 = np.array([0.5, 0.5], dtype=np.float64)
    for flat_idx in selected:
        pidx = all_pair_indices[int(flat_idx)]
        pe = episodes[pidx]
        e0, e1 = pe[MODES[0]], pe[MODES[1]]
        _require(np.allclose(actions_e[e0], actions_e[e1], atol=1e-6),
                 f"actions differ pair {pidx}")
        _require(np.allclose(physics_e[e0, query_step], physics_e[e1, query_step], atol=1e-6),
                 f"query physics differ pair {pidx}")

        x_pix = normalize_pixels(pixels_e[e0, query_step], pixel_scale)
        y0_pix = normalize_pixels(pixels_e[e0, future_step], pixel_scale)
        y1_pix = normalize_pixels(pixels_e[e1, future_step], pixel_scale)
        Y_pix = np.stack([y0_pix, y1_pix])[None]   # (1, 2, P)
        C_pix = float(pixel_conditional_variance(Y_pix, x_pix[None], w2)[0])
        m_pix = pixel_mean_displacement(Y_pix, x_pix[None], w2)[0]  # (P,)

        x_phys = physics_e[e0, query_step, tslice[0]:tslice[1]]
        y0_phys = physics_e[e0, future_step, tslice[0]:tslice[1]]
        y1_phys = physics_e[e1, future_step, tslice[0]:tslice[1]]
        Y_phys = np.stack([y0_phys, y1_phys])[None]
        diff_phys = Y_phys - x_phys[None, None, :]         # (1,2,D)
        m_phys = 0.5 * (diff_phys[0, 0] + diff_phys[0, 1])
        res_phys = diff_phys[0] - m_phys[None, :]
        C_phys = float(0.5 * np.sum(np.square(res_phys[0])) + 0.5 * np.sum(np.square(res_phys[1])))

        goal = goals_e[e0, query_step]
        goal_rel = goal[2:4] - x_phys
        theta = float(physics_e[e0, query_step, 10])
        desc = np.array([
            x_phys[0], x_phys[1],
            physics_e[e0, query_step, 8], physics_e[e0, query_step, 9],
            _math.sin(theta), _math.cos(theta),
            goal_rel[0], goal_rel[1],
        ])
        C_pix_list.append(C_pix)
        m_pix_list.append(m_pix)
        C_phys_list.append(C_phys)
        desc_list.append(desc)
        cids.append(pidx // 2)

    C_pix_arr = np.array(C_pix_list)
    m_pix_arr = np.stack(m_pix_list)
    C_phys_arr = np.array(C_phys_list)
    desc_arr = np.stack(desc_list)
    cid_arr = np.array(cids, dtype=np.int64)

    desc_scaled, _, _ = robust_scale(desc_arr)
    neighbors, _ = leave_cluster_out_knn(desc_scaled, cid_arr,
                                         max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk = background_variance_by_k(m_pix_arr, neighbors, KS)
    wq = np.full(len(C_pix_arr), 1.0 / len(C_pix_arr))
    rhos = {k: ratio_of_means(C_pix_arr, bvk[k], wq) for k in KS}
    ci = bootstrap_ratio_of_means_ci(C_pix_arr, bvk[MAIN_K], cid_arr,
                                     n_resamples=BOOTSTRAP_RESAMPLES, seed=seed)
    receipt["train_table_sha256_from_manifest"] = str(train_m.get("table_sha256", ""))
    return {
        "task": "motion",
        "status": "ok",
        "n_queries": len(C_pix_arr),
        "C_pixel": _distribution_summary(C_pix_arr),
        "C_physical_reference": _distribution_summary(C_phys_arr),
        "coordinate_definition": "per-pixel-channel MSE in [0,1]^P; KNN in PushT block-position space",
        "B": {str(k): _distribution_summary(bvk[k]) for k in KS},
        "rho": {str(k): float(rhos[k]) for k in KS},
        "rho_bootstrap_ci": ci,
        "evidence_type": "observed_matched_twin_binary",
        "condition_balance": {"n_conditions": 2, "weights": [0.5, 0.5]},
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# ActionDelay pixel auditor
# ---------------------------------------------------------------------------

def _delay_weights() -> tuple[list[int], list[float]]:
    """6 equal native-exposure groups: [0],[1],[2],[3],[4],[5..10]. d0..d4 each 1/6, d5..d10 each 1/36."""
    delays = list(range(11))
    raw: list[float] = [1.0 / 6.0 if d <= 4 else 1.0 / 36.0 for d in delays]
    total = sum(raw)
    return delays, [w / total for w in raw]


def audit_action_delay_raw_pixel(
    cfg: dict[str, Any],
    repo_root: Path,
    check_only: bool,
    sample_per_task: int,
    seed: int,
) -> dict[str, Any]:
    import lance  # type: ignore

    table_dir = Path(cfg["train_table_dir"]).resolve()
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(table_dir)
    training_only_guard(manifest_path)
    _require(table_dir.exists(), f"ActionDelay train table dir missing: {table_dir}")
    _require(manifest_path.exists(), f"ActionDelay manifest missing: {manifest_path}")

    ep_steps = cfg.get("episode_steps", 50)
    query_step = cfg.get("query_step", 30)
    future_step = cfg.get("future_step", 35)
    hist_steps = cfg.get("history_steps", [0, 5, 10, 15, 20, 25])
    tslice = cfg.get("target_slice", [0, 2])
    pixel_column = cfg.get("pixel_column", "obs")
    image_shape = cfg.get("image_shape", [224, 224, 3])
    pixel_scale = float(cfg.get("pixel_scale", 255.0))
    pixel_dim = int(image_shape[0]) * int(image_shape[1]) * int(image_shape[2])
    delay_values, delay_weights_list = _delay_weights()

    manifest_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                manifest_rows.append(json.loads(line))

    train_manifest_rows = [
        row for row in manifest_rows if str(row.get("split", "")).lower() == "train"
    ]
    by_pair: dict[str, dict[int, dict[str, Any]]] = {}
    for row in train_manifest_rows:
        pair_id = str(row["pair_id"])
        delay = int(row["factors"]["action.delay_steps"])
        _require(delay in delay_values, f"unexpected ActionDelay value {delay}")
        delay_rows = by_pair.setdefault(pair_id, {})
        _require(delay not in delay_rows, f"duplicate ActionDelay row {pair_id}/d{delay}")
        delay_rows[delay] = row
    _require(len(by_pair) == 32, f"expected 32 ActionDelay pair shards, found {len(by_pair)}")
    for pair_id, delay_rows in by_pair.items():
        _require(set(delay_rows) == set(delay_values), f"incomplete delays for {pair_id}")

    def _row_path(row: dict[str, Any]) -> Path:
        path = table_dir / Path(str(row["output_path"])).name
        training_only_guard(path)
        _require(path.exists(), f"missing ActionDelay table: {path}")
        return path

    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "manifest_rows": len(manifest_rows),
        "train_manifest_rows": len(train_manifest_rows),
        "train_storage_digest": manifest_storage_digest(train_manifest_rows),
        "pair_shards": len(by_pair),
        "pixel_dim": pixel_dim,
        "image_shape": image_shape,
        "pixel_scale": pixel_scale,
        "check_only": check_only,
    }

    if check_only:
        for delay_rows in by_pair.values():
            for row in delay_rows.values():
                _row_path(row)
        return {"task": "action_delay", "status": "check_ok", "receipt": receipt}

    selected_steps = sorted(set(hist_steps + list(range(query_step, future_step + 1))))

    def _load_selected(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        dataset = lance.dataset(str(path))
        names = set(dataset.schema.names)
        _require({"episode_idx", "step_idx", "proprio", "action", pixel_column} <= names,
                 f"ActionDelay schema mismatch: {path}")
        predicate = "step_idx IN (" + ",".join(str(v) for v in selected_steps) + ")"
        table = dataset.to_table(
            columns=["episode_idx", "step_idx", "proprio", "action", pixel_column],
            filter=predicate,
        )
        episode = _scalar_col(table, "episode_idx", np.int64)
        step = _scalar_col(table, "step_idx", np.int64)
        position = _fixed_list_col(table, "proprio", 2)
        action = _fixed_list_col(table, "action", 2)
        pixels = _pixel_col(table, pixel_column, pixel_dim)
        order = np.lexsort((step, episode))
        episode, step = episode[order], step[order]
        position, action, pixels = position[order], action[order], pixels[order]
        unique_episodes = np.unique(episode)
        expected_episode = np.arange(len(unique_episodes), dtype=np.int64)
        _require(np.array_equal(unique_episodes, expected_episode),
                 f"non-contiguous ActionDelay episodes in {path.name}")
        expected_step = np.tile(np.asarray(selected_steps), len(unique_episodes))
        _require(np.array_equal(step, expected_step),
                 f"missing selected ActionDelay steps in {path.name}")
        width = len(selected_steps)
        return (
            position.reshape(len(unique_episodes), width, 2),
            action.reshape(len(unique_episodes), width, 2),
            pixels.reshape(len(unique_episodes), width, pixel_dim),
            unique_episodes,
        )

    w_arr = np.asarray(delay_weights_list, dtype=np.float64)
    step_to_col = {step: index for index, step in enumerate(selected_steps)}
    history_cols = [step_to_col[step] for step in hist_steps]
    query_col = step_to_col[query_step]
    future_col = step_to_col[future_step]
    action_cols = [step_to_col[step] for step in range(query_step, future_step)]

    # Load all pairs to build candidate list for balanced sampling
    sorted_pairs = sorted(by_pair.items())
    # Each pair yields n_episodes queries; collect them all then sample
    pair_episode_counts: list[int] = []
    all_candidate_data: list[dict[str, Any]] = []

    action_match_max = 0.0
    for pair_index, (pid, delay_rows) in enumerate(sorted_pairs):
        positions_by_delay: list[np.ndarray] = []
        actions_by_delay: list[np.ndarray] = []
        pixels_by_delay: list[np.ndarray] = []
        episode_ids: np.ndarray | None = None
        for delay in delay_values:
            position, action, pix, current_episode_ids = _load_selected(_row_path(delay_rows[delay]))
            if episode_ids is None:
                episode_ids = current_episode_ids
            else:
                _require(np.array_equal(episode_ids, current_episode_ids),
                         f"episode identity differs in {pid}/d{delay}")
            positions_by_delay.append(position)
            actions_by_delay.append(action)
            pixels_by_delay.append(pix)
        assert episode_ids is not None
        positions = np.stack(positions_by_delay, axis=1)   # (E, C, S, 2)
        actions = np.stack(actions_by_delay, axis=1)       # (E, C, S, 2)
        pix_arr = np.stack(pixels_by_delay, axis=1)        # (E, C, S, P)
        ref_actions = actions[:, 0]
        action_resid = float(np.max(np.abs(actions - ref_actions[:, None])))
        action_match_max = max(action_match_max, action_resid)
        _require(action_resid <= 1.0e-6, f"actions differ across delays in {pid}")
        query_positions = positions[:, :, query_col, tslice[0]:tslice[1]]
        query_resid = float(np.max(np.abs(query_positions - query_positions[:, :1])))
        _require(query_resid <= 1.0e-6, f"query differs across delays in {pid}")

        n_eps = len(episode_ids)
        pair_episode_counts.append(n_eps)
        for ep_offset in range(n_eps):
            all_candidate_data.append({
                "pair_index": pair_index,
                "ep_offset": ep_offset,
                "positions": positions[ep_offset],   # (C, S, 2)
                "actions": ref_actions[ep_offset],   # (S, 2)
                "pixels": pix_arr[ep_offset],        # (C, S, P)
            })

    # Balanced sample: group by pair_index, draw sample_per_task total
    total_candidates = len(all_candidate_data)
    cand_groups = np.array([d["pair_index"] for d in all_candidate_data], dtype=np.int64)
    n_groups = len(sorted_pairs)
    # If sample_per_task not divisible by n_groups, floor
    actual_sample = (sample_per_task // n_groups) * n_groups
    if actual_sample == 0:
        actual_sample = n_groups
    selected_indices = deterministic_balanced_sample(cand_groups, actual_sample, seed)

    C_pix_list: list[float] = []
    m_pix_list: list[np.ndarray] = []
    C_phys_list: list[float] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []

    for flat_idx in selected_indices:
        cand = all_candidate_data[int(flat_idx)]
        pair_index = cand["pair_index"]
        positions = cand["positions"]   # (C, S, 2)
        ref_acts = cand["actions"]      # (S, 2)
        pix_cond = cand["pixels"]       # (C, S, P)

        x_pix = normalize_pixels(pix_cond[0, query_col], pixel_scale)   # query from delay=0
        Y_pix = np.stack([normalize_pixels(pix_cond[c, future_col], pixel_scale)
                          for c in range(len(delay_values))])  # (C, P)
        Y_pix_n = Y_pix[None]  # (1, C, P)
        C_pix = float(pixel_conditional_variance(Y_pix_n, x_pix[None], w_arr)[0])
        m_pix = pixel_mean_displacement(Y_pix_n, x_pix[None], w_arr)[0]

        x_phys = positions[0, query_col, tslice[0]:tslice[1]]
        Y_phys = positions[:, future_col, tslice[0]:tslice[1]]   # (C, 2)
        diffs_phys = Y_phys - x_phys[None, :]
        m_phys = np.einsum("c,cd->d", w_arr, diffs_phys)
        res_phys = diffs_phys - m_phys[None, :]
        C_phys = float(np.einsum("c,c->", w_arr,
                                 np.sum(np.square(res_phys), axis=1)))

        q_action = ref_acts[action_cols].ravel()
        desc = np.concatenate([x_phys, q_action])
        C_pix_list.append(C_pix)
        m_pix_list.append(m_pix)
        C_phys_list.append(C_phys)
        desc_list.append(desc)
        cids.append(pair_index * 1000 + cand["ep_offset"])

    C_pix_arr = np.array(C_pix_list)
    m_pix_arr = np.stack(m_pix_list)
    C_phys_arr = np.array(C_phys_list)
    desc_arr = np.stack(desc_list)
    cid_arr = np.array(cids, dtype=np.int64)

    _require(len(C_pix_arr) > max(KS) + 10, f"too few ActionDelay queries: {len(C_pix_arr)}")
    desc_scaled, _, _ = robust_scale(desc_arr)
    neighbors, _ = leave_cluster_out_knn(desc_scaled, cid_arr,
                                         max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk = background_variance_by_k(m_pix_arr, neighbors, KS)
    wq = np.full(len(C_pix_arr), 1.0 / len(C_pix_arr))
    rhos = {k: ratio_of_means(C_pix_arr, bvk[k], wq) for k in KS}
    ci = bootstrap_ratio_of_means_ci(C_pix_arr, bvk[MAIN_K], cid_arr,
                                     n_resamples=BOOTSTRAP_RESAMPLES, seed=seed)
    receipt["action_match_residual_max"] = float(action_match_max)
    return {
        "task": "action_delay",
        "status": "ok",
        "n_queries": len(C_pix_arr),
        "C_pixel": _distribution_summary(C_pix_arr),
        "C_physical_reference": _distribution_summary(C_phys_arr),
        "coordinate_definition": "per-pixel-channel MSE in [0,1]^P; KNN in TwoRoom agent-position space",
        "B": {str(k): _distribution_summary(bvk[k]) for k in KS},
        "rho": {str(k): float(rhos[k]) for k in KS},
        "rho_bootstrap_ci": ci,
        "evidence_type": "observed_matched_condition_group",
        "condition_balance": {
            "n_conditions": len(delay_values),
            "weights": {str(d): float(w) for d, w in zip(delay_values, delay_weights_list)},
        },
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Speed pixel — pending (no simulator pixel renderer, no observed twins)
# ---------------------------------------------------------------------------

def audit_speed_raw_pixel_pending(
    cfg: dict[str, Any],
    repo_root: Path,
    check_only: bool,
) -> dict[str, Any]:
    """Speed has no same-query cross-speed observed pixel twins and no simulator
    pixel renderer.  Return a structured pending record rather than partial data."""
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(manifest_path)
    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path) if manifest_path.exists() else "missing",
        "pixel_status": SPEED_PIXEL_STATUS,
        "check_only": check_only,
    }
    return {
        "task": "speed",
        "status": "pending",
        "pixel_status": SPEED_PIXEL_STATUS,
        "reason": (
            "Speed Training has no same-query cross-speed observed pixel twins and "
            "no simulator pixel renderer is available; pixel C/B/rho cannot be "
            "computed from observed data alone."
        ),
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Training-only raw-pixel conditional visibility CPU audit (v1)"
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to icl_training_raw_pixel_visibility_v1.yaml")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory (must not exist)")
    parser.add_argument("--check-only", action="store_true",
                        help="Schema/manifest/path check only; skip KNN/bootstrap")
    parser.add_argument("--sample-per-task", type=int, default=None,
                        help="Override per-task sample count (default from config or 256)")
    parser.add_argument("--skip-speed", action="store_true",
                        help="Skip the simulator-rendered Speed counterfactual audit")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    training_only_guard(config_path)
    _require(config_path.exists(), f"Config not found: {config_path}")

    cfg = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]

    output_dir = Path(args.output_dir).resolve()
    exclusive_mkdir(output_dir)

    sample_cfg = cfg.get("sample", {})
    effective_sample = (
        args.sample_per_task
        if args.sample_per_task is not None
        else int(sample_cfg.get("per_task", SAMPLE_PER_TASK))
    )
    sample_seed = int(sample_cfg.get("seed", BOOTSTRAP_SEED))

    audit_id = str(uuid.uuid4())
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    TASK_AUDITORS = {
        "motion": audit_motion_raw_pixel,
        "action_strength": audit_action_strength_raw_pixel,
        "action_delay": audit_action_delay_raw_pixel,
    }

    per_task_results: list[dict[str, Any]] = []
    errors: list[str] = []

    tasks_cfg = cfg.get("tasks", {})
    for task_name, auditor_fn in TASK_AUDITORS.items():
        task_cfg = tasks_cfg.get(task_name, {})
        if not task_cfg.get("enabled", True):
            per_task_results.append({"task": task_name, "status": "disabled"})
            continue
        try:
            result = auditor_fn(task_cfg, repo_root, args.check_only,
                                effective_sample, sample_seed)
            per_task_results.append(result)
        except Exception as exc:
            per_task_results.append({"task": task_name, "status": "error", "error": str(exc)})
            errors.append(f"{task_name}: {exc}")

    # Speed uses frozen simulator-rendered counterfactuals on Training queries.
    speed_cfg = tasks_cfg.get("speed", {})
    if not args.skip_speed and speed_cfg.get("enabled", True):
        try:
            result = audit_speed_raw_pixel(
                speed_cfg, repo_root, args.check_only, effective_sample, sample_seed
            )
            per_task_results.append(result)
        except Exception as exc:
            per_task_results.append({"task": "speed", "status": "error", "error": str(exc)})
            errors.append(f"speed: {exc}")

    if args.check_only:
        overall_status = "check_ok" if not errors else "check_error"
    else:
        # pending is not an error
        non_error = all(
            r.get("status") in ("ok", "check_ok", "pending", "disabled")
            for r in per_task_results
        )
        overall_status = "ok" if (not errors and non_error) else "partial_error"

    summary = {
        "schema_version": 1,
        "audit_id": audit_id,
        "started_at": started_at,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": overall_status,
        "check_only": args.check_only,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "sample_per_task": effective_sample,
        "errors": errors,
        "task_statuses": {r["task"]: r["status"] for r in per_task_results},
        "evidence_boundary": {
            "pixels_decoded": not args.check_only,
            "model_loaded": MODEL_LOADED,
            "optimizer_steps": OPTIMIZER_STEPS,
            "pixel_units": PIXEL_UNITS,
            "cross_task_absolute_value_comparability": (
                "descriptive_only; tasks differ in renderer, condition cardinality, "
                "evidence type, and local-neighborhood geometry"
            ),
            "pixel_target_separation_status": PIXEL_TARGET_SEPARATION_STATUS,
            "latent_status": LATENT_STATUS,
            "binding_cause_status": BINDING_CAUSE_STATUS,
            "claim_scope": CLAIM_SCOPE,
            "speed_pixel_status": SPEED_PIXEL_STATUS,
        },
    }

    _json_dump(output_dir / "summary.json", summary)

    with (output_dir / "per_task.jsonl").open("x", encoding="utf-8") as fh:
        for r in per_task_results:
            fh.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")

    receipt_obj = {
        "schema_version": 1,
        "audit_id": audit_id,
        "status": overall_status,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "config_sha256": file_sha256(config_path),
        "script_sha256": file_sha256(Path(__file__).resolve()),
        "optimizer_steps": OPTIMIZER_STEPS,
        "pixels_decoded": not args.check_only,
        "model_loaded": MODEL_LOADED,
    }
    _json_dump(output_dir / "receipt.json", receipt_obj)

    print(json.dumps({
        "status": overall_status,
        "audit_id": audit_id,
        "output_dir": str(output_dir),
        "task_statuses": summary["task_statuses"],
        "errors": errors,
    }, indent=2))

    if errors:
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# ActionStrength pixel auditor
# ---------------------------------------------------------------------------

def audit_action_strength_raw_pixel(
    cfg: dict[str, Any],
    repo_root: Path,
    check_only: bool,
    sample_per_task: int,
    seed: int,
) -> dict[str, Any]:
    import lance  # type: ignore
    import math as _math
    from collections import defaultdict as _dd

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance)
    training_only_guard(manifest_path)
    _require(train_lance.exists(), f"ActionStrength train.lance missing: {train_lance}")
    _require(manifest_path.exists(), f"ActionStrength manifest missing: {manifest_path}")

    pair_count = cfg.get("expected_pair_count", 2048)
    ep_steps = cfg.get("episode_steps", 20)
    query_step = cfg.get("query_step", 10)
    future_step = cfg.get("future_step", 15)
    tslice = cfg.get("target_slice", [6, 8])
    pixel_column = cfg.get("pixel_column", "obs")
    image_shape = cfg.get("image_shape", [96, 96, 3])
    pixel_scale = float(cfg.get("pixel_scale", 255.0))
    pixel_dim = int(image_shape[0]) * int(image_shape[1]) * int(image_shape[2])

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_m = manifest.get("splits", {}).get("train", {})
    manifest_pairs = train_m.get("pairs", [])
    _require(len(manifest_pairs) == pair_count,
             f"ActionStrength manifest pair count {len(manifest_pairs)} != {pair_count}")

    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "pair_count": pair_count,
        "pixel_dim": pixel_dim,
        "image_shape": image_shape,
        "pixel_scale": pixel_scale,
        "check_only": check_only,
    }
    if check_only:
        return {"task": "action_strength", "status": "check_ok", "receipt": receipt}

    columns = ["episode_idx", "step_idx", "action", "physics_state",
               "goal_state", "pair_id", "hidden_mode", "split", pixel_column]
    table = lance.dataset(str(train_lance)).to_table(columns=columns)
    expected_rows = 2 * pair_count * ep_steps
    _require(table.num_rows == expected_rows,
             f"ActionStrength row count {table.num_rows} != {expected_rows}")

    ep_idx = _scalar_col(table, "episode_idx", np.int64)
    st_idx = _scalar_col(table, "step_idx", np.int64)
    actions = _fixed_list_col(table, "action", 2)
    physics = _fixed_list_col(table, "physics_state", 12)
    goals = _fixed_list_col(table, "goal_state", 7)
    pixels = _pixel_col(table, pixel_column, pixel_dim)
    pair_ids_raw = np.asarray(table["pair_id"].to_pylist(), dtype=object)
    modes_raw = np.asarray(table["hidden_mode"].to_pylist(), dtype=object)

    order = np.lexsort((st_idx, ep_idx))
    ep_idx = ep_idx[order]; st_idx = st_idx[order]
    actions = actions[order]; physics = physics[order]
    goals = goals[order]; pixels = pixels[order]
    pair_ids_raw = pair_ids_raw[order]; modes_raw = modes_raw[order]

    n_eps = 2 * pair_count
    actions_e = actions.reshape(n_eps, ep_steps, 2)
    physics_e = physics.reshape(n_eps, ep_steps, 12)
    goals_e = goals.reshape(n_eps, ep_steps, 7)
    pixels_e = pixels.reshape(n_eps, ep_steps, pixel_dim)
    pair_ids_e = pair_ids_raw.reshape(n_eps, ep_steps)
    modes_e = modes_raw.reshape(n_eps, ep_steps)

    episodes: dict[str, list[int]] = _dd(list)
    for ep in range(n_eps):
        episodes[str(pair_ids_e[ep, 0])].append(ep)

    sorted_pairs = sorted(episodes.items())
    group_ids = np.arange(len(sorted_pairs), dtype=np.int64)
    selected = deterministic_balanced_sample(group_ids, sample_per_task, seed)

    C_pix_list: list[float] = []
    m_pix_list: list[np.ndarray] = []
    C_phys_list: list[float] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []
    w2 = np.array([0.5, 0.5], dtype=np.float64)

    for flat_idx in selected:
        pidx = int(flat_idx)
        pid, eps = sorted_pairs[pidx]
        _require(len(eps) == 2, f"ActionStrength pair {pid} does not have two episodes")
        by_mode = {str(modes_e[ep, 0]): ep for ep in eps}
        _require(set(by_mode) == {"low_gain", "high_gain"},
                 f"ActionStrength pair {pid} has unexpected modes {sorted(by_mode)}")
        e0, e1 = by_mode["low_gain"], by_mode["high_gain"]
        _require(np.allclose(physics_e[e0, query_step], physics_e[e1, query_step], atol=1e-5),
                 f"ActionStrength query physics differ at pair {pid}")

        x_pix = normalize_pixels(pixels_e[e0, query_step], pixel_scale)
        y0_pix = normalize_pixels(pixels_e[e0, future_step], pixel_scale)
        y1_pix = normalize_pixels(pixels_e[e1, future_step], pixel_scale)
        Y_pix = np.stack([y0_pix, y1_pix])[None]
        C_pix = float(pixel_conditional_variance(Y_pix, x_pix[None], w2)[0])
        m_pix = pixel_mean_displacement(Y_pix, x_pix[None], w2)[0]

        x_phys = physics_e[e0, query_step, tslice[0]:tslice[1]]
        y0_phys = physics_e[e0, future_step, tslice[0]:tslice[1]]
        y1_phys = physics_e[e1, future_step, tslice[0]:tslice[1]]
        diff0 = y0_phys - x_phys; diff1 = y1_phys - x_phys
        m_phys = 0.5 * (diff0 + diff1)
        C_phys = float(0.5 * np.sum(np.square(diff0 - m_phys)) + 0.5 * np.sum(np.square(diff1 - m_phys)))

        goal = goals_e[e0, query_step]
        goal_rel = goal[2:4] - x_phys
        theta = float(physics_e[e0, query_step, 10])
        q_action = actions_e[e0, query_step:future_step].ravel()
        desc = np.concatenate([
            x_phys, [physics_e[e0, query_step, 8], physics_e[e0, query_step, 9],
                     _math.sin(theta), _math.cos(theta)], goal_rel, q_action,
        ])
        C_pix_list.append(C_pix)
        m_pix_list.append(m_pix)
        C_phys_list.append(C_phys)
        desc_list.append(desc)
        cids.append(pidx)

    C_pix_arr = np.array(C_pix_list)
    m_pix_arr = np.stack(m_pix_list)
    C_phys_arr = np.array(C_phys_list)
    desc_arr = np.stack(desc_list)
    cid_arr = np.array(cids, dtype=np.int64)

    desc_scaled, _, _ = robust_scale(desc_arr)
    neighbors, _ = leave_cluster_out_knn(desc_scaled, cid_arr,
                                         max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk = background_variance_by_k(m_pix_arr, neighbors, KS)
    wq = np.full(len(C_pix_arr), 1.0 / len(C_pix_arr))
    rhos = {k: ratio_of_means(C_pix_arr, bvk[k], wq) for k in KS}
    ci = bootstrap_ratio_of_means_ci(C_pix_arr, bvk[MAIN_K], cid_arr,
                                     n_resamples=BOOTSTRAP_RESAMPLES, seed=seed)
    receipt["train_lance_version"] = int(lance.dataset(str(train_lance)).version)
    return {
        "task": "action_strength",
        "status": "ok",
        "n_queries": len(C_pix_arr),
        "C_pixel": _distribution_summary(C_pix_arr),
        "C_physical_reference": _distribution_summary(C_phys_arr),
        "coordinate_definition": "per-pixel-channel MSE in [0,1]^P; KNN in PushT block-position space",
        "B": {str(k): _distribution_summary(bvk[k]) for k in KS},
        "rho": {str(k): float(rhos[k]) for k in KS},
        "rho_bootstrap_ci": ci,
        "evidence_type": "observed_matched_twin_binary",
        "condition_balance": {"n_conditions": 2, "weights": [0.5, 0.5]},
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Real PNG-backed implementations. These definitions intentionally replace the
# early scaffold implementations above after their data-contract review.
# ---------------------------------------------------------------------------

def _sample_without_replacement(count: int, total: int, seed: int) -> np.ndarray:
    _require(0 < total <= count, f"cannot sample {total} from {count}")
    return np.sort(np.random.default_rng(seed).choice(count, total, replace=False))


def _string_in_filter(column: str, values: Iterable[str]) -> str:
    quoted = ["'" + str(value).replace("'", "''") + "'" for value in values]
    _require(bool(quoted), f"empty values for {column} filter")
    return f"{column} IN ({','.join(quoted)})"


def _step_filter(steps: Iterable[int]) -> str:
    values = sorted({int(value) for value in steps})
    return "step_idx IN (" + ",".join(str(value) for value in values) + ")"


def _finalize_pixel_task(
    *,
    task: str,
    conditional: list[float],
    histories: list[float] | None,
    mean_displacements: list[np.ndarray],
    physical_reference: list[float],
    descriptors: list[np.ndarray],
    cluster_ids: list[int],
    evidence_type: str,
    condition_balance: dict[str, Any],
    coordinate_definition: str,
    receipt: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    C = np.asarray(conditional, dtype=np.float64)
    means = np.stack(mean_displacements).astype(np.float32, copy=False)
    C_phys = np.asarray(physical_reference, dtype=np.float64)
    desc = np.stack(descriptors).astype(np.float64, copy=False)
    cids = np.asarray(cluster_ids, dtype=np.int64)
    _require(len(C) > max(KS) + 10, f"too few {task} queries: {len(C)}")
    scaled, _, _ = robust_scale(desc)
    neighbors, _ = leave_cluster_out_knn(
        scaled, cids, max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE
    )
    backgrounds = background_variance_by_k(means, neighbors, KS)
    weights = np.full(len(C), 1.0 / len(C), dtype=np.float64)
    rhos = {k: ratio_of_means(C, backgrounds[k], weights) for k in KS}
    ci = bootstrap_ratio_of_means_ci(
        C, backgrounds[MAIN_K], cids,
        n_resamples=BOOTSTRAP_RESAMPLES, seed=seed,
    )
    history_payload: dict[str, Any]
    if histories is None:
        history_payload = {
            "status": "not_estimable_from_observed_training_twins",
            "reason": "no observed same-query cross-condition Training histories",
        }
    else:
        history_payload = _distribution_summary(np.asarray(histories, dtype=np.float64))
    return {
        "task": task,
        "status": "ok",
        "n_queries": int(len(C)),
        "units": PIXEL_UNITS,
        "C_pixel": _distribution_summary(C),
        "H_pixel": history_payload,
        "C_physical_reference": _distribution_summary(C_phys),
        "coordinate_definition": coordinate_definition,
        "B_pixel": {str(k): _distribution_summary(backgrounds[k]) for k in KS},
        "rho_pixel": {str(k): float(rhos[k]) for k in KS},
        "rho_pixel_k64_bootstrap_ci": ci,
        "evidence_type": evidence_type,
        "condition_balance": condition_balance,
        "receipt": receipt,
    }


def _read_selected_push_rows(
    *,
    train_lance: Path,
    pair_ids: list[str],
    steps: list[int],
    pixel_column: str,
) -> dict[str, dict[str, dict[int, dict[str, Any]]]]:
    import lance

    dataset = lance.dataset(str(train_lance))
    required = {
        "step_idx", "action", "physics_state", "goal_state",
        "pair_id", "hidden_mode", pixel_column,
    }
    _require(required <= set(dataset.schema.names), f"PushT schema lacks {sorted(required - set(dataset.schema.names))}")
    predicate = _string_in_filter("pair_id", pair_ids) + " AND " + _step_filter(steps)
    table = dataset.to_table(columns=sorted(required), filter=predicate)
    step_values = _scalar_col(table, "step_idx", np.int64)
    actions = _fixed_list_col(table, "action", 2)
    physics = _fixed_list_col(table, "physics_state", 12)
    goals = _fixed_list_col(table, "goal_state", 7)
    blobs = _pixel_blobs(table, pixel_column)
    pair_values = table["pair_id"].to_pylist()
    mode_values = table["hidden_mode"].to_pylist()
    records: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    for index, (pair_id, mode, step) in enumerate(zip(pair_values, mode_values, step_values)):
        step_records = records.setdefault(str(pair_id), {}).setdefault(str(mode), {})
        _require(int(step) not in step_records, f"duplicate row {pair_id}/{mode}/step{step}")
        step_records[int(step)] = {
            "action": actions[index],
            "physics": physics[index],
            "goal": goals[index],
            "pixels": blobs[index],
        }
    return records


def audit_motion_raw_pixel(
    cfg: dict[str, Any], repo_root: Path, check_only: bool,
    sample_per_task: int, seed: int,
) -> dict[str, Any]:
    import lance

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance); training_only_guard(manifest_path)
    _require(train_lance.exists() and manifest_path.exists(), "Motion Training inputs missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest = manifest["splits"]["train"]
    pair_count = int(cfg.get("expected_pair_count", 8192))
    _require(len(train_manifest["pairs"]) == pair_count, "Motion manifest pair count mismatch")
    pixel_column = str(cfg.get("pixel_column", "pixels"))
    image_shape = tuple(cfg.get("image_shape", [224, 224, 3]))
    dataset = lance.dataset(str(train_lance))
    _require(pixel_column in dataset.schema.names, f"Motion lacks {pixel_column}")
    receipt = {
        "manifest_sha256": file_sha256(manifest_path),
        "train_table_sha256_from_manifest": str(train_manifest["table_sha256"]),
        "sample_per_task": int(sample_per_task),
        "check_only": bool(check_only),
    }
    if check_only:
        return {"task": "motion", "status": "check_ok", "receipt": receipt}

    _require(sample_per_task % 2 == 0, "Motion sample count must preserve both twin directions")
    twin_indices = _sample_without_replacement(pair_count // 2, sample_per_task // 2, seed)
    pair_indices = np.sort(np.concatenate([2 * twin_indices, 2 * twin_indices + 1]))
    pair_ids = [
        f"pmd-train-{int(index):06d}-{'forward' if int(index) % 2 == 0 else 'reverse'}"
        for index in pair_indices
    ]
    query_step = int(cfg.get("query_step", 10)); future_step = int(cfg.get("future_step", 15))
    history_steps = [int(value) for value in cfg.get("history_steps", [0, 5])]
    steps = sorted(set(history_steps + list(range(query_step, future_step + 1))))
    records = _read_selected_push_rows(
        train_lance=train_lance, pair_ids=pair_ids, steps=steps, pixel_column=pixel_column
    )
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    modes = ("faster_decay", "no_extra_decay")
    C: list[float] = []; H: list[float] = []; means: list[np.ndarray] = []
    C_phys: list[float] = []; desc: list[np.ndarray] = []; cids: list[int] = []
    query_pixel_residual = 0.0; action_residual = 0.0
    for pair_index, pair_id in zip(pair_indices, pair_ids):
        by_mode = records.get(pair_id, {})
        _require(set(by_mode) == set(modes), f"incomplete Motion modes for {pair_id}")
        for mode in modes:
            _require(set(by_mode[mode]) == set(steps), f"incomplete Motion steps for {pair_id}/{mode}")
        query_pixel_residual = max(
            query_pixel_residual,
            float(by_mode[modes[0]][query_step]["pixels"] != by_mode[modes[1]][query_step]["pixels"]),
        )
        for step in range(query_step, future_step):
            action_residual = max(action_residual, float(np.max(np.abs(
                by_mode[modes[0]][step]["action"] - by_mode[modes[1]][step]["action"]
            ))))
        _require(query_pixel_residual == 0.0, f"Motion query pixels differ for {pair_id}")
        x_pixel = decode_pixel_blob(by_mode[modes[0]][query_step]["pixels"], image_shape)
        futures = np.stack([
            decode_pixel_blob(by_mode[mode][future_step]["pixels"], image_shape) for mode in modes
        ])
        C.append(float(pixel_conditional_variance(futures[None], x_pixel[None], weights)[0]))
        means.append(pixel_mean_displacement(futures[None], x_pixel[None], weights)[0].astype(np.float32))
        history = np.stack([[decode_pixel_blob(by_mode[mode][step]["pixels"], image_shape)
                             for step in history_steps] for mode in modes])
        H.append(pixel_history_conditional_variance(history, weights))
        query = by_mode[modes[0]][query_step]["physics"]
        x_phys = query[6:8]
        y_phys = np.stack([by_mode[mode][future_step]["physics"][6:8] for mode in modes])
        C_phys.append(float(np.sum(np.square(y_phys[1] - y_phys[0])) / 4.0))
        goal = by_mode[modes[0]][query_step]["goal"]
        theta = float(query[10]); goal_relative = goal[2:4] - x_phys
        desc.append(np.asarray([
            x_phys[0], x_phys[1], query[8], query[9],
            math.sin(theta), math.cos(theta), goal_relative[0], goal_relative[1],
        ], dtype=np.float64))
        cids.append(int(pair_index) // 2)
    _require(action_residual <= 1.0e-6, f"Motion query actions differ: {action_residual}")
    receipt.update({
        "sample_pair_ids_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
        "query_pixel_mismatch_max": query_pixel_residual,
        "query_action_residual_max": action_residual,
    })
    return _finalize_pixel_task(
        task="motion", conditional=C, histories=H, mean_displacements=means,
        physical_reference=C_phys, descriptors=desc, cluster_ids=cids,
        evidence_type="observed_matched_twin_binary",
        condition_balance={"n_conditions": 2, "weights": weights.tolist()},
        coordinate_definition="raw RGB target; KNN uses frozen PushT physical query descriptor",
        receipt=receipt, seed=seed,
    )


def audit_action_strength_raw_pixel(
    cfg: dict[str, Any], repo_root: Path, check_only: bool,
    sample_per_task: int, seed: int,
) -> dict[str, Any]:
    import lance

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance); training_only_guard(manifest_path)
    _require(train_lance.exists() and manifest_path.exists(), "ActionStrength Training inputs missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_count = int(cfg.get("expected_pair_count", 2048))
    _require(len(manifest["splits"]["train"]["pairs"]) == pair_count, "ActionStrength pair count mismatch")
    pixel_column = str(cfg.get("pixel_column", "pixels"))
    image_shape = tuple(cfg.get("image_shape", [224, 224, 3]))
    dataset = lance.dataset(str(train_lance))
    _require(pixel_column in dataset.schema.names, f"ActionStrength lacks {pixel_column}")
    receipt = {
        "manifest_sha256": file_sha256(manifest_path),
        "train_lance_version": int(dataset.version),
        "sample_per_task": int(sample_per_task),
        "check_only": bool(check_only),
    }
    if check_only:
        return {"task": "action_strength", "status": "check_ok", "receipt": receipt}

    pair_indices = _sample_without_replacement(pair_count, sample_per_task, seed)
    pair_ids = [f"phrm-train-{int(index):05d}" for index in pair_indices]
    query_step = int(cfg.get("query_step", 10)); future_step = int(cfg.get("future_step", 15))
    history_steps = [int(value) for value in cfg.get("history_steps", [0, 5])]
    steps = sorted(set(history_steps + list(range(query_step, future_step + 1))))
    records = _read_selected_push_rows(
        train_lance=train_lance, pair_ids=pair_ids, steps=steps, pixel_column=pixel_column
    )
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    modes = ("low_gain", "high_gain")
    C: list[float] = []; H: list[float] = []; means: list[np.ndarray] = []
    C_phys: list[float] = []; desc: list[np.ndarray] = []; cids: list[int] = []
    action_residual = 0.0
    for pair_index, pair_id in zip(pair_indices, pair_ids):
        by_mode = records.get(pair_id, {})
        _require(set(by_mode) == set(modes), f"incomplete ActionStrength modes for {pair_id}")
        for mode in modes:
            _require(set(by_mode[mode]) == set(steps), f"incomplete ActionStrength steps for {pair_id}/{mode}")
        _require(by_mode[modes[0]][query_step]["pixels"] == by_mode[modes[1]][query_step]["pixels"],
                 f"ActionStrength query pixels differ for {pair_id}")
        for step in range(query_step, future_step):
            action_residual = max(action_residual, float(np.max(np.abs(
                by_mode[modes[0]][step]["action"] - by_mode[modes[1]][step]["action"]
            ))))
        x_pixel = decode_pixel_blob(by_mode[modes[0]][query_step]["pixels"], image_shape)
        futures = np.stack([
            decode_pixel_blob(by_mode[mode][future_step]["pixels"], image_shape) for mode in modes
        ])
        C.append(float(pixel_conditional_variance(futures[None], x_pixel[None], weights)[0]))
        means.append(pixel_mean_displacement(futures[None], x_pixel[None], weights)[0].astype(np.float32))
        history = np.stack([[decode_pixel_blob(by_mode[mode][step]["pixels"], image_shape)
                             for step in history_steps] for mode in modes])
        H.append(pixel_history_conditional_variance(history, weights))
        query = by_mode[modes[0]][query_step]["physics"]
        x_phys = query[6:8]
        y_phys = np.stack([by_mode[mode][future_step]["physics"][6:8] for mode in modes])
        C_phys.append(float(np.sum(np.square(y_phys[1] - y_phys[0])) / 4.0))
        goal = by_mode[modes[0]][query_step]["goal"]
        theta = float(query[10]); goal_relative = goal[2:4] - x_phys
        actions = np.concatenate([by_mode[modes[0]][step]["action"] for step in range(query_step, future_step)])
        desc.append(np.concatenate([x_phys, [query[8], query[9], math.sin(theta), math.cos(theta)],
                                    goal_relative, actions]))
        cids.append(int(pair_index))
    _require(action_residual <= 1.0e-6, f"ActionStrength query actions differ: {action_residual}")
    receipt.update({
        "sample_pair_ids_sha256": hashlib.sha256("\n".join(pair_ids).encode()).hexdigest(),
        "query_action_residual_max": action_residual,
    })
    return _finalize_pixel_task(
        task="action_strength", conditional=C, histories=H, mean_displacements=means,
        physical_reference=C_phys, descriptors=desc, cluster_ids=cids,
        evidence_type="observed_matched_twin_binary",
        condition_balance={"n_conditions": 2, "weights": weights.tolist()},
        coordinate_definition="raw RGB target; KNN uses frozen PushT physical query/action descriptor",
        receipt=receipt, seed=seed,
    )


def audit_action_delay_raw_pixel(
    cfg: dict[str, Any], repo_root: Path, check_only: bool,
    sample_per_task: int, seed: int,
) -> dict[str, Any]:
    """Measure observed cross-delay visibility from balanced Training twins."""
    import lance

    table_dir = Path(cfg["train_table_dir"]).resolve()
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(table_dir); training_only_guard(manifest_path)
    _require(table_dir.exists() and manifest_path.exists(), "ActionDelay Training inputs missing")
    manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    train_rows = [row for row in manifest_rows if str(row.get("split", "")).lower() == "train"]
    delays, delay_weight_values = _delay_weights()
    weights = np.asarray(delay_weight_values, dtype=np.float64)
    by_pair: dict[str, dict[int, dict[str, Any]]] = {}
    for row in train_rows:
        pair_id = str(row["pair_id"])
        delay = int(row["factors"]["action.delay_steps"])
        _require(delay not in by_pair.setdefault(pair_id, {}), f"duplicate {pair_id}/d{delay}")
        by_pair[pair_id][delay] = row
    _require(len(by_pair) == 32, f"expected 32 ActionDelay pair shards, found {len(by_pair)}")
    _require(all(set(rows) == set(delays) for rows in by_pair.values()), "incomplete ActionDelay grid")

    def row_path(row: dict[str, Any]) -> Path:
        path = (table_dir / Path(str(row["output_path"])).name).resolve()
        training_only_guard(path)
        _require(path.exists(), f"missing ActionDelay table: {path}")
        return path

    pixel_column = str(cfg.get("pixel_column", "pixels"))
    image_shape = tuple(cfg.get("image_shape", [224, 224, 3]))
    query_step = int(cfg.get("query_step", 30)); future_step = int(cfg.get("future_step", 35))
    history_steps = [int(value) for value in cfg.get("history_steps", [0, 5, 10, 15, 20, 25])]
    selected_steps = sorted(set(history_steps + list(range(query_step, future_step + 1))))
    per_pair = sample_per_task // len(by_pair)
    _require(sample_per_task % len(by_pair) == 0 and per_pair > 0,
             "ActionDelay sample count must be positive and divisible by 32")
    episode_count = int(next(iter(train_rows)).get("episodes", 160))
    _require(per_pair <= episode_count, "ActionDelay sample exceeds episodes per pair")
    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "train_storage_digest": manifest_storage_digest(train_rows),
        "train_manifest_rows": len(train_rows),
        "pair_shards": len(by_pair),
        "sample_per_task": int(sample_per_task),
        "episodes_per_pair_sampled": int(per_pair),
        "check_only": bool(check_only),
    }
    if check_only:
        for delay_rows in by_pair.values():
            for row in delay_rows.values():
                dataset = lance.dataset(str(row_path(row)))
                required = {"episode_idx", "step_idx", "proprio", "action", pixel_column}
                _require(required <= set(dataset.schema.names), "ActionDelay schema mismatch")
        return {"task": "action_delay", "status": "check_ok", "receipt": receipt}

    C: list[float] = []; H: list[float] = []; means: list[np.ndarray] = []
    C_phys: list[float] = []; desc: list[np.ndarray] = []; cids: list[int] = []
    sample_ids: list[str] = []
    action_residual_max = 0.0; query_position_residual_max = 0.0
    query_pixel_mismatch_max = 0.0

    for pair_index, (pair_id, delay_rows) in enumerate(sorted(by_pair.items())):
        episode_ids = _sample_without_replacement(
            episode_count, per_pair, seed + 1009 * pair_index
        )
        episode_clause = "episode_idx IN (" + ",".join(str(int(v)) for v in episode_ids) + ")"
        step_clause = _step_filter(selected_steps)
        records_by_delay: dict[int, dict[int, dict[int, dict[str, Any]]]] = {}
        for delay in delays:
            dataset = lance.dataset(str(row_path(delay_rows[delay])))
            required = {"episode_idx", "step_idx", "proprio", "action", pixel_column}
            _require(required <= set(dataset.schema.names), f"ActionDelay schema mismatch: {pair_id}/d{delay}")
            table = dataset.to_table(
                columns=sorted(required), filter=episode_clause + " AND " + step_clause
            )
            episodes = _scalar_col(table, "episode_idx", np.int64)
            steps = _scalar_col(table, "step_idx", np.int64)
            positions = _fixed_list_col(table, "proprio", 2)
            actions = _fixed_list_col(table, "action", 2)
            blobs = _pixel_blobs(table, pixel_column)
            delay_records: dict[int, dict[int, dict[str, Any]]] = {}
            for index, (episode_id, step) in enumerate(zip(episodes, steps)):
                step_records = delay_records.setdefault(int(episode_id), {})
                _require(int(step) not in step_records, "duplicate ActionDelay selected row")
                step_records[int(step)] = {
                    "position": positions[index], "action": actions[index], "pixels": blobs[index]
                }
            expected_episode_ids = {int(value) for value in episode_ids}
            _require(set(delay_records) == expected_episode_ids, f"missing episodes: {pair_id}/d{delay}")
            _require(all(set(rows) == set(selected_steps) for rows in delay_records.values()),
                     f"missing steps: {pair_id}/d{delay}")
            records_by_delay[delay] = delay_records

        for episode_id in episode_ids:
            episode_id = int(episode_id)
            query_records = [records_by_delay[d][episode_id][query_step] for d in delays]
            reference_query = query_records[0]
            query_position_residual_max = max(
                query_position_residual_max,
                max(float(np.max(np.abs(row["position"] - reference_query["position"])))
                    for row in query_records),
            )
            query_pixel_mismatch_max = max(
                query_pixel_mismatch_max,
                float(any(row["pixels"] != reference_query["pixels"] for row in query_records)),
            )
            for step in range(query_step, future_step):
                reference_action = records_by_delay[0][episode_id][step]["action"]
                action_residual_max = max(
                    action_residual_max,
                    max(float(np.max(np.abs(records_by_delay[d][episode_id][step]["action"] - reference_action)))
                        for d in delays),
                )
            x_pixel = decode_pixel_blob(reference_query["pixels"], image_shape)
            futures = np.stack([
                decode_pixel_blob(records_by_delay[d][episode_id][future_step]["pixels"], image_shape)
                for d in delays
            ])
            C.append(float(pixel_conditional_variance(futures[None], x_pixel[None], weights)[0]))
            means.append(pixel_mean_displacement(futures[None], x_pixel[None], weights)[0].astype(np.float32))
            histories = np.stack([[
                decode_pixel_blob(records_by_delay[d][episode_id][step]["pixels"], image_shape)
                for step in history_steps
            ] for d in delays])
            H.append(pixel_history_conditional_variance(histories, weights))

            x_phys = reference_query["position"]
            y_phys = np.stack([
                records_by_delay[d][episode_id][future_step]["position"] for d in delays
            ])
            physical_center = np.einsum("c,cd->d", weights, y_phys)
            C_phys.append(float(np.einsum(
                "c,c->", weights, np.sum(np.square(y_phys - physical_center[None]), axis=1)
            )))
            query_actions = np.concatenate([
                records_by_delay[0][episode_id][step]["action"]
                for step in range(query_step, future_step)
            ])
            desc.append(np.concatenate([x_phys, query_actions]))
            cids.append(len(cids))
            sample_ids.append(f"{pair_id}:e{episode_id}")

    _require(action_residual_max <= 1.0e-6, f"ActionDelay actions differ: {action_residual_max}")
    _require(query_position_residual_max <= 1.0e-6,
             f"ActionDelay query positions differ: {query_position_residual_max}")
    _require(query_pixel_mismatch_max == 0.0, "ActionDelay query pixels differ")
    receipt.update({
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "action_match_residual_max": action_residual_max,
        "query_position_residual_max": query_position_residual_max,
        "query_pixel_mismatch_max": query_pixel_mismatch_max,
    })
    return _finalize_pixel_task(
        task="action_delay", conditional=C, histories=H, mean_displacements=means,
        physical_reference=C_phys, descriptors=desc, cluster_ids=cids,
        evidence_type="observed_matched_condition_group",
        condition_balance={
            "n_conditions": len(delays),
            "weights": {str(d): float(w) for d, w in zip(delays, delay_weight_values)},
        },
        coordinate_definition="raw RGB target; KNN uses frozen TwoRoom query-position/action descriptor",
        receipt=receipt, seed=seed,
    )


def _tworoom_collision_raw(
    pos: np.ndarray, proposed: np.ndarray, **geometry: Any,
) -> np.ndarray:
    """Pure NumPy replay of the frozen vertical-wall TwoRoom collision rule."""
    image_size = float(geometry.get("image_size", 224.0))
    border_size = float(geometry.get("border_size", 14.0))
    agent_radius = float(geometry.get("agent_radius", 7.0))
    wall_center = float(geometry.get("wall_center", 112.0))
    wall_thickness = int(geometry.get("wall_thickness", 10))
    door_center = float(geometry.get("door_center", 49.0))
    door_half_size = float(geometry.get("door_half_size", 14.0))
    door_margin = float(geometry.get("door_margin", 1.75))
    current = np.asarray(pos, dtype=np.float32)
    candidate = np.asarray(proposed, dtype=np.float32).copy()
    candidate = np.clip(
        candidate, np.float32(border_size + agent_radius),
        np.float32(image_size - border_size - agent_radius),
    ).astype(np.float32)
    half = wall_thickness // 2
    effective_left = wall_center - half - agent_radius
    effective_right = wall_center + half + agent_radius
    in_door = (
        door_center - door_half_size - door_margin
        <= float(candidate[1])
        <= door_center + door_half_size + door_margin
    )
    if float(current[0]) < wall_center:
        if float(candidate[0]) > effective_left and not in_door:
            candidate[0] = np.float32(effective_left - 0.5)
    elif float(candidate[0]) < effective_right and not in_door:
        candidate[0] = np.float32(effective_right + 0.5)
    return candidate.astype(np.float64)


def _speed_manifest_value(row: dict[str, Any]) -> float:
    value = row.get("factors", {}).get("agent.speed")
    _require(value is not None, f"Speed row lacks agent.speed: {row.get('scenario_id')}")
    return float(value)


def _speed_training_path(data_root: Path, row: dict[str, Any]) -> Path:
    prefix = Path("artifacts/synthesis/data/tworoom_speed_full_v1")
    output_path = Path(str(row["output_path"]))
    try:
        relative = output_path.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(f"unexpected Speed output_path: {output_path}") from exc
    path = (data_root / relative).resolve()
    training_only_guard(path)
    _require(path.exists(), f"Speed Training table missing: {path}")
    return path


def audit_speed_raw_pixel(
    cfg: dict[str, Any], repo_root: Path, check_only: bool,
    sample_per_task: int, seed: int,
) -> dict[str, Any]:
    """Render cross-speed futures on observed Training queries and verify replay."""
    import lance

    manifest_path = _resolve(repo_root, cfg["manifest"])
    data_root = Path(cfg["data_root"]).resolve()
    training_only_guard(manifest_path); training_only_guard(data_root)
    _require(manifest_path.exists() and data_root.exists(), "Speed Training inputs missing")
    all_manifest_rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scenario_rows = [
        row for row in all_manifest_rows if str(row.get("split", "")).lower() == "train"
    ]
    n_speeds = int(cfg.get("n_speeds", 32))
    by_speed: dict[float, list[dict[str, Any]]] = {}
    for row in scenario_rows:
        by_speed.setdefault(_speed_manifest_value(row), []).append(row)
    training_speeds = sorted(by_speed)
    _require(len(training_speeds) == n_speeds,
             f"expected {n_speeds} Speed values, found {len(training_speeds)}")
    _require(sample_per_task % n_speeds == 0,
             "Speed sample count must be divisible by number of speeds")
    per_speed = sample_per_task // n_speeds
    _require(per_speed > 0 and all(len(rows) >= per_speed for rows in by_speed.values()),
             "Speed sample exceeds Training scenarios per speed")
    pixel_column = str(cfg.get("pixel_column", "pixels"))
    image_shape = tuple(cfg.get("image_shape", [224, 224, 3]))
    geometry = dict(cfg.get("geometry", {}))
    selected_steps = [0, 5, 10, 11, 12, 13, 14, 15]
    step_clause = _step_filter(selected_steps)
    selected_scenarios: list[dict[str, Any]] = []
    for speed in training_speeds:
        ranked = sorted(
            by_speed[speed],
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['scenario_id']}".encode("utf-8")
            ).hexdigest(),
        )
        selected_scenarios.extend(ranked[:per_speed])
    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "train_storage_digest": manifest_storage_digest(scenario_rows),
        "n_train_scenarios": len(scenario_rows),
        "n_training_speeds": len(training_speeds),
        "sample_per_task": int(sample_per_task),
        "scenarios_per_observed_speed": int(per_speed),
        "check_only": bool(check_only),
        "evidence_type": SPEED_PIXEL_STATUS,
    }
    if check_only:
        for row in scenario_rows:
            dataset = lance.dataset(str(_speed_training_path(data_root, row)))
            required = {"episode_idx", "step_idx", "proprio", "action", "goal_state", pixel_column}
            _require(required <= set(dataset.schema.names), "Speed schema mismatch")
        return {"task": "speed", "status": "check_ok", "receipt": receipt}

    candidates: list[dict[str, Any]] = []
    for row in selected_scenarios:
        scene_path = _speed_training_path(data_root, row)
        dataset = lance.dataset(str(scene_path))
        required = {"episode_idx", "step_idx", "proprio", "action", "goal_state", pixel_column}
        _require(required <= set(dataset.schema.names), f"Speed schema mismatch: {scene_path}")
        table = dataset.to_table(columns=sorted(required), filter=step_clause)
        episodes = _scalar_col(table, "episode_idx", np.int64)
        steps = _scalar_col(table, "step_idx", np.int64)
        positions = _fixed_list_col(table, "proprio", 2)
        actions = _fixed_list_col(table, "action", 2)
        goals = _fixed_list_col(table, "goal_state", 2)
        blobs = _pixel_blobs(table, pixel_column)
        records: dict[int, dict[int, dict[str, Any]]] = {}
        for index, (episode_id, step) in enumerate(zip(episodes, steps)):
            records.setdefault(int(episode_id), {})[int(step)] = {
                "position": positions[index], "action": actions[index],
                "goal": goals[index], "pixels": blobs[index],
            }
        valid_episode_ids = [
            episode_id for episode_id, rows in records.items()
            if set(rows) == set(selected_steps)
        ]
        _require(valid_episode_ids, f"Speed scenario has no complete query: {row['scenario_id']}")
        episode_id = min(
            valid_episode_ids,
            key=lambda value: hashlib.sha256(
                f"{seed}:{row['scenario_id']}:{value}".encode("utf-8")
            ).hexdigest(),
        )
        candidates.append({
            "scenario_id": str(row["scenario_id"]),
            "episode_id": int(episode_id),
            "observed_speed": _speed_manifest_value(row),
            "rows": records[int(episode_id)],
        })

    import torch
    from stable_worldmodel.envs.two_room.env import TwoRoomEnv

    env = TwoRoomEnv(render_target=False)
    env.reset(options={
        "variation": (),
        "variation_values": {
            "agent.speed": np.asarray([5.0], dtype=np.float32),
            "agent.radius": np.asarray([7.0], dtype=np.float32),
            "door.position": np.asarray([49, 49, 49]),
            "door.size": np.asarray([14, 14, 14]),
            "door.number": 1,
            "wall.axis": 1,
            "wall.thickness": 10,
            "rendering.render_target": 0,
        },
        "state": np.asarray([60.0, 112.0], dtype=np.float32),
        "target_state": np.asarray([164.0, 112.0], dtype=np.float32),
    })

    def render_position(position: np.ndarray) -> np.ndarray:
        chw = env._render_frame(torch.as_tensor(position, dtype=torch.float32))
        return chw.cpu().numpy().transpose(1, 2, 0).reshape(-1).astype(np.float32) / np.float32(255.0)

    weights = np.full(n_speeds, 1.0 / n_speeds, dtype=np.float64)
    C: list[float] = []; means: list[np.ndarray] = []; C_phys: list[float] = []
    desc: list[np.ndarray] = []; cids: list[int] = []; sample_ids: list[str] = []
    replay_residuals: list[float] = []; render_residuals: list[float] = []
    observed_speed_counts = {str(speed): 0 for speed in training_speeds}
    for candidate in candidates:
        rows = candidate["rows"]
        x_phys = rows[10]["position"]
        query_actions = np.stack([rows[step]["action"] for step in range(10, 15)])
        counterfactual_positions: list[np.ndarray] = []
        for speed in training_speeds:
            position = x_phys.copy()
            for action in query_actions:
                proposed = np.asarray(position, dtype=np.float32) + (
                    np.clip(action, -1.0, 1.0).astype(np.float32) * np.float32(speed)
                )
                position = _tworoom_collision_raw(position, proposed, **geometry)
            counterfactual_positions.append(position)
        observed_speed = float(candidate["observed_speed"])
        observed_index = training_speeds.index(observed_speed)
        observed_future = rows[15]["position"]
        replay_residuals.append(float(np.max(np.abs(
            counterfactual_positions[observed_index] - observed_future
        ))))
        x_pixel = decode_pixel_blob(rows[10]["pixels"], image_shape)
        futures = np.stack([render_position(position) for position in counterfactual_positions])
        stored_future = decode_pixel_blob(rows[15]["pixels"], image_shape)
        render_residuals.append(float(np.max(np.abs(futures[observed_index] - stored_future))))
        C.append(float(pixel_conditional_variance(futures[None], x_pixel[None], weights)[0]))
        means.append(pixel_mean_displacement(futures[None], x_pixel[None], weights)[0].astype(np.float32))
        positions_array = np.stack(counterfactual_positions)
        physical_center = np.mean(positions_array, axis=0)
        C_phys.append(float(np.mean(np.sum(np.square(
            positions_array - physical_center[None]
        ), axis=1))))
        goal_relative = rows[10]["goal"] - x_phys
        desc.append(np.concatenate([x_phys, goal_relative, query_actions.ravel()]))
        cids.append(len(cids))
        observed_speed_counts[str(observed_speed)] += 1
        sample_ids.append(f"{candidate['scenario_id']}:e{candidate['episode_id']}")

    replay = np.asarray(replay_residuals, dtype=np.float64)
    render = np.asarray(render_residuals, dtype=np.float64)
    _require(float(np.max(replay)) <= 1.0e-3,
             f"Speed physical replay residual too large: {float(np.max(replay))}")
    _require(float(np.max(render)) == 0.0,
             f"Speed renderer mismatch: {float(np.max(render))}")
    receipt.update({
        "sample_ids_sha256": hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
        "physical_replay_residual_max": float(np.max(replay)),
        "physical_replay_residual_p99": float(np.percentile(replay, 99)),
        "rendered_observed_future_residual_max": float(np.max(render)),
        "observed_speed_query_counts": observed_speed_counts,
    })
    return _finalize_pixel_task(
        task="speed", conditional=C, histories=None, mean_displacements=means,
        physical_reference=C_phys, descriptors=desc, cluster_ids=cids,
        evidence_type=SPEED_PIXEL_STATUS,
        condition_balance={"n_conditions": n_speeds, "weights": "uniform_1_over_n_speeds"},
        coordinate_definition="simulator-rendered raw RGB target; KNN uses frozen TwoRoom query/action descriptor",
        receipt=receipt, seed=seed,
    )


if __name__ == "__main__":
    main()
