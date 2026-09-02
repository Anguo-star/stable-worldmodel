#!/usr/bin/env python3
"""Training-only cross-task conditional visibility CPU auditor (v1).

Measures whether physical conditional energy is detectable in frozen Training
pools across Motion, ActionStrength, ActionDelay, and Speed tasks using
exact leave-cluster-out KNN and cluster bootstrap.

Evidence boundary (immutable):
  pixel_target_separation_status = pending_separate_raw_pixel_audit
  latent_status                  = pending_gpu_forward
  binding_cause_status           = not_claimed
  claim_scope                    = frozen_training_only_descriptive_upstream_visibility

No model is loaded, no optimizer step is taken, no pixels are decoded.
Do not use CPU-layer results to claim model use of history or binding cause.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
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
PIXELS_DECODED: bool = False
MODEL_LOADED: bool = False
OPTIMIZER_STEPS: int = 0
PIXEL_TARGET_SEPARATION_STATUS: str = "pending_separate_raw_pixel_audit"
HISTORY_INFERABILITY_STATUS: str = "pending_separate_full_observation_or_raw_pixel_audit"
LATENT_STATUS: str = "pending_gpu_forward"
BINDING_CAUSE_STATUS: str = "not_claimed"
CLAIM_SCOPE: str = "frozen_training_only_descriptive_upstream_visibility"
SPEED_EVIDENCE_TYPE: str = "simulator_counterfactual_on_training_queries"

KS: tuple[int, ...] = (32, 64, 128)
MAIN_K: int = 64
NEIGHBOR_CHUNK_SIZE: int = 128
BOOTSTRAP_RESAMPLES: int = 4096
BOOTSTRAP_SEED: int = 20260901
TAU: float = 1.0e-8

_FORBIDDEN_TOKENS: frozenset[str] = frozenset(
    {"development", "public", "test", "validation", "val"}
)


# ---------------------------------------------------------------------------
# Guards and utilities
# ---------------------------------------------------------------------------

def training_only_guard(path: Path) -> None:
    """Raise ValueError if any path token matches a forbidden split name."""
    parts = set(path.parts)
    # also split on underscores and hyphens within each part
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


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(v for v in path.rglob("*") if v.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\0")
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
# Core math (tested directly)
# ---------------------------------------------------------------------------

def weighted_conditional_variance(
    Y: "np.ndarray",   # (N, C, D)
    x: "np.ndarray",   # (N, D)
    w: "np.ndarray",   # (C,) — must sum to 1
) -> "np.ndarray":
    """Per-query conditional variance C_u = sum_c w_c ||(y_uc-x_u)-m_u||^2.

    For binary uniform w=[0.5,0.5] this equals ||Delta y||^2 / 4 exactly.
    """
    Y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    _require(Y.ndim == 3, "Y must be (N,C,D)")
    _require(x.ndim == 2 and x.shape == (Y.shape[0], Y.shape[2]), "x shape mismatch")
    _require(w.ndim == 1 and len(w) == Y.shape[1], "w shape mismatch")
    _require(math.isclose(float(w.sum()), 1.0, abs_tol=1e-9), "w must sum to 1")
    diffs = Y - x[:, None, :]          # (N, C, D)
    m = np.einsum("c,ncd->nd", w, diffs)  # (N, D)
    residuals = diffs - m[:, None, :]  # (N, C, D)
    sq = np.einsum("ncd,ncd->nc", residuals, residuals)  # (N, C)
    return np.einsum("c,nc->n", w, sq)  # (N,)


def history_energy(
    history_states: "np.ndarray",  # (N, T_hist, D) — model-visible physical history
    x: "np.ndarray",               # (N, D) — query physical state
) -> "np.ndarray":
    """H_u = mean squared distance of visible history states from query state."""
    H = np.asarray(history_states, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    _require(H.ndim == 3 and x.ndim == 2, "bad shapes for history_energy")
    _require(H.shape[0] == x.shape[0] and H.shape[2] == x.shape[1], "shape mismatch")
    diffs = H - x[:, None, :]  # (N, T_hist, D)
    return np.mean(np.sum(np.square(diffs), axis=2), axis=1)  # (N,)


def weighted_history_conditional_variance(
    histories: "np.ndarray",  # (N, C, T_hist, D)
    w: "np.ndarray",          # (C,)
) -> "np.ndarray":
    """Mean over visible time of the weighted cross-condition history variance."""
    histories = np.asarray(histories, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    _require(histories.ndim == 4, "histories must be (N,C,T,D)")
    _require(w.shape == (histories.shape[1],), "history weights shape mismatch")
    _require(math.isclose(float(w.sum()), 1.0, abs_tol=1e-9), "w must sum to 1")
    center = np.einsum("c,nctd->ntd", w, histories)
    residual = histories - center[:, None, :, :]
    per_condition_time = np.sum(np.square(residual), axis=3)
    return np.mean(np.einsum("c,nct->nt", w, per_condition_time), axis=1)


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
    descriptors: "np.ndarray",   # (N, D) already scaled
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
        q = desc[start:stop]                          # (Q, D)
        diff = q[:, None, :] - desc[None, :, :]      # (Q, N, D)
        dsq = np.einsum("qnd,qnd->qn", diff, diff, optimize=False)  # (Q, N)
        for local, row in enumerate(range(start, stop)):
            # mask own cluster
            dsq[local, members[int(cids[row])]] = np.inf
            # partition to get top max_k
            part = np.argpartition(dsq[local], max_k - 1)[:max_k]
            cutoff = float(np.max(dsq[local, part]))
            strict = np.flatnonzero(dsq[local] < cutoff)
            tied = np.flatnonzero(dsq[local] == cutoff)
            tied = tied[np.argsort(tied, kind="stable")]  # stable tie-break by index
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
    mean_displacements: "np.ndarray",   # (N, D) per-cluster mean displacement
    neighbors: "np.ndarray",            # (N, max_k) neighbor indices
    ks: Iterable[int],
) -> dict[int, "np.ndarray"]:
    """B_loc,k: for each query, mean sq distance of k neighbor m_v's around their mean."""
    means = np.asarray(mean_displacements, dtype=np.float64)
    _finite("mean displacements", means)
    result: dict[int, "np.ndarray"] = {}
    for k in ks:
        k = int(k)
        _require(0 < k <= neighbors.shape[1], f"invalid k={k}")
        nb_means = means[neighbors[:, :k]]            # (N, k, D)
        center = nb_means.mean(axis=1, keepdims=True)  # (N, 1, D)
        variance = np.mean(
            np.sum(np.square(nb_means - center), axis=2), axis=1
        )  # (N,)
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
# Bootstrap CI and distribution summary
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

    # build lookup: cluster -> indices
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
# Config loader
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except ImportError:
        # fallback: very minimal YAML subset for flat/nested dicts (no anchors)
        raise ImportError("PyYAML is required: pip install pyyaml")


def _resolve(base: Path, s: str) -> Path:
    p = Path(s)
    if p.is_absolute():
        return p.resolve()
    return (base / p).resolve()


# ---------------------------------------------------------------------------
# Motion loader
# ---------------------------------------------------------------------------

def _fixed_list_col(table: Any, name: str, width: int) -> "np.ndarray":
    col = table[name].combine_chunks()
    vals = np.asarray(col.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
    return vals.reshape(len(col), width)


def _scalar_col(table: Any, name: str, dtype: Any) -> "np.ndarray":
    return np.asarray(
        table[name].combine_chunks().to_numpy(zero_copy_only=False), dtype=dtype
    )


def audit_motion(cfg: dict[str, Any], repo_root: Path, check_only: bool) -> dict[str, Any]:
    import lance  # type: ignore

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance)
    training_only_guard(manifest_path)
    _require(train_lance.exists(), f"Motion train.lance missing: {train_lance}")
    _require(manifest_path.exists(), f"Motion manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits", {})
    train_m = splits.get("train", {})
    pair_count = cfg.get("expected_pair_count", 8192)
    ep_steps = cfg.get("episode_steps", 20)
    query_step = cfg.get("query_step", 10)
    future_step = cfg.get("future_step", 15)
    hist_steps = cfg.get("history_steps", [0, 5])
    tslice = cfg.get("target_slice", [6, 8])

    manifest_pairs = train_m.get("pairs", [])
    _require(len(manifest_pairs) == pair_count,
             f"Motion manifest pair count {len(manifest_pairs)} != {pair_count}")

    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "pair_count": pair_count,
        "check_only": check_only,
    }

    if check_only:
        return {"task": "motion", "status": "check_ok", "receipt": receipt}

    columns = ["episode_idx", "step_idx", "action", "physics_state",
               "goal_state", "pair_id", "hidden_mode", "split"]
    table = lance.dataset(str(train_lance)).to_table(columns=columns)
    expected_rows = 2 * pair_count * ep_steps
    _require(table.num_rows == expected_rows,
             f"Motion row count {table.num_rows} != {expected_rows}")

    ep_idx = _scalar_col(table, "episode_idx", np.int64)
    st_idx = _scalar_col(table, "step_idx", np.int64)
    actions = _fixed_list_col(table, "action", 2)
    physics = _fixed_list_col(table, "physics_state", 12)
    goals = _fixed_list_col(table, "goal_state", 7)
    pair_ids_raw = np.asarray(table["pair_id"].to_pylist(), dtype=object)
    modes_raw = np.asarray(table["hidden_mode"].to_pylist(), dtype=object)

    order = np.lexsort((st_idx, ep_idx))
    ep_idx = ep_idx[order]; st_idx = st_idx[order]
    actions = actions[order]; physics = physics[order]
    goals = goals[order]; pair_ids_raw = pair_ids_raw[order]; modes_raw = modes_raw[order]

    n_eps = 2 * pair_count
    actions_e = actions.reshape(n_eps, ep_steps, 2)
    physics_e = physics.reshape(n_eps, ep_steps, 12)
    goals_e = goals.reshape(n_eps, ep_steps, 7)
    pair_ids_e = pair_ids_raw.reshape(n_eps, ep_steps)
    modes_e = modes_raw.reshape(n_eps, ep_steps)

    # group episodes by pair_index and mode
    MODES = ("faster_decay", "no_extra_decay")
    episodes: dict[int, dict[str, int]] = {}
    for ep in range(n_eps):
        pid = str(pair_ids_e[ep, 0])
        import re as _re
        m = _re.fullmatch(r"pmd-train-(\d{6})-(forward|reverse)", pid)
        _require(m is not None, f"unexpected pair_id: {pid}")
        pidx = int(m.group(1))
        mode = str(modes_e[ep, 0])
        _require(mode in MODES, f"unexpected mode: {mode}")
        by_mode = episodes.setdefault(pidx, {})
        _require(mode not in by_mode, f"duplicate pair/mode: {pidx}/{mode}")
        by_mode[mode] = ep

    C_list, H_list, desc_list, m_list, twin_ids = [], [], [], [], []
    for pidx in range(pair_count):
        pe = episodes[pidx]
        _require(set(pe) == set(MODES), f"incomplete pair {pidx}")
        e0 = pe[MODES[0]]; e1 = pe[MODES[1]]
        _require(np.allclose(actions_e[e0], actions_e[e1], atol=1e-6),
                 f"actions differ pair {pidx}")
        _require(np.allclose(physics_e[e0, query_step], physics_e[e1, query_step], atol=1e-6),
                 f"query physics differ pair {pidx}")
        x = physics_e[e0, query_step, tslice[0]:tslice[1]]
        y0 = physics_e[e0, future_step, tslice[0]:tslice[1]]
        y1 = physics_e[e1, future_step, tslice[0]:tslice[1]]
        Y = np.stack([y0, y1])[None]  # (1,2,D)
        C_val = float(weighted_conditional_variance(Y, x[None], np.array([0.5, 0.5]))[0])
        hist0 = np.stack([physics_e[e0, s, tslice[0]:tslice[1]] for s in hist_steps])
        hist1 = np.stack([physics_e[e1, s, tslice[0]:tslice[1]] for s in hist_steps])
        hist = np.stack([hist0, hist1])[None]
        H_val = float(weighted_history_conditional_variance(
            hist, np.array([0.5, 0.5])
        )[0])
        m_val = 0.5 * ((y0 - x) + (y1 - x))
        goal = goals_e[e0, query_step]
        goal_rel = goal[2:4] - x
        import math as _math
        theta = float(physics_e[e0, query_step, 10])
        desc = np.array([x[0], x[1], physics_e[e0, query_step, 8], physics_e[e0, query_step, 9],
                         _math.sin(theta), _math.cos(theta), goal_rel[0], goal_rel[1]])
        C_list.append(C_val); H_list.append(H_val)
        m_list.append(m_val); desc_list.append(desc)
        twin_ids.append(pidx // 2)

    C = np.array(C_list); H = np.array(H_list)
    m_arr = np.stack(m_list)
    desc_arr = np.stack(desc_list)
    twin_arr = np.array(twin_ids, dtype=np.int64)
    desc_scaled, _, _ = robust_scale(desc_arr)

    neighbors, dists = leave_cluster_out_knn(desc_scaled, twin_arr,
                                             max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk = background_variance_by_k(m_arr, neighbors, KS)
    w = np.full(len(C), 1.0 / len(C))
    rhos = {k: ratio_of_means(C, bvk[k], w) for k in KS}
    ci = bootstrap_ratio_of_means_ci(C, bvk[MAIN_K], twin_arr)
    receipt["train_table_sha256_from_manifest"] = str(train_m["table_sha256"])
    return {
        "task": "motion",
        "status": "ok",
        "n_queries": len(C),
        "C": _distribution_summary(C),
        "target_slice_history_condition_energy_physical": _distribution_summary(H),
        "coordinate_definition": "PushT block position physics_state[6:8], pixel units squared",
        "B": {str(k): _distribution_summary(bvk[k]) for k in KS},
        "rho": {str(k): float(rhos[k]) for k in KS},
        "rho_bootstrap_ci": ci,
        "evidence_type": "observed_matched_twin_binary",
        "condition_balance": {"n_conditions": 2, "weights": [0.5, 0.5]},
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# ActionStrength loader
# ---------------------------------------------------------------------------

def audit_action_strength(cfg: dict[str, Any], repo_root: Path, check_only: bool) -> dict[str, Any]:
    import lance  # type: ignore

    train_lance = _resolve(repo_root, cfg["train_lance"])
    manifest_path = _resolve(repo_root, cfg["manifest"])
    training_only_guard(train_lance)
    training_only_guard(manifest_path)
    _require(train_lance.exists(), f"ActionStrength train.lance missing: {train_lance}")
    _require(manifest_path.exists(), f"ActionStrength manifest missing: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    pair_count = cfg.get("expected_pair_count", 2048)
    ep_steps = cfg.get("episode_steps", 20)
    query_step = cfg.get("query_step", 10)
    future_step = cfg.get("future_step", 15)
    hist_steps = cfg.get("history_steps", [0, 5])
    tslice = cfg.get("target_slice", [6, 8])

    train_m = manifest.get("splits", {}).get("train", {})
    manifest_pairs = train_m.get("pairs", [])
    _require(len(manifest_pairs) == pair_count,
             f"ActionStrength manifest pair count {len(manifest_pairs)} != {pair_count}")

    receipt: dict[str, Any] = {"manifest_sha256": file_sha256(manifest_path),
                                "pair_count": pair_count, "check_only": check_only}
    if check_only:
        return {"task": "action_strength", "status": "check_ok", "receipt": receipt}

    columns = ["episode_idx", "step_idx", "action", "physics_state",
               "goal_state", "pair_id", "hidden_mode", "split"]
    table = lance.dataset(str(train_lance)).to_table(columns=columns)
    expected_rows = 2 * pair_count * ep_steps
    _require(table.num_rows == expected_rows,
             f"ActionStrength row count {table.num_rows} != {expected_rows}")

    ep_idx = _scalar_col(table, "episode_idx", np.int64)
    st_idx = _scalar_col(table, "step_idx", np.int64)
    actions = _fixed_list_col(table, "action", 2)
    physics = _fixed_list_col(table, "physics_state", 12)
    goals = _fixed_list_col(table, "goal_state", 7)
    pair_ids_raw = np.asarray(table["pair_id"].to_pylist(), dtype=object)
    modes_raw = np.asarray(table["hidden_mode"].to_pylist(), dtype=object)

    order = np.lexsort((st_idx, ep_idx))
    ep_idx = ep_idx[order]; st_idx = st_idx[order]
    actions = actions[order]; physics = physics[order]
    goals = goals[order]; pair_ids_raw = pair_ids_raw[order]; modes_raw = modes_raw[order]

    n_eps = 2 * pair_count
    actions_e = actions.reshape(n_eps, ep_steps, 2)
    physics_e = physics.reshape(n_eps, ep_steps, 12)
    goals_e = goals.reshape(n_eps, ep_steps, 7)
    pair_ids_e = pair_ids_raw.reshape(n_eps, ep_steps)
    modes_e = modes_raw.reshape(n_eps, ep_steps)

    # Pair episodes by pair_id and hidden_mode.
    from collections import defaultdict as _dd
    episodes: dict[str, list[int]] = _dd(list)
    for ep in range(n_eps):
        pid = str(pair_ids_e[ep, 0])
        episodes[pid].append(ep)

    C_list: list[float] = []
    H_list: list[float] = []
    m_list: list[np.ndarray] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []
    action_match_residuals: list[float] = []

    import math as _math
    for pidx, (pid, eps) in enumerate(sorted(episodes.items())):
        _require(len(eps) == 2, f"ActionStrength pair {pid} does not have two modes")
        by_mode = {str(modes_e[ep, 0]): ep for ep in eps}
        _require(set(by_mode) == {"low_gain", "high_gain"},
                 f"ActionStrength pair {pid} has unexpected modes {sorted(by_mode)}")
        e0, e1 = by_mode["low_gain"], by_mode["high_gain"]
        # The complete action sequence is an exact matched invariant.
        act_resid = float(np.max(np.abs(
            actions_e[e0] - actions_e[e1]
        )))
        action_match_residuals.append(act_resid)
        _require(act_resid < 1e-5, f"ActionStrength query actions differ at pair {pid}: {act_resid}")
        _require(np.allclose(physics_e[e0, query_step], physics_e[e1, query_step], atol=1e-5),
                 f"ActionStrength query physics differ at pair {pid}")

        x = physics_e[e0, query_step, tslice[0]:tslice[1]]
        y0 = physics_e[e0, future_step, tslice[0]:tslice[1]]
        y1 = physics_e[e1, future_step, tslice[0]:tslice[1]]
        Y = np.stack([y0, y1])[None]
        C_val = float(weighted_conditional_variance(Y, x[None], np.array([0.5, 0.5]))[0])
        hist0 = np.stack([physics_e[e0, s, tslice[0]:tslice[1]] for s in hist_steps])
        hist1 = np.stack([physics_e[e1, s, tslice[0]:tslice[1]] for s in hist_steps])
        H_val = float(weighted_history_conditional_variance(
            np.stack([hist0, hist1])[None], np.array([0.5, 0.5])
        )[0])
        m_val = 0.5 * ((y0 - x) + (y1 - x))
        goal = goals_e[e0, query_step]
        goal_rel = goal[2:4] - x
        theta = float(physics_e[e0, query_step, 10])
        q_action = actions_e[e0, query_step:future_step].ravel()
        desc = np.concatenate([
            x, [physics_e[e0, query_step, 8], physics_e[e0, query_step, 9],
                _math.sin(theta), _math.cos(theta)], goal_rel, q_action
        ])
        C_list.append(C_val); H_list.append(H_val)
        m_list.append(m_val); desc_list.append(desc); cids.append(pidx)

    C = np.array(C_list); H = np.array(H_list)
    m_arr = np.stack(m_list)
    desc_arr = np.stack(desc_list)
    cid_arr = np.array(cids, dtype=np.int64)
    desc_scaled, _, _ = robust_scale(desc_arr)
    neighbors, _ = leave_cluster_out_knn(desc_scaled, cid_arr,
                                         max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk = background_variance_by_k(m_arr, neighbors, KS)
    w = np.full(len(C), 1.0 / len(C))
    rhos = {k: ratio_of_means(C, bvk[k], w) for k in KS}
    ci = bootstrap_ratio_of_means_ci(C, bvk[MAIN_K], cid_arr)
    receipt["train_lance_version"] = int(lance.dataset(str(train_lance)).version)
    receipt["action_match_residual_max"] = float(max(action_match_residuals)) if action_match_residuals else 0.0
    return {
        "task": "action_strength",
        "status": "ok",
        "n_queries": len(C),
        "C": _distribution_summary(C),
        "target_slice_history_condition_energy_physical": _distribution_summary(H),
        "coordinate_definition": "PushT block position physics_state[6:8], pixel units squared",
        "B": {str(k): _distribution_summary(bvk[k]) for k in KS},
        "rho": {str(k): float(rhos[k]) for k in KS},
        "rho_bootstrap_ci": ci,
        "evidence_type": "observed_matched_twin_binary",
        "condition_balance": {"n_conditions": 2, "weights": [0.5, 0.5]},
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# ActionDelay loader
# ---------------------------------------------------------------------------

def _delay_weights() -> tuple[list[int], list[float]]:
    """Build (delay_values, normalized_weights) per spec.

    6 equal native-exposure groups: [0],[1],[2],[3],[4],[5..10].
    d0..d4 each 1/6; d5..d10 each 1/36. Sums to 1.
    """
    delays = list(range(11))
    raw: list[float] = [1.0 / 6.0 if d <= 4 else 1.0 / 36.0 for d in delays]
    total = sum(raw)
    return delays, [w / total for w in raw]


def audit_action_delay(cfg: dict[str, Any], repo_root: Path, check_only: bool) -> dict[str, Any]:
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
        "check_only": check_only,
    }

    if check_only:
        for delay_rows in by_pair.values():
            for row in delay_rows.values():
                _row_path(row)
        return {"task": "action_delay", "status": "check_ok", "receipt": receipt}

    selected_steps = sorted(set(hist_steps + list(range(query_step, future_step + 1))))

    def _load_selected(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dataset = lance.dataset(str(path))
        names = set(dataset.schema.names)
        _require({"episode_idx", "step_idx", "proprio", "action"} <= names,
                 f"ActionDelay schema mismatch: {path}")
        predicate = "step_idx IN (" + ",".join(str(v) for v in selected_steps) + ")"
        table = dataset.to_table(
            columns=["episode_idx", "step_idx", "proprio", "action"],
            filter=predicate,
        )
        episode = _scalar_col(table, "episode_idx", np.int64)
        step = _scalar_col(table, "step_idx", np.int64)
        position = _fixed_list_col(table, "proprio", 2)
        action = _fixed_list_col(table, "action", 2)
        order = np.lexsort((step, episode))
        episode, step = episode[order], step[order]
        position, action = position[order], action[order]
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
            unique_episodes,
        )

    C_list: list[float] = []
    H_list: list[float] = []
    m_list: list[np.ndarray] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []
    action_match_max = 0.0
    w_arr = np.asarray(delay_weights_list, dtype=np.float64)
    step_to_col = {step: index for index, step in enumerate(selected_steps)}
    history_cols = [step_to_col[step] for step in hist_steps]
    query_col = step_to_col[query_step]
    future_col = step_to_col[future_step]
    action_cols = [step_to_col[step] for step in range(query_step, future_step)]

    for pair_index, (pid, delay_rows) in enumerate(sorted(by_pair.items())):
        positions_by_delay: list[np.ndarray] = []
        actions_by_delay: list[np.ndarray] = []
        episode_ids: np.ndarray | None = None
        for delay in delay_values:
            position, action, current_episode_ids = _load_selected(_row_path(delay_rows[delay]))
            if episode_ids is None:
                episode_ids = current_episode_ids
            else:
                _require(np.array_equal(episode_ids, current_episode_ids),
                         f"episode identity differs in {pid}/d{delay}")
            positions_by_delay.append(position)
            actions_by_delay.append(action)
        assert episode_ids is not None
        positions = np.stack(positions_by_delay, axis=1)  # (E,C,S,2)
        actions = np.stack(actions_by_delay, axis=1)
        ref_actions = actions[:, 0]
        action_resid = float(np.max(np.abs(actions - ref_actions[:, None])))
        action_match_max = max(action_match_max, action_resid)
        _require(action_resid <= 1.0e-6, f"actions differ across delays in {pid}")
        query_positions = positions[:, :, query_col, tslice[0]:tslice[1]]
        query_resid = float(np.max(np.abs(query_positions - query_positions[:, :1])))
        _require(query_resid <= 1.0e-6, f"query differs across delays in {pid}")

        for ep_offset, _episode_id in enumerate(episode_ids):
            x = query_positions[ep_offset, 0]
            Y = positions[ep_offset, :, future_col, tslice[0]:tslice[1]][None]
            C_val = float(weighted_conditional_variance(Y, x[None], w_arr)[0])
            histories = positions[ep_offset][:, history_cols, tslice[0]:tslice[1]][None]
            H_val = float(weighted_history_conditional_variance(histories, w_arr)[0])
            diffs = Y[0] - x[None, :]
            m_val = np.einsum("c,cd->d", w_arr, diffs)
            q_action = ref_actions[ep_offset, action_cols].ravel()
            desc = np.concatenate([x, q_action])
            C_list.append(C_val)
            H_list.append(H_val)
            m_list.append(m_val)
            desc_list.append(desc)
            cids.append(pair_index * len(episode_ids) + ep_offset)

    _require(len(C_list) > max(KS) + 10, f"too few ActionDelay clusters: {len(C_list)}")
    C = np.array(C_list)
    H = np.array(H_list)
    m_arr2 = np.stack(m_list)
    desc_arr2 = np.stack(desc_list)
    cid_arr2 = np.array(cids, dtype=np.int64)
    desc_scaled2, _, _ = robust_scale(desc_arr2)
    neighbors2, _ = leave_cluster_out_knn(desc_scaled2, cid_arr2,
                                          max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk2 = background_variance_by_k(m_arr2, neighbors2, KS)
    w2 = np.full(len(C), 1.0 / len(C))
    rhos2 = {k: ratio_of_means(C, bvk2[k], w2) for k in KS}
    ci2 = bootstrap_ratio_of_means_ci(C, bvk2[MAIN_K], cid_arr2)
    receipt["action_match_residual_max"] = float(action_match_max)
    return {
        "task": "action_delay",
        "status": "ok",
        "n_queries": len(C),
        "C": _distribution_summary(C),
        "target_slice_history_condition_energy_physical": _distribution_summary(H),
        "coordinate_definition": "TwoRoom agent position proprio[0:2], pixel units squared",
        "B": {str(k): _distribution_summary(bvk2[k]) for k in KS},
        "rho": {str(k): float(rhos2[k]) for k in KS},
        "rho_bootstrap_ci": ci2,
        "evidence_type": "observed_matched_condition_group",
        "condition_balance": {
            "n_conditions": len(delay_values),
            "weights": {str(d): float(w) for d, w in zip(delay_values, delay_weights_list)},
        },
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Speed loader  (simulator counterfactual replay — NOT observed matched twin)
# ---------------------------------------------------------------------------

def _tworoom_collision(
    pos: np.ndarray,
    proposed: np.ndarray,
    *,
    image_size: float = 224.0,
    border_size: float = 14.0,
    agent_radius: float = 7.0,
    wall_center: float = 112.0,
    wall_thickness: int = 10,
    door_center: float = 49.0,
    door_half_size: float = 14.0,
    door_margin: float = 1.75,
) -> np.ndarray:
    """Pure NumPy copy of the frozen vertical-wall TwoRoom collision rule."""
    current = np.asarray(pos, dtype=np.float32)
    candidate = np.asarray(proposed, dtype=np.float32).copy()
    lower = np.float32(border_size + agent_radius)
    upper = np.float32(image_size - border_size - agent_radius)
    candidate = np.clip(candidate, lower, upper).astype(np.float32)
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


def _manifest_speed(row: dict[str, Any]) -> float:
    value = row.get("factors", {}).get("agent.speed")
    _require(value is not None, f"Speed manifest row lacks factors.agent.speed: {row.get('scenario_id')}")
    return float(value)


def _speed_scene_path(data_root: Path, row: dict[str, Any]) -> Path:
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


def audit_speed(
    cfg: dict[str, Any],
    repo_root: Path,
    check_only: bool,
    speed_sample_limit: int | None = None,
) -> dict[str, Any]:
    manifest_path = _resolve(repo_root, cfg["manifest"])
    data_root = Path(cfg["data_root"]).resolve()
    training_only_guard(manifest_path)
    training_only_guard(data_root)
    _require(manifest_path.exists(), f"Speed manifest missing: {manifest_path}")
    _require(data_root.exists(), f"Speed data root missing: {data_root}")

    n_speeds = int(cfg.get("n_speeds", 32))
    num_steps = int(cfg.get("num_steps", 4))
    frameskip = int(cfg.get("frameskip", 5))
    _require(num_steps == 4 and frameskip == 5,
             "Speed v1 is frozen to num_steps=4 and frameskip=5")
    tslice = cfg.get("target_slice", [0, 2])
    total_sample = speed_sample_limit if speed_sample_limit is not None else int(
        cfg.get("total", 4096)
    )
    sample_seed = int(cfg.get("seed", BOOTSTRAP_SEED))
    geometry = dict(cfg.get("geometry", {}))

    # Read manifest — only train scenarios
    scenario_rows: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                row = json.loads(line)
                if str(row.get("split", "")).lower() == "train":
                    scenario_rows.append(row)

    # Extract the 32 training speeds from manifest.
    speed_set: set[float] = set()
    for row in scenario_rows:
        speed_set.add(_manifest_speed(row))
    training_speeds = sorted(speed_set)
    _require(len(training_speeds) == n_speeds,
             f"Expected {n_speeds} training speeds, found {len(training_speeds)}")

    receipt: dict[str, Any] = {
        "manifest_sha256": file_sha256(manifest_path),
        "n_train_scenarios": len(scenario_rows),
        "n_training_speeds": len(training_speeds),
        "train_storage_digest": manifest_storage_digest(scenario_rows),
        "evidence_type": SPEED_EVIDENCE_TYPE,
        "check_only": check_only,
    }

    if check_only:
        for row in scenario_rows:
            _speed_scene_path(data_root, row)
        return {"task": "speed", "status": "check_ok", "receipt": receipt}

    _require(total_sample % n_speeds == 0,
             "total_sample must be divisible by n_speeds for balanced sampling")
    per_speed = total_sample // n_speeds

    import lance  # type: ignore

    # Enumerate valid episode-start queries first. Some episodes terminate before
    # raw step 15, so selecting episode ids before checking rows would bias or fail.
    selected_steps = [0, 5, 10, 11, 12, 13, 14, 15]
    step_array = np.asarray(selected_steps, dtype=np.int64)
    step_clause = ",".join(str(value) for value in selected_steps)
    candidates_by_speed: dict[float, list[dict[str, Any]]] = {
        speed: [] for speed in training_speeds
    }
    scenario_valid_counts: dict[str, int] = {}
    for scenario in sorted(scenario_rows, key=lambda item: str(item["scenario_id"])):
        scenario_id = str(scenario["scenario_id"])
        scene_path = _speed_scene_path(data_root, scenario)
        dataset = lance.dataset(str(scene_path))
        _require({"episode_idx", "step_idx", "proprio", "action", "goal_state"}
                 <= set(dataset.schema.names), f"Speed schema mismatch: {scene_path}")
        table = dataset.to_table(
            columns=["episode_idx", "step_idx", "proprio", "action", "goal_state"],
            filter=f"step_idx IN ({step_clause})",
        )
        episode = _scalar_col(table, "episode_idx", np.int64)
        step = _scalar_col(table, "step_idx", np.int64)
        position = _fixed_list_col(table, "proprio", 2)
        action = _fixed_list_col(table, "action", 2)
        goal = _fixed_list_col(table, "goal_state", 2)
        order = np.lexsort((step, episode))
        episode, step = episode[order], step[order]
        position, action, goal = position[order], action[order], goal[order]
        valid_episode_ids = [
            int(episode_id)
            for episode_id in np.unique(episode)
            if np.array_equal(step[episode == episode_id], step_array)
        ]
        scenario_valid_counts[scenario_id] = len(valid_episode_ids)
        _require(len(valid_episode_ids) >= 8,
                 f"Speed scenario has fewer than eight valid queries: {scenario_id}")
        ranked_episode_ids = sorted(
            valid_episode_ids,
            key=lambda episode_id: hashlib.sha256(
                f"{sample_seed}:{scenario_id}:{episode_id}".encode("utf-8")
            ).hexdigest(),
        )[:8]
        observed_speed = _manifest_speed(scenario)
        for episode_id in sorted(ranked_episode_ids):
            mask = episode == episode_id
            candidates_by_speed[observed_speed].append({
                "scenario_id": scenario_id,
                "episode_id": episode_id,
                "observed_speed": observed_speed,
                "position": position[mask].copy(),
                "action": action[mask].copy(),
                "goal": goal[mask].copy(),
            })

    rng = np.random.default_rng(sample_seed)
    sampled: list[dict[str, Any]] = []
    for speed in training_speeds:
        pool = candidates_by_speed[speed]
        _require(len(pool) >= per_speed,
                 f"speed {speed}: only {len(pool)} capped queries, need {per_speed}")
        chosen = np.sort(rng.choice(len(pool), size=per_speed, replace=False))
        sampled.extend(pool[int(index)] for index in chosen)
    receipt["scenario_valid_query_count_min"] = int(min(scenario_valid_counts.values()))
    receipt["scenario_valid_query_count_max"] = int(max(scenario_valid_counts.values()))

    # For each selected Training query, verify five-step replay at the observed
    # speed, then replay the same query action block under all 32 Training speeds.

    C_list: list[float] = []
    H_list: list[float] = []
    m_list: list[np.ndarray] = []
    desc_list: list[np.ndarray] = []
    cids: list[int] = []
    consistency_residuals: list[float] = []
    observed_speed_counts = {str(speed): 0 for speed in training_speeds}

    speed_weights = np.full(n_speeds, 1.0 / n_speeds, dtype=np.float64)

    cluster_id = 0
    for candidate in sampled:
        pos_sequence = candidate["position"]
        action_sequence = candidate["action"]
        goal_sequence = candidate["goal"]
        obs_speed = float(candidate["observed_speed"])
        x = pos_sequence[2, tslice[0]:tslice[1]]
        query_actions = action_sequence[2:7]
        cf_futures: list[np.ndarray] = []
        for speed in training_speeds:
            cf_pos = x.copy()
            for query_action in query_actions:
                proposed = np.asarray(cf_pos, dtype=np.float32) + (
                    np.clip(query_action, -1.0, 1.0).astype(np.float32)
                    * np.float32(speed)
                )
                cf_pos = _tworoom_collision(cf_pos, proposed, **geometry)
            cf_futures.append(cf_pos)

        obs_future = pos_sequence[7, tslice[0]:tslice[1]]
        replay_obs = cf_futures[training_speeds.index(obs_speed)]
        consistency_residuals.append(float(np.max(np.abs(replay_obs - obs_future))))
        observed_speed_counts[str(obs_speed)] += 1
        Y = np.stack(cf_futures)[None]
        C_val = float(weighted_conditional_variance(Y, x[None], speed_weights)[0])
        observed_history = np.stack([pos_sequence[0], pos_sequence[1]])[None]
        H_val = float(history_energy(observed_history, x[None])[0])
        diffs = Y[0] - x[None, :]
        m_val = np.einsum("c,cd->d", speed_weights, diffs)
        goal_relative = goal_sequence[2] - x
        desc = np.concatenate([x, goal_relative, query_actions.ravel()])
        C_list.append(C_val)
        H_list.append(H_val)
        m_list.append(m_val)
        desc_list.append(desc)
        cids.append(cluster_id)
        cluster_id += 1

    _require(len(C_list) > max(KS) + 10, f"too few Speed queries: {len(C_list)}")
    C = np.array(C_list)
    H = np.array(H_list)
    m_arr3 = np.stack(m_list)
    desc_arr3 = np.stack(desc_list)
    cid_arr3 = np.array(cids, dtype=np.int64)
    desc_scaled3, _, _ = robust_scale(desc_arr3)
    neighbors3, _ = leave_cluster_out_knn(desc_scaled3, cid_arr3,
                                          max_k=max(KS), chunk_size=NEIGHBOR_CHUNK_SIZE)
    bvk3 = background_variance_by_k(m_arr3, neighbors3, KS)
    w3 = np.full(len(C), 1.0 / len(C))
    rhos3 = {k: ratio_of_means(C, bvk3[k], w3) for k in KS}
    ci3 = bootstrap_ratio_of_means_ci(C, bvk3[MAIN_K], cid_arr3)
    residual_array = np.asarray(consistency_residuals, dtype=np.float64)
    _require(len(residual_array) == total_sample,
             f"Speed computed {len(residual_array)} queries, expected {total_sample}")
    _require(float(np.max(residual_array)) <= 1.0e-3,
             f"Speed replay residual too large: {float(np.max(residual_array))}")
    receipt["replay_residual_max"] = float(np.max(residual_array))
    receipt["replay_residual_p99"] = float(np.percentile(residual_array, 99))
    receipt["n_queries_computed"] = len(C)
    receipt["observed_speed_query_counts"] = observed_speed_counts
    return {
        "task": "speed",
        "status": "ok",
        "n_queries": len(C),
        "C": _distribution_summary(C),
        "visible_history_motion_energy_target_slice": _distribution_summary(H),
        "target_slice_history_condition_energy_physical": {
            "status": "not_estimable_from_observed_training_twins",
            "reason": "Speed Training has no same-query cross-speed observed twins",
        },
        "coordinate_definition": "TwoRoom agent position proprio[0:2], pixel units squared",
        "B": {str(k): _distribution_summary(bvk3[k]) for k in KS},
        "rho": {str(k): float(rhos3[k]) for k in KS},
        "rho_bootstrap_ci": ci3,
        "evidence_type": SPEED_EVIDENCE_TYPE,
        "condition_balance": {
            "n_conditions": n_speeds,
            "weights": "uniform_1_over_n_speeds",
        },
        "receipt": receipt,
    }


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-task Training-only conditional visibility CPU audit (v1)"
    )
    parser.add_argument("--config", required=True, type=Path,
                        help="Path to icl_training_conditional_visibility_v1.yaml")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory (must not exist)")
    parser.add_argument("--check-only", action="store_true",
                        help="Schema/manifest/path check only; skip KNN/bootstrap/replay")
    parser.add_argument("--speed-sample-limit", type=int, default=None,
                        help="Override total speed sample count (for fast testing)")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    training_only_guard(config_path)
    _require(config_path.exists(), f"Config not found: {config_path}")

    cfg = load_config(config_path)
    repo_root = Path(__file__).resolve().parents[3]

    output_dir = Path(args.output_dir).resolve()
    exclusive_mkdir(output_dir)

    audit_id = str(uuid.uuid4())
    started_at = datetime.datetime.utcnow().isoformat() + "Z"

    TASK_AUDITORS = {
        "motion": audit_motion,
        "action_strength": audit_action_strength,
        "action_delay": audit_action_delay,
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
            result = auditor_fn(task_cfg, repo_root, args.check_only)
            per_task_results.append(result)
        except Exception as exc:
            per_task_results.append({"task": task_name, "status": "error", "error": str(exc)})
            errors.append(f"{task_name}: {exc}")

    # Speed task gets the sample-limit override
    speed_cfg = tasks_cfg.get("speed", {})
    if speed_cfg.get("enabled", True):
        speed_sample_cfg = cfg.get("speed_sample", {})
        effective_limit = args.speed_sample_limit or speed_sample_cfg.get("total", 4096)
        try:
            merged_speed_cfg = {**speed_cfg, **speed_sample_cfg}
            result = audit_speed(merged_speed_cfg, repo_root, args.check_only, effective_limit)
            per_task_results.append(result)
        except Exception as exc:
            per_task_results.append({"task": "speed", "status": "error", "error": str(exc)})
            errors.append(f"speed: {exc}")

    if args.check_only:
        overall_status = "check_ok" if not errors else "check_error"
    else:
        overall_status = "ok" if not errors else "partial_error"

    summary = {
        "schema_version": 1,
        "audit_id": audit_id,
        "started_at": started_at,
        "completed_at": datetime.datetime.utcnow().isoformat() + "Z",
        "status": overall_status,
        "check_only": args.check_only,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "errors": errors,
        "task_statuses": {r["task"]: r["status"] for r in per_task_results},
        "cross_task_comparability": (
            "descriptive_only_no_universal_absolute_threshold; task coordinates, "
            "condition cardinalities, and Speed evidence type differ"
        ),
        "evidence_boundary": {
            "pixels_decoded": PIXELS_DECODED,
            "model_loaded": MODEL_LOADED,
            "optimizer_steps": OPTIMIZER_STEPS,
            "pixel_target_separation_status": PIXEL_TARGET_SEPARATION_STATUS,
            "history_inferability_status": HISTORY_INFERABILITY_STATUS,
            "latent_status": LATENT_STATUS,
            "binding_cause_status": BINDING_CAUSE_STATUS,
            "claim_scope": CLAIM_SCOPE,
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
        "pixels_decoded": PIXELS_DECODED,
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


if __name__ == "__main__":
    main()
