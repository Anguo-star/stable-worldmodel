#!/usr/bin/env bash

# Multi-node env-var sweep wrapper for scripts/train/run_trainer.sh.
#
# Any listed env var may contain either one value or NNODES comma-separated
# values. This node picks the value at NODE_RANK and then execs run_trainer.sh.
#
# Cloud launcher conventions:
#   MA_NUM_HOSTS    total node count
#   VC_TASK_INDEX   zero-based node rank

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NNODES="${MA_NUM_HOSTS:-1}"
NODE_RANK="${VC_TASK_INDEX:-0}"

echo "[batch] NNODES=${NNODES} NODE_RANK=${NODE_RANK}"

SWEEP_VARS=(
    dataset_name dataset_path dataset_root config trainer_file output_model_name run_name subdir
    seed max_epochs batch_size num_workers train_split frameskip
    history_size num_preds embed_dim loss_sigreg_weight
    trainer_devices trainer_accelerator trainer_precision
    trainer_default_root_dir trainer_fast_dev_run
    trainer_limit_train_batches trainer_limit_val_batches
    hydra_run_dir hydra_job_chdir
    persistent_workers prefetch_factor pin_memory
    wandb_enabled wandb_project wandb_entity wandb_name wandb_id
    multitask_sampling multitask_balance_val
    multitask_tworoom_name multitask_pusht_name multitask_reacher_name
    skip_train
)

split_values_for_var() {
    local raw_value="$1"
    local values=()
    IFS=',' read -ra values <<< "${raw_value}"
    printf '%s\n' "${values[@]}"
}

if ! [[ "${NNODES}" =~ ^[0-9]+$ ]] || [ "${NNODES}" -lt 1 ]; then
    echo "[batch][error] invalid NNODES=${NNODES}" >&2
    exit 2
fi
if ! [[ "${NODE_RANK}" =~ ^[0-9]+$ ]] || [ "${NODE_RANK}" -ge "${NNODES}" ]; then
    echo "[batch][error] invalid NODE_RANK=${NODE_RANK}" >&2
    exit 2
fi

declare -A var_values_csv
max_len=1
for v in "${SWEEP_VARS[@]}"; do
    raw="${!v-}"
    if [ -z "${raw}" ]; then
        continue
    fi
    var_values_csv[$v]="${raw}"
    mapfile -t arr < <(split_values_for_var "${raw}")
    n=${#arr[@]}
    if [ "${n}" -gt "${max_len}" ]; then
        max_len=${n}
    fi
done

echo "[batch] max sweep length detected: ${max_len}"

if [ "${NNODES}" -ne "${max_len}" ]; then
    echo "[batch][error] NNODES (${NNODES}) must equal max sweep length (${max_len})." >&2
    exit 2
fi

if [ "${max_len}" -gt 1 ]; then
    unique_key="${run_name:-${output_model_name:-}}"
    if [ -z "${unique_key}" ]; then
        echo "[batch][error] set output_model_name or run_name for sweeps" >&2
        exit 2
    fi
    name_csv="${run_name:-${output_model_name:-}}"
    mapfile -t name_arr < <(split_values_for_var "${name_csv}")
    if [ "${#name_arr[@]}" -ne "${NNODES}" ]; then
        echo "[batch][error] output_model_name/run_name must have ${NNODES} values" >&2
        exit 2
    fi
    declare -A seen_names=()
    for name in "${name_arr[@]}"; do
        if [ -n "${seen_names[$name]:-}" ]; then
            echo "[batch][error] duplicate run name '${name}'" >&2
            exit 2
        fi
        seen_names[$name]=1
    done
fi

for v in "${!var_values_csv[@]}"; do
    mapfile -t arr < <(split_values_for_var "${var_values_csv[$v]}")
    n=${#arr[@]}
    if [ "${n}" -ne 1 ] && [ "${n}" -ne "${NNODES}" ]; then
        echo "[batch][error] env var '${v}' has ${n} values; expected 1 or ${NNODES}" >&2
        echo "[batch][error] raw value: ${var_values_csv[$v]}" >&2
        exit 2
    fi
done

echo "[batch] resolved per-node overrides:"
for v in "${!var_values_csv[@]}"; do
    mapfile -t arr < <(split_values_for_var "${var_values_csv[$v]}")
    if [ "${#arr[@]}" -eq 1 ]; then
        picked="${arr[0]}"
    else
        picked="${arr[$NODE_RANK]}"
    fi
    export "${v}=${picked}"
    echo "  ${v}=${picked}"
done

echo "[batch] launching run_trainer.sh"
exec bash "${SCRIPT_DIR}/run_trainer.sh"
