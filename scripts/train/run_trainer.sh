#!/usr/bin/env bash

# Cloud-friendly LeWM training entry for stable-worldmodel.
#
# This script intentionally launches training only. Evaluation and diagnostics
# should stay as separate steps until the stable-worldmodel migration is
# validated.
#
# Common env vars:
#   dataset_name          multitask_3 | tworoom | pusht | reacher | cube
#   output_model_name     checkpoint/run name
#   config                Hydra config name, default lewm
#   trainer_file          training script, default scripts/train/lewm.py
#   STABLEWM_HOME         stable-worldmodel cache/checkpoint root
#   LOCAL_DATASET_DIR     optional dataset cache root
#   dataset_path          optional single-dataset path override
#
# Optional overrides:
#   seed max_epochs batch_size num_workers train_split frameskip
#   history_size num_preds embed_dim loss_sigreg_weight
#   trainer_devices trainer_accelerator trainer_precision
#   hydra_run_dir hydra_job_chdir trainer_default_root_dir trainer_fast_dev_run
#   wandb_enabled wandb_project wandb_entity

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}" || exit 1

export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-swm}}"

add_override() {
    local key="$1"
    local value="${2:-}"
    if [ -n "${value}" ]; then
        CMD_ARGS+=("${key}=${value}")
    fi
}

require_env() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "[train][error] ${name} is required" >&2
        exit 2
    fi
}

resolve_data_group() {
    case "$1" in
        multitask_3|mt3)
            echo "multitask_3"
            ;;
        tworoom)
            echo "tworoom"
            ;;
        pusht)
            echo "pusht"
            ;;
        reacher)
            echo "dmc"
            ;;
        cube)
            echo "ogb"
            ;;
        *)
            echo "[train][error] unknown dataset_name '$1'" >&2
            exit 2
            ;;
    esac
}

dataset_name="${dataset_name:-multitask_3}"
data_group="$(resolve_data_group "${dataset_name}")"

config_name="${config:-lewm}"
config_name="${config_name##*/}"
config_name="${config_name%.yaml}"
config_name="${config_name%.yml}"

trainer_file="${trainer_file:-scripts/train/lewm.py}"
output_model_name="${output_model_name:-${dataset_name}_lewm}"
run_name="${run_name:-${output_model_name}}"
subdir="${subdir:-${run_name}}"

require_env STABLEWM_HOME

CMD_ARGS=()
add_override "data" "${data_group}"
add_override "data.dataset.name" "${dataset_path:-}"
add_override "seed" "${seed:-}"
add_override "output_model_name" "${run_name}"
add_override "subdir" "${subdir}"
add_override "hydra.run.dir" "${hydra_run_dir:-}"
add_override "hydra.job.chdir" "${hydra_job_chdir:-}"

add_override "trainer.max_epochs" "${max_epochs:-}"
add_override "trainer.devices" "${trainer_devices:-}"
add_override "trainer.accelerator" "${trainer_accelerator:-}"
add_override "trainer.precision" "${trainer_precision:-}"
add_override "++trainer.default_root_dir" "${trainer_default_root_dir:-}"
add_override "++trainer.fast_dev_run" "${trainer_fast_dev_run:-}"
add_override "++trainer.limit_train_batches" "${trainer_limit_train_batches:-}"
add_override "++trainer.limit_val_batches" "${trainer_limit_val_batches:-}"

add_override "loader.batch_size" "${batch_size:-}"
add_override "loader.num_workers" "${num_workers:-}"
add_override "loader.persistent_workers" "${persistent_workers:-}"
add_override "loader.prefetch_factor" "${prefetch_factor:-}"
add_override "loader.pin_memory" "${pin_memory:-}"

add_override "train_split" "${train_split:-}"
add_override "data.dataset.frameskip" "${frameskip:-}"
add_override "wm.history_size" "${history_size:-}"
add_override "wm.num_preds" "${num_preds:-}"
add_override "embed_dim" "${embed_dim:-}"
add_override "loss.sigreg.weight" "${loss_sigreg_weight:-}"

add_override "wandb.enabled" "${wandb_enabled:-}"
add_override "wandb.config.project" "${wandb_project:-}"
add_override "wandb.config.entity" "${wandb_entity:-}"
add_override "wandb.config.name" "${wandb_name:-${run_name}}"
add_override "wandb.config.id" "${wandb_id:-${subdir}}"

add_override "data.dataset.sampling" "${multitask_sampling:-}"
add_override "data.dataset.balance_val" "${multitask_balance_val:-}"
add_override "data.dataset.items.0.name" "${multitask_tworoom_name:-}"
add_override "data.dataset.items.1.name" "${multitask_pusht_name:-}"
add_override "data.dataset.items.2.name" "${multitask_reacher_name:-}"

if [ "${skip_train:-0}" = "1" ]; then
    echo "[train] skipped (skip_train=1)"
    exit 0
fi

echo "==================================================="
echo "[train] repo:        ${REPO_DIR}"
echo "[train] dataset:     ${dataset_name} (data=${data_group})"
echo "[train] config:      ${config_name}"
echo "[train] run_name:    ${run_name}"
echo "[train] subdir:      ${subdir}"
echo "[train] STABLEWM_HOME=${STABLEWM_HOME}"
if [ -n "${LOCAL_DATASET_DIR:-}" ]; then
    echo "[train] LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR}"
fi
echo "==================================================="

python "${trainer_file}" --config-name="${config_name}" "${CMD_ARGS[@]}"
status=$?
if [ ${status} -ne 0 ]; then
    echo "[train] failed with status ${status}" >&2
    exit ${status}
fi

echo "==================================================="
echo "[train] done"
echo "[train] checkpoint dir: ${STABLEWM_HOME%/}/checkpoints/${subdir}"
echo "==================================================="
