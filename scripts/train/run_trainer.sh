#!/usr/bin/env bash

# Cloud-friendly LeWM training entry for stable-worldmodel.
#
# This script launches training and can optionally run a post-train eval sweep
# when post_train_eval=1.
#
# Common env vars:
#   dataset_name          multitask_3 | multitask_4 | tworoom | pusht | reacher | cube
#   output_model_name     checkpoint/run name
#   config                Hydra config name, default lewm
#   trainer_file          training script, default scripts/train/lewm.py
#   STABLEWM_HOME         stable-worldmodel cache/checkpoint root
#   LOCAL_DATASET_DIR     optional dataset cache root
#   dataset_path          optional single-dataset path override
#   dataset_root          optional LeWorldModel data dir containing lewm_*.lance;
#                         also supplies single-task dataset_path by default
#
# Optional overrides:
#   seed max_epochs batch_size num_workers train_split frameskip
#   history_size num_preds embed_dim loss_sigreg_weight
#   trainer_devices trainer_accelerator trainer_precision
#   hydra_run_dir hydra_job_chdir trainer_default_root_dir trainer_fast_dev_run
#   logger_backend swanlab_enabled swanlab_project swanlab_workspace
#   swanlab_experiment_name swanlab_logdir swanlab_mode
#   swanlab_collect_hardware swanlab_hardware_monitor swanlab_log_hyperparams
#   wandb_enabled wandb_project wandb_entity
#
# Optional post-train eval:
#   post_train_eval      1 to run eval_wm.py after training; default 0
#   eval_tasks           task configs to eval; default dataset_name, or
#                        "tworoom pusht reacher" for multitask_3, or
#                        "tworoom pusht reacher cube" for multitask_4
#   eval_num_eval        episodes per seed; default 50
#   eval_seeds           space-separated seed list; default "42 43 44"
#   eval_epoch           checkpoint epoch; default max_epochs
#   eval_dataset_name    override eval dataset path/name for every eval task
#   eval_pusht_dataset_name    pusht-specific eval dataset override
#   eval_tworoom_dataset_name  tworoom-specific eval dataset override
#   eval_reacher_dataset_name  reacher-specific eval dataset override
#   eval_cube_dataset_name     cube-specific eval dataset override
#   eval_mujoco_gl       MuJoCo backend for eval; default osmesa
#   eval_keep_videos     1 to keep eval videos; default 0

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

resolve_eval_tasks() {
    if [ -n "${eval_tasks:-}" ]; then
        echo "${eval_tasks}"
        return 0
    fi

    case "${dataset_name}" in
        multitask_3|mt3)
            echo "tworoom pusht reacher"
            ;;
        multitask_4|mt4)
            echo "tworoom pusht reacher cube"
            ;;
        *)
            echo "${dataset_name}"
            ;;
    esac
}

first_existing_path() {
    local candidate
    for candidate in "$@"; do
        if [ -e "${candidate}" ]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 1
}

resolve_eval_dataset_name() {
    local task="$1"
    local ag_data_root
    ag_data_root="$(cd "${REPO_DIR}/../.." && pwd)"
    local world_model_root="${ag_data_root}/data/world_model"
    local quentinll_root="${world_model_root}/quentinll"
    local lewm_lance_root="${world_model_root}/lance-format/LeWorldModel/data"
    local resolved

    if [ -n "${eval_dataset_name:-}" ]; then
        echo "${eval_dataset_name}"
        return 0
    fi

    case "${task}" in
        reacher)
            if [ -n "${eval_reacher_dataset_name:-}" ]; then
                echo "${eval_reacher_dataset_name}"
            elif resolved="$(first_existing_path \
                "${quentinll_root}/reacher.h5" \
                "${quentinll_root}/lewm-reacher/reacher.h5" \
                "${lewm_lance_root}/lewm_reacher.lance")"; then
                echo "${resolved}"
            else
                echo "reacher"
            fi
            ;;
        tworoom)
            if [ -n "${eval_tworoom_dataset_name:-}" ]; then
                echo "${eval_tworoom_dataset_name}"
            elif resolved="$(first_existing_path \
                "${quentinll_root}/tworoom.h5" \
                "${quentinll_root}/lewm-tworooms/tworoom.h5" \
                "${lewm_lance_root}/lewm_tworoom.lance")"; then
                echo "${resolved}"
            else
                echo "tworoom"
            fi
            ;;
        pusht)
            if [ -n "${eval_pusht_dataset_name:-}" ]; then
                echo "${eval_pusht_dataset_name}"
            elif resolved="$(first_existing_path \
                "${quentinll_root}/pusht_expert_train.h5" \
                "${quentinll_root}/lewm-pusht/pusht_expert_train.h5" \
                "${quentinll_root}/lewm-pusht/datasets/pusht_expert_train.h5" \
                "${lewm_lance_root}/lewm_pusht.lance")"; then
                echo "${resolved}"
            else
                echo "pusht_expert_train"
            fi
            ;;
        cube)
            if [ -n "${eval_cube_dataset_name:-}" ]; then
                echo "${eval_cube_dataset_name}"
            elif resolved="$(first_existing_path \
                "${quentinll_root}/ogbench/cube_single_expert.h5" \
                "${quentinll_root}/lewm-cube/ogbench/cube_single_expert.h5" \
                "${lewm_lance_root}/lewm_cube.lance")"; then
                echo "${resolved}"
            else
                echo "ogbench/cube_single_expert"
            fi
            ;;
        *)
            echo "${task}"
            ;;
    esac
}

append_eval_json() {
    local metrics_file="$1"
    local task="$2"
    local weights_path="$3"
    local eval_dataset="$4"

    python - "${metrics_file}" "${task}" "${weights_path}" "${eval_dataset}" <<'PY_EVAL_JSON'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
task = sys.argv[2]
weights_path = sys.argv[3]
eval_dataset = sys.argv[4]
text = path.read_text()

success_match = re.search(r"'success_rate':\s*([0-9.]+)", text)
time_match = re.search(r"evaluation_time:\s*([0-9.eE+-]+)\s+seconds", text)
episode_match = re.search(r"'episode_successes':\s*array\(\[(.*?)\]\)", text, re.S)
seeds_match = re.search(r"'seeds':\s*(None|\[[^\]]*\])", text, re.S)

metrics = {}
if success_match:
    metrics["success_rate"] = float(success_match.group(1))
if episode_match:
    metrics["episode_successes"] = [
        token == "True"
        for token in re.findall(r"\bTrue\b|\bFalse\b", episode_match.group(1))
    ]
if seeds_match:
    metrics["seeds"] = None if seeds_match.group(1) == "None" else []

payload = {
    "evaluation_time": float(time_match.group(1)) if time_match else None,
    "eval_dataset_path": eval_dataset,
    "metrics": metrics,
    "task": task,
    "weights_path": weights_path,
}

with path.open("a") as f:
    f.write("==== JSON ====\n")
    f.write(json.dumps(payload, sort_keys=True))
    f.write("\n")
PY_EVAL_JSON
}

run_post_train_eval() {
    if [ "${post_train_eval:-0}" != "1" ]; then
        echo "[eval] skipped (post_train_eval=${post_train_eval:-0})"
        return 0
    fi

    local eval_epoch_value="${eval_epoch:-${max_epochs:-}}"
    if [ -z "${eval_epoch_value}" ]; then
        echo "[eval][error] eval_epoch or max_epochs is required for post_train_eval=1" >&2
        return 2
    fi

    local ckpt_dir="${STABLEWM_HOME%/}/checkpoints/${subdir}"
    local weights_rel="${subdir}/weights_epoch_${eval_epoch_value}.pt"
    local weights_path="${ckpt_dir}/weights_epoch_${eval_epoch_value}.pt"
    if [ ! -f "${weights_path}" ]; then
        echo "[eval][error] missing checkpoint: ${weights_path}" >&2
        return 2
    fi

    local results_dir="${ckpt_dir}/eval_results"
    mkdir -p "${results_dir}"

    local eval_num="${eval_num_eval:-50}"
    local seeds="${eval_seeds:-42 43 44}"
    local history_len="${eval_history_len:-${history_size:-3}}"
    local mujoco_gl="${eval_mujoco_gl:-osmesa}"
    local corruption_type="${eval_corruption_type:-gaussian_noise}"
    local corruption_std="${eval_corruption_std:-0.0}"
    local corruption_factor="${eval_corruption_factor:-1.0}"
    local corruption_kernel_size="${eval_corruption_kernel_size:-1}"
    local corruption_apply_to="${eval_corruption_apply_to:-pixels}"

    echo "==================================================="
    echo "[eval] post-train eval enabled"
    echo "[eval] checkpoint: ${weights_path}"
    echo "[eval] tasks:      $(resolve_eval_tasks)"
    echo "[eval] seeds:      ${seeds}"
    echo "[eval] num_eval:   ${eval_num}"
    echo "==================================================="

    local task seed eval_dataset label log_path metrics_path raw_filename raw_path
    local before_videos video_path
    for task in $(resolve_eval_tasks); do
        eval_dataset="$(resolve_eval_dataset_name "${task}")"
        for seed in ${seeds}; do
            label="${task}_epoch${eval_epoch_value}_num${eval_num}_seed${seed}"
            log_path="${results_dir}/${label}.log"
            metrics_path="${results_dir}/${label}_metrics.txt"
            raw_filename="${label}_raw.txt"
            raw_path="${ckpt_dir}/${raw_filename}"

            if [ -f "${log_path}" ] || [ -f "${metrics_path}" ]; then
                echo "[eval][error] refusing to overwrite existing ${label} outputs" >&2
                return 2
            fi
            if [ -f "${raw_path}" ]; then
                echo "[eval][error] refusing to overwrite existing ${raw_path}" >&2
                return 2
            fi

            before_videos="$(mktemp)"
            find "${ckpt_dir}" -maxdepth 1 -type f -name '*.mp4' -print > "${before_videos}"

            echo "[eval] running ${label}"
            MUJOCO_GL="${mujoco_gl}" python scripts/plan/eval_wm.py \
                --config-name="${task}" \
                "policy=${weights_rel}" \
                "eval.dataset_name=${eval_dataset}" \
                "eval.num_eval=${eval_num}" \
                "seed=${seed}" \
                "output.filename=${raw_filename}" \
                "++plan_config.history_len=${history_len}" \
                "++eval.corruption.type=${corruption_type}" \
                "++eval.corruption.kernel_size=${corruption_kernel_size}" \
                "++eval.corruption.factor=${corruption_factor}" \
                "++eval.corruption.std=${corruption_std}" \
                "++eval.corruption.apply_to=[${corruption_apply_to}]" \
                > "${log_path}" 2>&1
            status=$?
            if [ ${status} -ne 0 ]; then
                rm -f "${before_videos}"
                echo "[eval][error] ${label} failed with status ${status}; see ${log_path}" >&2
                return ${status}
            fi
            if [ ! -f "${raw_path}" ]; then
                rm -f "${before_videos}"
                echo "[eval][error] ${label} did not write ${raw_path}" >&2
                return 2
            fi

            mv "${raw_path}" "${metrics_path}"
            append_eval_json "${metrics_path}" "${task}" "${weights_path}" "${eval_dataset}"

            if [ "${eval_keep_videos:-0}" != "1" ]; then
                while IFS= read -r video_path; do
                    if ! grep -Fxq "${video_path}" "${before_videos}"; then
                        rm -f "${video_path}"
                    fi
                done < <(find "${ckpt_dir}" -maxdepth 1 -type f -name '*.mp4' -print)
            fi
            rm -f "${before_videos}"
            echo "[eval] wrote ${metrics_path}"
        done
    done
}

resolve_data_group() {
    case "$1" in
        multitask_3|mt3)
            echo "multitask_3"
            ;;
        multitask_4|mt4)
            echo "multitask_4"
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
logger_backend="${logger_backend:-swanlab}"
if [ "${logger_backend}" = "swanlab" ] && [ -z "${swanlab_enabled:-}" ]; then
    swanlab_enabled=True
fi

require_env STABLEWM_HOME

if [ -n "${dataset_root:-}" ]; then
    dataset_root="${dataset_root%/}"
    multitask_tworoom_name="${multitask_tworoom_name:-${dataset_root}/lewm_tworoom.lance}"
    multitask_pusht_name="${multitask_pusht_name:-${dataset_root}/lewm_pusht.lance}"
    multitask_reacher_name="${multitask_reacher_name:-${dataset_root}/lewm_reacher.lance}"
    multitask_cube_name="${multitask_cube_name:-${dataset_root}/lewm_cube.lance}"
    case "${dataset_name}" in
        tworoom)
            dataset_path="${dataset_path:-${dataset_root}/lewm_tworoom.lance}"
            ;;
        pusht)
            dataset_path="${dataset_path:-${dataset_root}/lewm_pusht.lance}"
            ;;
        reacher)
            dataset_path="${dataset_path:-${dataset_root}/lewm_reacher.lance}"
            ;;
        cube)
            dataset_path="${dataset_path:-${dataset_root}/lewm_cube.lance}"
            ;;
    esac
fi

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

add_override "logger_backend" "${logger_backend:-}"
add_override "swanlab.enabled" "${swanlab_enabled:-}"
add_override "swanlab.config.project" "${swanlab_project:-}"
add_override "swanlab.config.workspace" "${swanlab_workspace:-}"
add_override "swanlab.config.experiment_name" "${swanlab_experiment_name:-${run_name}}"
add_override "swanlab.config.id" "${swanlab_id:-${subdir}}"
add_override "swanlab.config.logdir" "${swanlab_logdir:-}"
add_override "swanlab.config.mode" "${swanlab_mode:-}"
add_override "swanlab.collect_hardware" "${swanlab_collect_hardware:-}"
add_override "swanlab.hardware_monitor" "${swanlab_hardware_monitor:-}"
add_override "swanlab.log_hyperparams" "${swanlab_log_hyperparams:-}"

add_override "wandb.enabled" "${wandb_enabled:-}"
add_override "wandb.config.project" "${wandb_project:-}"
add_override "wandb.config.entity" "${wandb_entity:-}"
add_override "wandb.config.name" "${wandb_name:-${run_name}}"
add_override "wandb.config.id" "${wandb_id:-${subdir}}"

add_override "data.dataset.sampling" "${multitask_sampling:-}"
add_override "data.dataset.balance_val" "${multitask_balance_val:-}"
if [ "${data_group}" = "multitask_3" ] || [ "${data_group}" = "multitask_4" ]; then
    add_override "data.dataset.items.0.name" "${multitask_tworoom_name:-}"
    add_override "data.dataset.items.1.name" "${multitask_pusht_name:-}"
    add_override "data.dataset.items.2.name" "${multitask_reacher_name:-}"
fi
if [ "${data_group}" = "multitask_4" ]; then
    add_override "data.dataset.items.3.name" "${multitask_cube_name:-}"
fi

if [ "${skip_train:-0}" = "1" ]; then
    echo "[train] skipped (skip_train=1)"
    run_post_train_eval
    exit 0
fi

echo "==================================================="
echo "[train] repo:        ${REPO_DIR}"
echo "[train] dataset:     ${dataset_name} (data=${data_group})"
echo "[train] config:      ${config_name}"
echo "[train] run_name:    ${run_name}"
echo "[train] subdir:      ${subdir}"
echo "[train] logger:      ${logger_backend}"
echo "[train] STABLEWM_HOME=${STABLEWM_HOME}"
if [ -n "${dataset_path:-}" ]; then
    echo "[train] dataset_path=${dataset_path}"
fi
if [ -n "${LOCAL_DATASET_DIR:-}" ]; then
    echo "[train] LOCAL_DATASET_DIR=${LOCAL_DATASET_DIR}"
fi
echo "==================================================="

if [ "${logger_backend}" = "swanlab" ] && [ -n "${SWANLAB_API_KEY:-}" ]; then
    swanlab login -k "${SWANLAB_API_KEY}"
fi

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

run_post_train_eval
