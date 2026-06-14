#!/usr/bin/env bash

# Train LeWM for 33 epochs, then evaluate weights_epoch_33.pt.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export max_epochs=33
export post_train_eval=1
export eval_epoch=33

export output_model_name="${output_model_name:-${dataset_name:-multitask_3}_lewm_e33}"
export run_name="${run_name:-${output_model_name}}"
export subdir="${subdir:-${run_name}}"

echo "==================================================="
echo "[epoch33] max_epochs=${max_epochs}"
echo "[epoch33] post_train_eval=${post_train_eval}"
echo "[epoch33] eval_epoch=${eval_epoch}"
echo "[epoch33] run_name=${run_name}"
echo "[epoch33] subdir=${subdir}"
echo "==================================================="

exec bash "${SCRIPT_DIR}/run_trainer.sh"
