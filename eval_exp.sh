#!/bin/bash
# Evaluate an eval-only experiment (its config sets eval.ckpt_exp) over several seeds, then print the results table.
# usage: ./eval_exp.sh minimal_bc/configs/exp006_exec5.yaml      (seeds 0 1 2)
set -eo pipefail
cd "$(dirname "$0")"
source /root/venvs/ocbench/bin/activate
CONFIG=$1
EXP=$(basename "$CONFIG" .yaml)
SUFFIX=$(grep -qE '^\s*policy:\s*flow' "$CONFIG" && echo _flow || true)  # flow configs use train_flow.py / eval_flow.py
for SEED in ${SEEDS:-0 1 2}; do
    mkdir -p runs/$EXP/seed$SEED
    python minimal_bc/eval$SUFFIX.py "$CONFIG" $SEED 2>&1 | grep -v -i glfw | tee runs/$EXP/seed$SEED/eval.log
done
python minimal_bc/report.py
