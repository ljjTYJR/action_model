#!/bin/bash
# Train and evaluate one experiment over several seeds, then print the results table.
# usage: ./run_exp.sh minimal_bc/configs/exp002_chunk25.yaml      (seeds 0 1 2)
#        SEEDS="0 1 2 3 4" ./run_exp.sh <config>                   (custom seeds)
set -eo pipefail
cd "$(dirname "$0")"
source /root/venvs/ocbench/bin/activate
CONFIG=$1
EXP=$(basename "$CONFIG" .yaml)
SUFFIX=$(grep -qE '^\s*policy:\s*flow' "$CONFIG" && echo _flow || true)  # flow configs use train_flow.py / eval_flow.py
for SEED in ${SEEDS:-0 1 2}; do
    mkdir -p runs/$EXP/seed$SEED
    python minimal_bc/train$SUFFIX.py "$CONFIG" $SEED 2>&1 | grep -v -i glfw | tee runs/$EXP/seed$SEED/train.log
    python minimal_bc/eval$SUFFIX.py "$CONFIG" $SEED 2>&1 | grep -v -i glfw | tee runs/$EXP/seed$SEED/eval.log
done
python minimal_bc/report.py
