#!/bin/bash
# Generate the larger OCBench datasets used for the blog-style experiments (runs on GPU via MJWarp).
set -e
cd "$(dirname "$0")"
PY=/root/venvs/ocbench/bin/python
GEN=ocbench/data_gen_scripts/generate_ocbench.py

gen() {  # env_name, num_episodes, output name
    echo "=== $(date '+%F %T') generating $3 ($2 episodes)"
    $PY $GEN --env_name=$1 --num_episodes=$2 --max_parallel_episodes=2500 --seed=0 \
        --save_path=data/$3.npz 2>&1 | grep -E "^(Train|Validation)|Error"
}

gen block-single-task1-v0 10000 block-single-task1-v0-10k
gen block-single-task1-v0 50000 block-single-task1-v0-50k
gen block-double-task1-v0 30000 block-double-task1-v0-30k
echo "=== $(date '+%F %T') done"
