# Success rate of every kept checkpoint (ckpt_<step>.pt) of a run, for success-vs-steps curves.
# Same protocol as eval.py / eval_flow.py (eval seeds, open-loop chunks), without videos or results.csv rows.
import os
import sys
import glob
import json
import yaml
import numpy as np
import ocbench
from checkpoint import load_checkpoint
from model import BCModel, BCFlowModel
import eval as mse_eval
import eval_flow

if __name__ == '__main__':
    # usage: python minimal_bc/eval_steps.py <config.yaml> [seed]   -> runs/<exp>/seed<N>/eval_steps.json
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else config['train']['seed']
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join('runs', exp_name, f'seed{seed}')
    eval_cfg = config['eval']
    device = config['train']['device']
    env = ocbench.make(eval_cfg['env_name'])
    flow = config['train'].get('policy') == 'flow'

    paths = sorted(glob.glob(os.path.join(run_dir, 'ckpt_*.pt')), key=lambda p: int(os.path.basename(p)[5:-3]))
    results = {}
    for path in paths:
        step = int(os.path.basename(path)[5:-3])  # ckpt_<step>.pt
        model, act_normalizer, ckpt_config = load_checkpoint(path, device, BCFlowModel if flow else BCModel)
        chunk_size = ckpt_config['train']['chunk_size']
        execute_steps = eval_cfg.get('execute_steps', chunk_size)
        successes = []
        for ep in range(eval_cfg['n_episodes']):
            if flow:
                pred = ckpt_config['train'].get('pred', 'v')
                result = eval_flow.run_episode(env, model, act_normalizer, chunk_size, execute_steps, eval_cfg['flow_steps'],
                                               pred, eval_cfg['seed'] + ep, device)
            else:
                result = mse_eval.run_episode(env, model, act_normalizer, chunk_size, execute_steps, eval_cfg['seed'] + ep, device)
            successes.append(result[0])
        results[step] = float(np.mean(successes))
        print(f"step {step:8d} | success rate {results[step]:.2f} over {len(successes)} episodes")
    env.close()
    with open(os.path.join(run_dir, 'eval_steps.json'), 'w') as f:
        json.dump(results, f, indent=2)
