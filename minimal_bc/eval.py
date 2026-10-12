# Closed-loop evaluation of a trained BC policy in the simulator, with rollout videos.
import os
import sys
import csv
import json
import time
import yaml
import imageio
import numpy as np
import torch
import ocbench
from checkpoint import load_checkpoint


def run_episode(env, model, act_normalizer, chunk_size, execute_steps, seed, device, render=False):
    obs, info = env.reset(seed=seed)
    frames = [env.render()] if render else []
    done, success, length = False, False, 0
    while not done:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs.astype(np.float32)).to(device)[None]
            pred = model(obs_t)[0].cpu().numpy()
        # normalized chunk prediction -> (chunk_size, 7) env actions in [-1, 1]
        actions = act_normalizer.unnormalize(pred.reshape(chunk_size, -1))
        for action in actions[:execute_steps]:  # execute the first steps of the chunk open-loop, then predict again
            obs, _, terminated, truncated, info = env.step(action)
            success = success or bool(info['success'])
            done = terminated or truncated
            length += 1
            if render:
                frames.append(env.render())
            if done:
                break
    return success, length, frames


if __name__ == '__main__':
    # usage: python minimal_bc/eval.py <config.yaml> [seed]   (same config and seed as training)
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else config['train']['seed']
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join('runs', exp_name, f'seed{seed}')
    eval_cfg = config['eval']
    device = config['train']['device']

    # eval-only experiments evaluate checkpoints trained by another experiment (eval.ckpt_exp)
    ckpt_dir = os.path.join('runs', eval_cfg.get('ckpt_exp', exp_name), f'seed{seed}')
    if ckpt_dir != run_dir:
        os.makedirs(run_dir, exist_ok=True)
        with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

    model, act_normalizer, ckpt_config = load_checkpoint(os.path.join(ckpt_dir, 'ckpt.pt'), device)
    chunk_size = ckpt_config['train']['chunk_size']  # use what the model was trained with
    execute_steps = eval_cfg.get('execute_steps', chunk_size)  # actions executed per chunk before predicting again
    env = ocbench.make(eval_cfg['env_name'], width=512, height=512)

    successes, lengths = [], []
    for ep in range(eval_cfg['n_episodes']):
        render = ep < eval_cfg['n_videos']
        success, length, frames = run_episode(env, model, act_normalizer, chunk_size, execute_steps, eval_cfg['seed'] + ep, device, render)
        successes.append(success)
        lengths.append(length)
        print(f"episode {ep:3d} | success {success} | length {length}")
        if render:
            path = os.path.join(run_dir, f"eval_ep{ep}_{'success' if success else 'fail'}.mp4")
            imageio.mimsave(path, frames, fps=50)
    env.close()

    result = dict(success_rate=float(np.mean(successes)), n_episodes=len(successes),
                  mean_length=float(np.mean(lengths)), successes=successes, lengths=lengths)
    with open(os.path.join(run_dir, 'eval.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"success rate {result['success_rate']:.2f} over {result['n_episodes']} episodes")

    # append one row per evaluated run to the shared results table
    with open(os.path.join(ckpt_dir, 'train_summary.json')) as f:
        summary = json.load(f)
    row = dict(time=time.strftime('%Y-%m-%d %H:%M'), exp=exp_name, seed=seed,
               n_steps=ckpt_config['train']['n_steps'], valid_loss=round(summary['valid_loss'], 6),
               success_rate=result['success_rate'], n_episodes=result['n_episodes'],
               mean_length=round(result['mean_length'], 1))
    results_path = os.path.join('runs', 'results.csv')
    new_file = not os.path.exists(results_path)
    with open(results_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)
