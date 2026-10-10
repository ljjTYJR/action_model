import os
import sys
import json
import shutil
import yaml
from data import read_data
from model import BCModel
from logger import Logger
from preprocess import ActionNormalizer
from checkpoint import save_checkpoint
import torch
from torch import nn

def get_chunk_idx(idx, chunk_size, ends):
    # given the idx and chunk_size and corresponding eposide ends
    # return chunkidx
    k = torch.searchsorted(ends, idx, right=True) # return which eposide inside
    chunk_idx = torch.minimum(idx[:, None] + torch.arange(chunk_size, device=device), (ends[k]-1)[:,None]) # (B,25), (B,)
    return chunk_idx

if __name__ == '__main__':
    # usage: python minimal_bc/train.py <config.yaml> [seed]
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.safe_load(f)
    train_cfg = config['train']
    if len(sys.argv) > 2:
        train_cfg['seed'] = int(sys.argv[2])
    exp_name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join('runs', exp_name, f"seed{train_cfg['seed']}")
    device = train_cfg['device']
    chunk_size = train_cfg['chunk_size']
    torch.manual_seed(train_cfg['seed'])

    # snapshot the code that produced this run, so results can always be traced back
    shutil.copytree('minimal_bc', os.path.join(run_dir, 'code'), dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__'))

    obs, acts, ends = read_data(config['data']['train_path'])
    valid_obs, valid_acts, valid_ends = read_data(config['data']['valid_path'])

    # normalize actions with statistics fitted on the training set only
    act_normalizer = ActionNormalizer.fit(acts)
    acts = act_normalizer.normalize(acts)
    valid_acts = act_normalizer.normalize(valid_acts)

    obs = torch.from_numpy(obs).to(device)
    acts = torch.from_numpy(acts).to(device)
    ends = torch.from_numpy(ends).to(device)
    valid_obs = torch.from_numpy(valid_obs).to(device)
    valid_acts = torch.from_numpy(valid_acts).to(device)
    valid_ends = torch.from_numpy(valid_ends).to(device)
    # get real-valid sets
    valid_idx = torch.arange(valid_acts.shape[0], device=device) # (B,):0-B
    valid_chunk_idx = get_chunk_idx(valid_idx, chunk_size, valid_ends)
    valid_acts = valid_acts[valid_chunk_idx].flatten(1) # (B, act*chunk)

    # model settings
    N, obs_dim = obs.shape
    act_dim = acts.shape[-1] * chunk_size # multiple actions predictions
    model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, **config['model'])
    bc_model = BCModel(**model_kwargs).to(device)

    # train loop
    optimizer = torch.optim.AdamW(
        bc_model.parameters(),
        lr = train_cfg['lr'],
    )
    bc_model.train()

    batch_size = train_cfg['batch_size']
    n_steps = train_cfg['n_steps']
    logger = Logger(print_every=train_cfg['print_every'])
    for step in range(n_steps):
        optimizer.zero_grad()
        idx = torch.randint(0, N, (batch_size,), device=device)
        obs_batch = obs[idx]
        chunk_idx = get_chunk_idx(idx, chunk_size, ends)
        gt_act = acts[chunk_idx] # (B, 25, 7)
        gt_act = gt_act.flatten(1)

        pred_act = bc_model(obs_batch)
        loss = ((pred_act - gt_act) ** 2).mean()
        loss.backward()
        optimizer.step()
        logger.log_train(step, loss)

        # validation
        if step % train_cfg['valid_every'] == 0 and step > 0:
            bc_model.eval()
            with torch.no_grad():
                valid_loss = ((bc_model(valid_obs) - valid_acts) ** 2).mean()
            bc_model.train()
            logger.log_valid(step, valid_loss)

        # periodic checkpoint, overwritten each time so the run dir always has the latest one
        if (step + 1) % train_cfg['save_every'] == 0 and step + 1 < n_steps:
            save_checkpoint(run_dir, bc_model, model_kwargs, act_normalizer, config, step + 1)

    save_checkpoint(run_dir, bc_model, model_kwargs, act_normalizer, config, n_steps)
    logger.plot(os.path.join(run_dir, 'loss.png'))

    # final validation loss for the results table
    bc_model.eval()
    with torch.no_grad():
        valid_loss = ((bc_model(valid_obs) - valid_acts) ** 2).mean().item()
    with open(os.path.join(run_dir, 'train_summary.json'), 'w') as f:
        json.dump(dict(valid_loss=valid_loss), f, indent=2)
    print(f"final valid loss {valid_loss:.6f}")
