import os
import copy
import sys
import json
import shutil
import yaml
from data import read_data
from model import BCModel, BCFlowModel
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

def sample_t(n, dist):
    # 'uniform': t ~ U[0, 1]. 'beta' (as in pi0): t = 0.999 * (1 - u), u ~ Beta(1.5, 1),
    # which puts more weight on noisy (small) t, where the action depends most on the observation.
    if dist == 'beta':
        u = torch.distributions.Beta(torch.tensor(1.5, device=device), torch.tensor(1.0, device=device)).sample((n, 1))
        return 0.999 * (1 - u)
    return torch.rand(n, 1, device=device)

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
    N, obs_dim = obs.shape # obs_dim:28
    act_dim = acts.shape[-1] * chunk_size # multiple actions predictions
    model_kwargs = dict(obs_dim=obs_dim, act_dim=act_dim, **config['model'])
    bc_model = BCFlowModel(**model_kwargs).to(device)
    # Speed only, same method: tf32 = TF32 matmuls (10-bit mantissa); compile = CUDA graphs for the
    # training step (fixed batch shape). Validation, checkpoints and evaluation use the eager model.
    if train_cfg.get('tf32'):
        torch.set_float32_matmul_precision('high')
    train_model = torch.compile(bc_model, mode='reduce-overhead') if train_cfg.get('compile') else bc_model

    # train loop
    optimizer = torch.optim.AdamW(
        bc_model.parameters(),
        lr = train_cfg['lr'],
    )
    # lr_schedule: 'constant' (default) or 'cosine' (decay from lr to 0 over n_steps)
    scheduler = None
    if train_cfg.get('lr_schedule') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_cfg['n_steps'])
    bc_model.train()
    # ema_decay: validate and save an exponential moving average of the weights (None = the raw weights)
    ema_decay = train_cfg.get('ema_decay')
    eval_model = copy.deepcopy(bc_model) if ema_decay else bc_model
    ema_params, params = list(eval_model.parameters()), list(bc_model.parameters())

    batch_size = train_cfg['batch_size']
    n_steps = train_cfg['n_steps']
    logger = Logger(print_every=train_cfg['print_every'])

    # random seed
    g = torch.Generator(device=device).manual_seed(15)
    valid_x0 = torch.randn(valid_acts.shape, generator=g, device=device)
    valid_t = torch.rand((valid_acts.shape[0], 1), generator=g, device=device)
    valid_xt = valid_x0 + (valid_acts - valid_x0) * valid_t
    # pred: 'v' = the network predicts the velocity x1 - x0; 'x' = it predicts the clean chunk x1
    pred_type = train_cfg.get('pred', 'v')
    t_dist = train_cfg.get('t_dist', 'uniform')
    valid_target = valid_acts if pred_type == 'x' else valid_acts - valid_x0

    @torch.no_grad()
    def compute_valid_loss(model):
        # mean squared error over the whole valid set, in chunks to keep the activation memory small
        total = 0.0
        for o, x, t, y in zip(*(v.split(65536) for v in (valid_obs, valid_xt, valid_t, valid_target))):
            total += ((model(o, x, t) - y) ** 2).sum()
        return total / valid_target.numel()

    # keep_every: also keep ckpt_<step>.pt every keep_every steps (and at step 0), for success-vs-steps curves
    keep_every = train_cfg.get('keep_every')
    if keep_every:
        save_checkpoint(run_dir, eval_model, model_kwargs, act_normalizer, config, 0, name='ckpt_0.pt')
    for step in range(n_steps):
        optimizer.zero_grad()
        idx = torch.randint(0, N, (batch_size,), device=device)
        obs_batch = obs[idx] # (B,ob_dim)
        chunk_idx = get_chunk_idx(idx, chunk_size, ends)
        gt_act = acts[chunk_idx] # (B, 25, 7)
        gt_act = gt_act.flatten(1)

        # generate noise and t
        x0 = torch.randn_like(gt_act, device=device)
        t = sample_t(batch_size, t_dist)
        xt = x0 + (gt_act - x0) * t

        pred = train_model(obs_batch, xt, t)
        target = gt_act if pred_type == 'x' else gt_act - x0
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        if ema_decay:
            with torch.no_grad():
                torch._foreach_lerp_(ema_params, params, 1 - ema_decay)  # all tensors in one launch
        logger.log_train(step, loss)

        # validation
        if step % train_cfg['valid_every'] == 0 and step > 0:
            eval_model.eval()
            valid_loss = compute_valid_loss(eval_model)
            eval_model.train()
            logger.log_valid(step, valid_loss)

        # periodic checkpoint, overwritten each time so the run dir always has the latest one
        if (step + 1) % train_cfg['save_every'] == 0 and step + 1 < n_steps:
            save_checkpoint(run_dir, eval_model, model_kwargs, act_normalizer, config, step + 1)
        if keep_every and (step + 1) % keep_every == 0:
            save_checkpoint(run_dir, eval_model, model_kwargs, act_normalizer, config, step + 1, name=f'ckpt_{step + 1}.pt')

    save_checkpoint(run_dir, eval_model, model_kwargs, act_normalizer, config, n_steps)
    logger.plot(os.path.join(run_dir, 'loss.png'))

    # final validation loss for the results table
    eval_model.eval()
    valid_loss = compute_valid_loss(eval_model).item()
    with open(os.path.join(run_dir, 'train_summary.json'), 'w') as f:
        json.dump(dict(valid_loss=valid_loss), f, indent=2)
    print(f"final valid loss {valid_loss:.6f}")
