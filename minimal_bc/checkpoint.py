# Save / load a trained model together with everything needed to use it.
import os
import yaml
import torch
from model import BCModel
from preprocess import ActionNormalizer


def save_checkpoint(run_dir, model, model_kwargs, act_normalizer, config, step, name='ckpt.pt'):
    # ckpt.pt holds the weights, the model arguments and the action normalizer statistics,
    # so evaluation can rebuild the exact same model and map its outputs back to env actions.
    os.makedirs(run_dir, exist_ok=True)
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'model_kwargs': model_kwargs,
        'act_center': act_normalizer.center,
        'act_half_range': act_normalizer.half_range,
        'act_clip': act_normalizer.clip,
        'config': config,
    }, os.path.join(run_dir, name))
    # A readable copy of the config next to the checkpoint.
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"saved checkpoint at step {step} to {os.path.join(run_dir, name)}")


def load_checkpoint(path, device='cuda', model_cls=BCModel):
    # model_cls: BCModel (MSE policy) or BCFlowModel (flow policy); both are built from the saved kwargs
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_cls(**ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    act_normalizer = ActionNormalizer(ckpt['act_center'], ckpt['act_half_range'], ckpt['act_clip'])
    return model, act_normalizer, ckpt['config']
