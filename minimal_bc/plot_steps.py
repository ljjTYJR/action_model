# Success rate, training loss and validation loss against training steps, one line per experiment
# (mean over seeds, shaded ± std when there are several seeds). Reads train.log and eval_steps.json.
# usage: python minimal_bc/plot_steps.py <out.png> "<legend title>" "<label>=<exp>" ["<label>=<exp>" ...]
import os
import re
import sys
import glob
import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from logger import TT

# one marker per experiment (circle, square, star, ...), so lines stay distinguishable without color
MARKERS = [('o', 6), ('s', 6), ('*', 10), ('^', 7), ('D', 6)]


def read_run(run_dir):
    # train loss is printed every print_every steps (single batches), valid loss every valid_every steps
    text = open(os.path.join(run_dir, 'train.log')).read()
    train = np.array([(int(s), float(v)) for s, v in re.findall(r'step\s+(\d+) \| train loss ([\d.]+)', text)])
    valid = dict((int(s), float(v)) for s, v in re.findall(r'step\s+(\d+) \| valid loss ([\d.]+)', text))
    success = {int(k): v for k, v in json.load(open(os.path.join(run_dir, 'eval_steps.json'))).items()}
    steps = sorted(success)
    valid[steps[-1]] = float(re.search(r'final valid loss ([\d.]+)', text).group(1))
    # training loss at a checkpoint step: mean over the preceding 10K steps; losses are not drawn at step 0,
    # where the untrained loss (~0.5) would flatten the rest of the curve
    train_at = [train[(train[:, 0] > s - 10_000) & (train[:, 0] <= s), 1].mean() if s > 0 else np.nan for s in steps]
    return steps, [100 * success[s] for s in steps], train_at, [valid.get(s, np.nan) if s > 0 else np.nan for s in steps]


def plot(out_path, legend_title, runs):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, facecolor='white')
    # loss titles follow the policy type: flow loss, MSE loss, or just "loss" when the runs are mixed
    policies = {yaml.safe_load(open(glob.glob(os.path.join('runs', exp, 'seed*', 'config.yaml'))[0]))['train'].get('policy', 'mse')
                for _, exp in runs}
    loss = {'flow': 'Flow Loss', 'mse': 'MSE Loss'}[policies.pop()] if len(policies) == 1 else 'Loss'
    titles = ['Success Rate', f'Training {loss}', f'Validation {loss}']
    # success in blue, losses in gray; lighter shade = first experiment
    blues = plt.cm.Blues(np.linspace(0.45, 0.8, len(runs)))
    grays = plt.cm.Greys(np.linspace(0.4, 0.75, len(runs)))
    for k, (label, exp) in enumerate(runs):
        per_seed = [read_run(d) for d in sorted(glob.glob(os.path.join('runs', exp, 'seed*')))
                    if os.path.exists(os.path.join(d, 'eval_steps.json'))]
        steps = np.array(per_seed[0][0])
        for i, ax in enumerate(axes):
            values = np.array([r[i + 1] for r in per_seed])
            mean, std = values.mean(0), values.std(0)  # NaN (no loss at step 0) is simply not drawn
            color = blues[k] if i == 0 else grays[k]
            marker, size = MARKERS[k]
            ax.plot(steps, mean, color=color, linewidth=2, marker=marker, markersize=size,
                    markerfacecolor='white', markeredgewidth=1.5)
            if len(per_seed) > 1:
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontfamily=TT, fontsize=16)
        ax.set_xlabel('Training Steps', fontfamily=TT, fontsize=13)
        ax.grid(True, linestyle='--', color='0.85')
        ax.set_xticks([0, steps[-1] / 2, steps[-1]])
        ax.xaxis.set_major_formatter(lambda x, _: '0' if x == 0 else f'{x / 1e6:g}M' if x >= 1e6 else f'{x / 1e3:g}K')
        for spine in ax.spines.values():
            spine.set_color('black')
    axes[0].set_ylim(-5, 105)
    for ax in axes[1:]:
        ax.set_ylim(bottom=0)
    fig.canvas.draw()  # create the tick labels before restyling them
    for ax in axes:
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontfamily(TT)
    handles = [plt.Line2D([], [], color=c, linewidth=2, marker=m, markersize=size, markerfacecolor='white', markeredgewidth=1.5)
               for c, (m, size) in zip(grays, MARKERS)]
    fig.legend(handles, [label for label, _ in runs], title=legend_title, loc='lower center', ncol=len(runs),
               frameon=False, prop={'family': TT, 'size': 13}, title_fontproperties={'family': TT, 'size': 13})
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=150, facecolor='white')
    print(f'saved {out_path}')


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2], [tuple(arg.split('=', 1)) for arg in sys.argv[3:]])
