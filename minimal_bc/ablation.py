# Build the ablation report (ablation.html) and figure (ablation.png) from finished runs.
# Each axis varies one factor around the chunking baseline; experiments without runs stay empty.
import os
import re
import glob
import json
import pickle
import time
import base64
import yaml
import numpy as np
import torch
from logger import plt, TT

BASE = 'exp003_chunk25'  # baseline for every axis: chunk 25, 10K episodes, 10K steps, depth 10 x 512

# (axis, planned value, experiment name; None = not created yet[, series]). Fill in names as configs are added.
# Entries with a series name are drawn as separate lines within the same axis.
PLAN = [
    ('chunk size', 1, 'exp001_single_step'),
    ('chunk size', 5, None),
    ('chunk size', 10, None),
    ('chunk size', 25, BASE),
    ('chunk size', 50, None),
    ('train steps', 10_000, BASE),
    ('train steps', 50_000, 'exp004_steps50k'),
    ('train steps', 100_000, 'exp005_steps100k'),
    ('train steps', 300_000, None),
    ('data scale (episodes)', 1_000, None),
    ('data scale (episodes)', 10_000, BASE),
    ('data scale (episodes)', 50_000, None),
    # around exp004 (50K steps): deeper plain MLPs
    ('depth', 10, 'exp004_steps50k'),
    ('depth', 15, 'exp010_depth15'),
    ('depth', 20, 'exp011_depth20'),
    ('depth', 40, 'exp012_depth40'),
    ('depth', 40, 'exp025_depth40_preln', 'pre-LN'),
    ('depth', 80, 'exp026_depth80_preln', 'pre-LN'),
    # around exp004 (50K steps): eval-only, execute only the first k actions of each 25-step chunk
    ('executed steps', 1, 'exp027_exec1'),
    ('executed steps', 5, 'exp006_exec5'),
    ('executed steps', 10, 'exp007_exec10'),
    ('executed steps', 20, 'exp008_exec20'),
    ('executed steps', 25, 'exp004_steps50k'),
    # around exp004 (50K steps): 0 = plain MLP, 1 = residual connections
    ('residual', 0, 'exp004_steps50k'),
    ('residual', 1, 'exp009_residual'),
    # width at depth 10: lr 5e-4 (tuned for width 512) vs. lr 1e-4 (the reference's lr for width 4096)
    ('width', 512, 'exp004_steps50k', 'lr 5e-4'),
    ('width', 4096, 'exp013_w4096', 'lr 5e-4'),
    ('width', 512, 'exp019_lr1e-4', 'lr 1e-4'),
    ('width', 4096, 'exp020_w4096_lr1e-4', 'lr 1e-4'),
    # width 4096: deeper plain vs. pre-LN residual MLPs
    ('depth @ width 4096', 10, 'exp013_w4096', 'plain, lr 5e-4'),
    ('depth @ width 4096', 15, 'exp014_w4096_d15', 'plain, lr 5e-4'),
    ('depth @ width 4096', 10, 'exp020_w4096_lr1e-4', 'plain, lr 1e-4'),
    ('depth @ width 4096', 20, 'exp022_w4096_d20_lr1e-4', 'plain, lr 1e-4'),
    ('depth @ width 4096', 10, 'exp023_w4096_d10_preln_lr1e-4', 'pre-LN, lr 1e-4'),
    ('depth @ width 4096', 20, 'exp024_w4096_d20_preln_lr1e-4', 'pre-LN, lr 1e-4'),
]
AXES = ['chunk size', 'train steps', 'data scale (episodes)', 'depth', 'executed steps', 'residual',
        'width', 'depth @ width 4096']
LOG_X = {'chunk size', 'train steps', 'data scale (episodes)', 'width'}
SERIES_STYLE = {'': dict(color='black', marker='o', linestyle='-'),
                'lr 5e-4': dict(color='0.6', marker='o', linestyle='-'),
                'lr 1e-4': dict(color='black', marker='o', linestyle='-'),
                'plain, lr 5e-4': dict(color='0.6', marker='o', linestyle='-'),
                'plain, lr 1e-4': dict(color='black', marker='o', linestyle='-'),
                'pre-LN, lr 1e-4': dict(color='black', marker='s', linestyle='--', markerfacecolor='white'),
                'pre-LN': dict(color='black', marker='s', linestyle='--', markerfacecolor='white')}


def collect(exp):
    # One dict per finished seed of an experiment (a seed counts once it has been evaluated).
    runs = []
    for d in sorted(glob.glob(f'runs/{exp}/seed*')):
        if not os.path.exists(f'{d}/eval.json'):
            continue
        cfg = yaml.safe_load(open(f'{d}/config.yaml'))
        # eval-only experiments reuse another experiment's training run (eval.ckpt_exp)
        src = f"runs/{cfg['eval'].get('ckpt_exp', exp)}/{os.path.basename(d)}"
        state = torch.load(f'{src}/ckpt.pt', map_location='cpu', weights_only=False)['model']
        meta = cfg['data']['train_path'].replace('.npz', '-metadata.pkl')
        train_losses = [float(v) for v in re.findall(r'train loss ([\d.]+)', open(f'{src}/train.log').read())]
        runs.append(dict(
            steps=cfg['train']['n_steps'],
            episodes=len(pickle.load(open(meta, 'rb'))['episodes']),
            depth=cfg['model']['depth'],
            width=cfg['model']['hidden_dim'],
            params=sum(v.numel() for v in state.values()),
            chunk=cfg['train'].get('chunk_size', cfg.get('chunk_size', 1)),  # older runs stored it at top level
            train_loss=np.mean(train_losses[-10:]),  # mean of the batch losses printed over the last 1000 steps
            valid_loss=json.load(open(f'{src}/train_summary.json'))['valid_loss'],
            success=json.load(open(f'{d}/eval.json'))['success_rate'],
        ))
    return runs


def stat(runs, key):
    v = np.array([r[key] for r in runs])
    return v.mean(), v.std()


def fmt(mean_std, spec):
    return f"{mean_std[0]:{spec}} <span class='std'>± {mean_std[1]:{spec}}</span>"


def write_html(rows, fig_paths, path):
    # Self-contained report: styled table plus the figure embedded as base64.
    head = ('<tr><th>value</th><th>experiment</th><th>seeds</th><th>train steps</th><th>episodes</th>'
            '<th>depth × width</th><th>params</th><th>chunk</th><th>train loss</th><th>valid loss</th>'
            '<th>success rate</th></tr>')
    body = []
    for axis in AXES:
        body.append(f"<tr class='group'><td colspan='11'>{axis}</td></tr>")
        for a, value, exp, series, runs in rows:
            if a != axis:
                continue
            cls = 'pending' if not runs else 'base' if exp == BASE else ''
            name = (exp or '—') + (' (baseline)' if exp == BASE else '')
            cells = [f'{value:,}' + (f' ({series})' if series else ''), f'<code>{name}</code>']
            if runs:
                r = runs[0]
                cells += [len(runs), f"{r['steps']:,}", f"{r['episodes']:,}", f"{r['depth']} × {r['width']}",
                          f"{r['params'] / 1e6:.2f}M", r['chunk'], fmt(stat(runs, 'train_loss'), '.4f'),
                          fmt(stat(runs, 'valid_loss'), '.4f'), fmt(stat(runs, 'success'), '.1%')]
            else:
                cells += ['pending'] + [''] * 8
            body.append(f"<tr class='{cls}'>" + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    imgs = ''.join(f'<img src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}" '
                   f'alt="Ablation plots: success rate, train loss and valid loss along each axis">' for p in fig_paths)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ablation Results</title>
<style>
  body {{ background: #fff; color: #000; font-family: "Latin Modern Mono", "CMU Typewriter Text", "Courier New", monospace;
         margin: 32px auto; max-width: 1400px; padding: 0 16px; }}
  h1 {{ font-size: 24px; margin-bottom: 4px; }}
  .meta {{ color: #555; margin-bottom: 16px; }}
  ul {{ color: #333; line-height: 1.6; }}
  .scroll {{ overflow-x: auto; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }}
  th, td {{ border: 1px solid #000; padding: 6px 10px; text-align: right; white-space: nowrap; }}
  th {{ background: #f2f2f2; }}
  td:nth-child(2) {{ text-align: left; }}
  tr.group td {{ background: #000; color: #fff; text-align: left; font-weight: bold; }}
  tr.base td {{ font-weight: bold; }}
  tr.pending td {{ color: #aaa; }}
  .std {{ color: #777; font-size: 12px; }}
  code {{ font-family: inherit; }}
  img {{ width: 100%; margin-top: 24px; border: 1px solid #000; }}
</style></head><body>
<h1>Ablation results</h1>
<div class="meta">Generated {time.strftime('%Y-%m-%d %H:%M')} by minimal_bc/ablation.py from runs/. Baseline for every axis: <b>{BASE}</b>.</div>
<ul>
  <li>Metrics are mean ± std over training seeds. Grey rows have not been run yet.</li>
  <li>Success rate: closed-loop, eval seeds 0–49 (the same 50 initial states for every run).</li>
  <li>Train loss: mean of the batch losses printed over the last 1000 steps.</li>
  <li>Train / valid loss: normalized actions averaged over the whole chunk, so they are <b>not comparable across chunk sizes</b>.</li>
</ul>
<div class="scroll"><table>{head}{''.join(body)}</table></div>
{imgs}
</body></html>
"""
    with open(path, 'w') as f:
        f.write(html)
    print(f"saved {path}")


def plot(rows, axes_names, path):
    # Rows: metrics; columns: ablation axes. White background, black frame and lines.
    metrics = [('success', 'Success rate'), ('train_loss', 'Train loss'), ('valid_loss', 'Valid loss')]
    fig, axes = plt.subplots(len(metrics), len(axes_names), figsize=(4 * len(axes_names), 10), facecolor='white')
    for j, axis in enumerate(axes_names):
        planned = sorted({v for a, v, _, _, _ in rows if a == axis})
        series_names = list(dict.fromkeys(sr for a, _, _, sr, _ in rows if a == axis))
        for i, (key, label) in enumerate(metrics):
            ax = axes[i, j]
            drawn = False
            for series in series_names:
                done = [(v, stat(runs, key)) for a, v, exp, sr, runs in rows if a == axis and sr == series and runs]
                if not done:
                    continue
                x, ys = zip(*sorted(done))
                ax.errorbar(x, [m for m, _ in ys], yerr=[s for _, s in ys], linewidth=1.5, capsize=3,
                            label=series or None, **SERIES_STYLE[series])
                drawn = True
            if drawn and len(series_names) > 1:
                ax.legend(frameon=False, prop={'family': TT, 'size': 9})
            if not drawn:
                ax.text(0.5, 0.5, 'no runs yet', transform=ax.transAxes, ha='center', va='center',
                        color='0.5', fontfamily=TT, fontsize=11)
            # fix the x range to the planned values, so pending positions are visible too
            if axis in LOG_X:
                ax.set_xscale('log')
                ax.set_xlim(min(planned) / 1.6, max(planned) * 1.6)
            else:
                ax.set_xlim(min(planned) - 2, max(planned) + 2)
            ax.set_xticks(planned)
            ax.set_xticklabels([f'{v:,}' for v in planned], fontsize=8)
            ax.minorticks_off()
            if i == 0:
                ax.set_title(axis, fontfamily=TT, fontsize=14)
            if i == len(metrics) - 1:
                ax.set_xlabel(axis, fontfamily=TT, fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontfamily=TT, fontsize=12)
            if key != 'success' and axis == 'chunk size':
                ax.text(0.98, 0.95, 'not comparable\nacross chunk sizes', transform=ax.transAxes,
                        ha='right', va='top', color='0.4', fontfamily=TT, fontsize=8)
            for spine in ax.spines.values():
                spine.set_color('black')
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor='white')
    print(f"saved {path}")


if __name__ == '__main__':
    rows = [(e[0], e[1], e[2], e[3] if len(e) > 3 else '', collect(e[2]) if e[2] else []) for e in PLAN]
    rows.sort(key=lambda r: (AXES.index(r[0]), r[3], r[1]))
    # two figures of four axes each, so the panels stay readable in the report
    figs = ['ablation.png', 'ablation_2.png']
    plot(rows, AXES[:4], figs[0])
    plot(rows, AXES[4:], figs[1])
    write_html(rows, figs, 'ablation.html')
