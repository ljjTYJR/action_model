# Experiment log

## Protocol (fixed for every experiment)

- **Task:** OCBench `block-single-task1-v0`, state observations (28-d), training data `block-single-task1-v0-10k` (all episodes, no success filter).
- **Main metric:** closed-loop success rate on `block-cpu-single-task1-v0`, eval seeds 0–49, i.e. the same 50 initial states for every run.
- **Seeds:** each experiment runs with training seeds 0, 1, 2. Report mean ± std.
- **Secondary metric:** final valid loss (normalized actions). Only comparable between runs that predict the same thing; e.g. not between single-step and chunked policies.
- **Rule:** each experiment changes exactly one thing relative to the experiment it is compared with.

How to run and summarize:

```bash
./run_exp.sh minimal_bc/configs/<exp>.yaml
/root/venvs/ocbench/bin/python minimal_bc/report.py
```

Each run lives in `runs/<exp>/seed<N>/`: `ckpt.pt`, `config.yaml`, `code/` (snapshot), `train.log`, `loss.png`, `train_summary.json`, `eval.log`, `eval.json`, rollout videos. Every evaluation appends a row to `runs/results.csv` (columns: time, exp, seed, n_steps, valid_loss, success_rate, n_episodes, mean_length).

## Entry template

```
## expNNN_name
- Date:
- Compared with:
- Change: (the one thing that differs)
- Hypothesis: (what you expect and why, written BEFORE running)
- Results: (paste the report.py line)
- Observations: (videos, curves, anything surprising)
- Conclusion / next step:
```

---

## exp001_single_step
- **Date:** 2026-10-10
- **Compared with:** none (baseline)
- **Change:** none. Single-step policy: MLP 10 × 512 + LayerNorm, percentile action normalization, 10K steps, batch 1024, lr 5e-4.
- **Hypothesis:** near-zero success. The blog reports that a single-step closed-loop policy fails even with infinite data, and an earlier single run (`runs/bc_depth10`) got 2%.
- **Results:** success 0.67% ± 0.94% (seeds: 2%, 0%, 0%); valid loss 0.0184 ± 0.0040.
- **Observations:** the arm reaches toward the cube, then drifts into poses the expert never takes (compounding error). Seed 2 has a much higher valid loss (0.0241 vs 0.0156) yet the same success, so valid loss says little about success here.
- **Conclusion / next step:** hypothesis confirmed. Low offline loss does not give closed-loop success. Next: action chunking.

## exp002_chunk25
- **Date:** 2026-10-10
- **Compared with:** exp001_single_step
- **Change:** `chunk_size: 1 → 25`. The model predicts the next 25 actions and executes them open-loop before predicting again.
- **Note:** this was a reference implementation written by Claude; its code was removed afterwards so that chunking can be re-implemented by hand. The exact code is preserved in `runs/exp002_chunk25/seed*/code/`. Chunks past an episode's end repeat the episode's last action (as in the OCBench reference).
- **Hypothesis:** clearly higher success than exp001. A chunk commits to one smooth trajectory segment, which matches how the scripted expert generates data (hidden per-episode speed and detour choices), and it reduces how often small errors feed back into the policy. Valid loss is not comparable to exp001 (predicting 25 steps ahead is harder). With only 10K steps and a 2.4M-parameter model, success may still be far from the expert's 84%.
- **Results:** success 32.7% ± 26.2% (seeds: 66%, 2%, 30%); valid loss over the full chunk 0.0251 ± 0.0010; valid loss of the first action only 0.0155 ± 0.0013 (comparable to exp001's 0.0184).
- **Observations:** a large average gain, but extremely unstable across training seeds: from 66% down to 2%, with nearly identical valid losses. With 3 seeds × 50 episodes the mean is uncertain by roughly ±15 points.
- **Conclusion / next step:** hypothesis confirmed on average: chunking turns ~0% into ~33%. The seed variance is the main open problem: any later change needs more seeds (or a longer, more converged training) before small differences can be trusted. Next: re-implement chunking by hand as exp003 and check it reproduces ~33% with similar spread; then train longer.
