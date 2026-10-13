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
- **Hypothesis:** near-zero success. The blog reports that a single-step closed-loop policy fails even with infinite data, and an earlier single test run (since deleted) got 2%.
- **Results:** success 0.67% ± 0.94% (seeds: 2%, 0%, 0%); valid loss 0.0184 ± 0.0040.
- **Observations:** the arm reaches toward the cube, then drifts into poses the expert never takes (compounding error). Seed 2 has a much higher valid loss (0.0241 vs 0.0156) yet the same success, so valid loss says little about success here.
- **Conclusion / next step:** hypothesis confirmed. Low offline loss does not give closed-loop success. Next: action chunking.

## exp003_chunk25
- **Date:** 2026-10-10
- **Compared with:** exp001_single_step
- **Change:** `chunk_size: 1 → 25` (`train.chunk_size` in the config). The model predicts the next 25 actions and executes them open-loop before predicting again. Chunks past an episode's end repeat the episode's last action (as in the OCBench reference); the episode of each row is found with `torch.searchsorted`.
- **Hypothesis:** clearly higher success than exp001. A chunk commits to one smooth trajectory segment, which matches how the scripted expert generates data (hidden per-episode speed and detour choices), and it reduces how often small errors feed back into the policy. Valid loss is not comparable to exp001 (predicting 25 steps ahead is harder). With only 10K steps and a 2.4M-parameter model, success may still be far from the expert's 84%.
- **Results:** success 32.7% ± 26.2% (seeds: 66%, 2%, 30%); valid loss over the full chunk 0.0251 ± 0.0010.
- **Observations:** a large average gain, but extremely unstable across training seeds: from 66% down to 2%, with nearly identical valid losses. With 3 seeds × 50 episodes the mean is uncertain by roughly ±15 points. An independent implementation of the same idea produced bit-identical results (same valid loss to 16 digits for every seed), which validates the chunk targets.
- **Conclusion / next step:** hypothesis confirmed on average: chunking turns ~0% into ~33%. The seed variance is the main open problem: any later change needs more seeds (or a longer, more converged training) before small differences can be trusted. Next: train longer (n_steps 10K → 100K).

## exp004_steps50k
- **Date:** 2026-10-10
- **Compared with:** exp003_chunk25
- **Change:** `n_steps: 10K → 50K` (5× longer training, ~7 passes over the data instead of ~1.4).
- **Hypothesis:** higher mean success and smaller spread across seeds than exp003. At 10K steps the valid loss was still falling, so the model is undertrained; the blog also reports that longer training keeps helping even after the valid loss stops improving. Valid loss should go down; if it goes up while success rises, that is the blog's "overfitting is not necessarily bad" effect.
- **Results:** success 56.7% ± 3.4% (seeds: 52%, 60%, 58%); valid loss 0.0226 ± 0.0004.
- **Observations:** much higher and far more stable than exp003 (32.7% ± 26.2%). Valid loss reaches its minimum (~0.021) around 23K–32K steps and is slightly higher again at 50K.
- **Conclusion / next step:** hypothesis confirmed: exp003 was undertrained, and longer training both raises success and removes most of the seed variance.

## exp005_steps100k
- **Date:** 2026-10-10
- **Compared with:** exp003_chunk25, exp004_steps50k
- **Change:** `n_steps: 10K → 100K` (10× longer training, ~14 passes over the data).
- **Hypothesis:** at least as good as exp004. If success keeps rising from 50K to 100K, training length is still the bottleneck and 300K is worth trying; if it plateaus, the next bottleneck is model capacity (residual + wider MLP). Caveat: with 3 seeds and exp003's ±26 point spread, differences under ~15 points between exp004 and exp005 cannot be trusted.
- **Results:** success 38.0% ± 13.9% (seeds: 22%, 56%, 36%); valid loss 0.0239 ± 0.0003.
- **Observations:** worse than exp004, with more spread again. Training is deterministic per seed, so the first 50K steps of every exp005 run are identical to exp004 (same valid losses): the drop comes purely from steps 50K–100K. Over those steps the valid loss keeps rising (0.0224–0.0232 at 50K → 0.0239–0.0247 at 100K), i.e. the model overfits, and success falls with it (seed 0: 52% → 22%).
- **Conclusion / next step:** hypothesis rejected: with this small model, constant lr 5e-4 and MSE regression, training past ~50K steps overfits and hurts success. This is the opposite of the blog's "overfitting is not necessarily bad", which was observed with much larger flow-matching models. 300K steps is not worth running in this setting. Next options, one at a time: a learning-rate decay (e.g. cosine) to stabilize the end of training; evaluating intermediate checkpoints to find the best step; then larger capacity (residual + wider), where the blog's longer-is-better regime may appear.

## exp006_exec5 / exp007_exec10 / exp008_exec20 (eval-only)
- **Date:** 2026-10-11
- **Compared with:** exp004_steps50k (executes all 25 actions)
- **Change:** no training; the exp004 checkpoints are evaluated while executing only the first 5 / 10 / 20 actions of each 25-step chunk before predicting again (`eval.execute_steps`, `eval.ckpt_exp`). Run with `./eval_exp.sh`.
- **Hypothesis:** re-planning more often lets the policy correct drift, but each new chunk may commit to a different mode than the one it interrupts (different speed or detour side), which makes the motion inconsistent. Expect 20 ≈ 25, 10 similar or slightly better, and 5 clearly worse, moving towards the single-step failure of exp001 (the blog: open-loop chunks beat closed-loop control on this data).
- **Results:** success vs. executed steps per chunk (same exp004 checkpoints): 5 → 10.0% ± 5.9% (18/4/8); 10 → 32.0% ± 5.9% (26/30/40); 20 → 56.0% ± 0.0% (56/56/56); 25 (exp004) → 56.7% ± 3.4%.
- **Observations:** success falls monotonically as re-planning gets more frequent; 20 and 25 are equivalent. Valid loss is identical by construction (same checkpoints), so only the execution strategy differs.
- **Conclusion / next step:** hypothesis partly confirmed: 5 is much worse and 20 ≈ 25, but 10 is also clearly worse, not better. More frequent re-planning hurts on this data, consistent with the blog's "open-loop beats closed-loop": consecutive chunks are not consistent with each other, so cutting a chunk short and switching to a new one makes the motion incoherent. Keep executing the full chunk.

## exp009_residual
- **Date:** 2026-10-11
- **Compared with:** exp004_steps50k
- **Change:** `model.residual: false → true`. Same layers and the same 2.48M parameters; a skip connection `h + f(h)` around hidden layers 2–10.
- **Hypothesis:** little or no gain. The plain 10 × 512 MLP with LayerNorm already trains well, and residual connections mostly matter for deeper or wider networks. A possible side effect: the summed residual stream may fit faster, so with 50K steps it may overfit earlier (cf. exp005).
- **Results:** success 28.0% ± 7.1% (seeds: 34%, 18%, 32%) vs. exp004's 56.7% ± 3.4%; valid loss 0.0223 ± 0.0003 vs. 0.0226 ± 0.0004.
- **Observations:** the residual model learns more slowly early on (valid loss 0.033 vs. 0.027 at 5K steps) and reaches its best valid loss later (~38–40K vs. ~23–32K steps), ending with practically the same valid loss as the plain model. Yet its success rate is half, a gap far larger than the seed spread. Another case where valid loss does not predict closed-loop success.
- **Conclusion / next step:** hypothesis rejected: at this size, adding skip connections (post-LayerNorm, `h + f(h)`) clearly hurts. Not yet clear why; candidates: the residual stream grows with depth because LayerNorm outputs are summed without a final normalization; or it simply needs a different number of steps. Residual connections are only expected to matter for much deeper / wider models, so keep the plain MLP for now and revisit residual blocks (e.g. the reference's two-layer block, or a pre-LN block) only when scaling up width.

## exp010_depth15 / exp011_depth20 / exp012_depth40
- **Date:** 2026-10-11
- **Compared with:** exp004_steps50k (depth 10)
- **Change:** `model.depth: 10 → 15 / 20 / 40`. Plain MLP (no residual), width 512, LayerNorm, chunk 25, 50K steps; evaluation executes the full 25-step chunk.
- **Hypothesis:** 15 and 20 about the same as depth 10, maybe slightly better (more capacity, and the plain 10-layer MLP trained without trouble). 40 likely worse: a 40-layer plain MLP is where optimization should start to degrade. Diagnostic: if depth 40 ends with a *higher train loss* than depth 10, that is the degradation problem (an optimization failure, not overfitting) and the case where residual connections are needed.
- **Results:** success vs. depth: 10 (exp004) → 56.7% ± 3.4%; 15 → 24.0% ± 24.4% (58/2/12); 20 → 33.3% ± 9.6% (38/42/20); 40 → 0.0% (0/0/0). Valid loss: 0.0226 / 0.0253 / 0.0239 / 0.1695.
- **Observations:** train loss over the last 5K steps rises with depth: 0.0174 (10) → 0.0213 (15) → 0.0200 (20) → 0.1729 (40). Depth 40 never trains: its loss stays near the level of predicting the mean action from the very start, and all three seeds end with the same valid loss (0.1695). Deeper models also learn more slowly early on (valid loss at 5K steps: 0.027 → 0.030 → 0.037 → 0.170).
- **Conclusion / next step:** hypothesis partly rejected: depth 15 and 20 are already worse than 10, not equal. The rising *train* loss shows this is the degradation problem (optimization), not overfitting; at 40 layers the plain MLP fails completely. Depth 10 is the best plain depth. To grow capacity, either go wider instead of deeper (the reference uses 8 × 4096), or fix the residual block (pre-LN `h + f(LN(h))` with a final LayerNorm) and re-test it at depth 20/40, where a working residual connection should recover the lost performance.

## exp013_w4096
- **Date:** 2026-10-11
- **Compared with:** exp004_steps50k (width 512)
- **Change:** `model.hidden_dim: 512 → 4096` (depth 10, plain MLP; 2.5M → 152M parameters). `valid_every: 500 → 2500` only to save time; validation does not affect training.
- **Hypothesis:** higher success than exp004. The blog found that this task needs very large MLPs (≥ 8 × 4096), and width does not cause the degradation that depth did (exp010–012). Risk: with constant lr and 50K steps, a 60× larger model may overfit sooner (cf. exp005).
- **Results:** success 53.3% ± 10.0% (seeds: 64%, 56%, 40%); valid loss 0.0254 ± 0.0003.
- **Observations:** no gain over width 512 (56.7%), despite 60× more parameters.
- **Conclusion / next step:** confounded: lr 5e-4 was tuned for width 512 and turned out to be too large at width 4096 (see exp014). Re-run at lr 1e-4 (exp020).

## exp014_w4096_d15
- **Date:** 2026-10-11
- **Compared with:** exp013_w4096 (depth 10)
- **Change:** `model.depth: 10 → 15` at width 4096, plain MLP, lr 5e-4.
- **Hypothesis:** the same degradation as at width 512: train loss rises with depth and success drops.
- **Results:** success 0.0% (0/0/0); valid loss 0.1696 ± 0.0002.
- **Observations:** complete collapse, worse than depth 15 at width 512 (24%). The loss stays at 0.17 = the MSE of always predicting the mean (normalized) action. Diagnosis (1K data, 2000-step probes): healthy at initialization, but within the first ~200 steps every input maps to the same last-layer feature (cosine similarity 1.000), after which the layer-1 gradient vanishes (~1e-8) and training cannot recover. It is not dead ReLUs (35–80% of units stay active). The same model trains normally with lr 1e-4, or with lr 5e-4 plus a 500-step warmup, or at width 512.
- **Conclusion / next step:** the collapse is caused by too large early updates: with Adam the per-weight step is ~lr regardless of width, so a wider layer changes its output more per step, and depth compounds it. lr 5e-4 is too large for width 4096; the OCBench reference uses 1e-4 there. exp015 (depth 20, lr 5e-4) was stopped after 8 minutes and exp016/exp018 (pre-LN at lr 5e-4) were cancelled; the width-4096 study is redone at lr 1e-4 (exp019–exp024).

## exp019_lr1e-4
- **Date:** 2026-10-11
- **Compared with:** exp004_steps50k (lr 5e-4)
- **Change:** `train.lr: 5e-4 → 1e-4` at width 512, depth 10. Needed as the lr-matched baseline for the width comparison (exp020), so that width is the only difference.
- **Hypothesis:** somewhat worse than exp004: at 5× lower lr, 50K steps covers less of the optimization, and exp004 at lr 5e-4 was not unstable at width 512.
- **Results:** success 61.3% ± 30.7% (seeds: 86%, 18%, 80%); valid loss 0.0249 ± 0.0013 (exp004: 56.7% ± 3.4%, 0.0226).
- **Observations:** two seeds are the best runs so far (86%, 80%), but seed 1 is at 18%. The mean is slightly above exp004, within noise; the spread is 9× larger. Valid loss is higher than exp004, as expected for a lower lr at a fixed number of steps.
- **Conclusion / next step:** hypothesis not confirmed: lr 1e-4 is not clearly worse, and can reach much higher success, but it is not reliably better. With a ±31-point spread, 3 seeds cannot separate lr 1e-4 from 5e-4 at width 512. It still serves as the lr-matched baseline for exp020. Worth a later look: why seed 1 fails (videos), and whether more seeds or an lr schedule stabilizes it.

## exp020_w4096_lr1e-4
- **Date:** 2026-10-11
- **Compared with:** exp019_lr1e-4 (width 512, same lr); also exp013 (same width, lr 5e-4)
- **Change:** `model.hidden_dim: 512 → 4096` at lr 1e-4, depth 10, plain.
- **Hypothesis:** better than exp019 and than exp013: with a stable lr, the 60× larger model should finally help, as in the blog. Risk: 50K steps at lr 1e-4 may be too short for 152M parameters (the reference trains for 2M steps).
- **Results:** success 61.3% ± 14.1% (seeds: 58%, 46%, 80%); valid loss 0.0236 ± 0.0001.
- **Observations:** same mean as exp019 (width 512, same lr: 61.3% ± 30.7%) with half the spread, and slightly above exp013 (width 4096, lr 5e-4: 53.3% ± 10.0%). The 60× larger model gives no clear gain in mean success at 50K steps.
- **Conclusion / next step:** hypothesis not confirmed: width 4096 is not clearly better than 512 at lr 1e-4 and 50K steps; the differences are within seed noise. Likely the model is far from converged (the reference trains for 2M steps); width may only pay off with much longer training. Keep in mind when reading exp021–exp024.

## exp022_w4096_d20_lr1e-4 (exp021 cancelled)
- **Date:** 2026-10-11
- **Compared with:** exp020 (depth 10, same width and lr); exp021 also vs. exp014 (lr 5e-4)
- **Change:** `model.depth: 10 → 15 / 20` at width 4096, lr 1e-4, plain.
- **Hypothesis:** no collapse at lr 1e-4 (the 2000-step probe trained normally), but still some degradation with depth: higher train loss and lower success than depth 10, milder than at width 512.
- **Results:** depth 20 (exp022): success 52.7% ± 17.9% (78/40/40); valid loss 0.0243 ± 0.0003. Depth 15 (exp021) was stopped during its first seed and removed, in favor of exp025.
- **Observations:** at lr 1e-4 depth 20 no longer collapses (compare 0% for depth 15 at lr 5e-4). There is still mild degradation: train loss over the last 5K steps rises from 0.0134 (depth 10) to 0.0167 (depth 20), and success is lower than depth 10 (61.3%), though within seed noise.
- **Conclusion / next step:** at lr 1e-4 the plain MLP stays trainable up to depth 20 at width 4096, with only mild degradation; extra depth brings no gain.

## exp023_w4096_d10_preln_lr1e-4 / exp024_w4096_d20_preln_lr1e-4
- **Date:** 2026-10-11
- **Compared with:** exp020 / exp022 (same width, depth and lr, plain)
- **Change:** `model.residual: false → pre_ln` (input projection, `h = h + Linear(ReLU(LayerNorm(h)))`, final LayerNorm; same parameter count).
- **Hypothesis:** at depth 10 about equal to plain; at depth 20 clearly better than plain, with train loss no longer rising with depth: the identity path keeps input information from being washed out.
- **Results:** depth 10 (exp023): success 38.7% ± 16.5% (28/26/62), valid loss 0.0233 vs. plain 61.3% (exp020), 0.0236. Depth 20 (exp024): success 50.0% ± 7.5% (48/42/60), valid loss 0.0228 vs. plain 52.7% (exp022), 0.0243.
- **Observations:** train loss over the last 5K steps: pre-LN 0.0163 (d10) and 0.0162 (d20), i.e. flat with depth, while plain rises from 0.0134 to 0.0167. So pre-LN does remove the depth degradation, but at depth 10 it fits the training data worse than the plain MLP, and its success is lower. At depth 20 the two are equal in success; pre-LN has a slightly lower valid loss.
- **Conclusion / next step:** hypothesis partly confirmed: pre-LN keeps train loss from rising with depth, but it does not beat the plain MLP at either depth, and is clearly worse at depth 10. At this scale (depth ≤ 20, 50K steps) residual connections are not needed. Overall none of the width/depth/residual changes moved success beyond ~60%, and most differences are within seed noise. Remaining suspects for the plateau: training length (50K vs. the reference's 2M steps), MSE regression on multimodal actions (the reference uses flow matching), and training on failed episodes (the reference uses success-only data).

## exp025_depth40_preln
- **Date:** 2026-10-12
- **Compared with:** exp012_depth40 (plain, same width 512, depth 40, lr 5e-4: collapsed to 0%)
- **Change:** `model.residual: false → pre_ln`. Width 512, depth 40, lr 5e-4, chunk 25, 50K steps.
- **Hypothesis:** pre-LN rescues depth 40: no collapse (train loss far below the 0.17 of mean prediction), and success in the range of the depth-10 plain model (exp004, 56.7%). The 2000-step probe at depth 40 / width 512 trained normally with pre-LN (loss 0.010 vs. 0.034 plain). At width 4096 pre-LN kept train loss flat with depth but did not beat plain, so a gain over exp004 is not expected.
- **Results:** success 65.3% ± 9.3% (seeds: 78%, 62%, 56%); valid loss 0.0223 ± 0.0003 (exp012: 0%, 0.1695; exp004: 56.7% ± 3.4%, 0.0226).
- **Observations:** no collapse: train loss over the last 5K steps is 0.0180, at the level of the depth-10 plain model (0.0174) instead of 0.1729 for plain depth 40. Highest mean success of all experiments so far and the lowest valid loss, though the margin over exp004 (+8.7 points) is within the seed spread.
- **Conclusion / next step:** hypothesis confirmed: pre-LN fully rescues depth 40 at width 512, with the same lr that made the plain model collapse. It is the best configuration so far, but not clearly better than exp004; more seeds would be needed to tell. Contrast with width 4096 / lr 1e-4, where pre-LN did not help: at width 512 / lr 5e-4 the deep plain stack fails, so the residual path matters.

## exp026_depth80_preln
- **Date:** 2026-10-12
- **Compared with:** exp025_depth40_preln
- **Change:** `model.depth: 40 → 80` (pre-LN residual, width 512, lr 5e-4, chunk 25, 50K steps; ~13M → ~26M parameters).
- **Hypothesis:** still trains (no collapse; train loss near exp025's 0.018), since the pre-LN identity path does not degrade with depth. Success about the same as exp025 (65.3%): going from 10 to 40 layers gained at most a few points, so another doubling is unlikely to add much, and 50K steps may be too few for an 80-layer model.
- **Results:** success 42.7% ± 15.9% (seeds: 64%, 38%, 26%); valid loss 0.0232 ± 0.0019 (exp025: 65.3% ± 9.3%, 0.0223).
- **Observations:** no collapse and no optimization trouble: train loss over the last 5K steps is 0.0182, the same as depth 40 (0.0180), and the best valid losses during training are also similar (0.0207–0.0222). Yet success is ~23 points lower on average and varies more across seeds; the final valid loss is slightly higher and less stable.
- **Conclusion / next step:** hypothesis partly rejected: depth 80 trains fine (pre-LN keeps it trainable) but does not help and probably hurts closed-loop success; doubling depth past 40 adds nothing at 50K steps. Depth is not the bottleneck; stop scaling depth. Best configuration remains exp025 (depth 40, pre-LN).

## exp027_exec1 (eval-only)
- **Date:** 2026-10-12
- **Compared with:** exp004 (executes 25 of 25) and exp006–008 (5 / 10 / 20); also exp001 (a single-step model)
- **Change:** no training; the exp004 chunk-25 checkpoints execute only the first action of each chunk and predict again every step, i.e. fully closed-loop, like the single-step policy.
- **Purpose:** separate the two possible sources of the chunking gain: (a) training on 25-step targets gives a better model, vs. (b) executing the chunk open-loop gives better behavior.
- **Hypothesis:** close to exp001 (≈0%), continuing the trend 25 → 20 → 10 → 5 (57% → 56% → 32% → 10%). If so, nearly all of the gain comes from open-loop execution, not from the multi-step training targets.
- **Results:** success 2.0% ± 1.6% (seeds: 0%, 4%, 2%), vs. 56.7% when the same checkpoints execute the full chunk, and 0.7% for the single-step model (exp001). Full curve over executed steps: 1 → 2%, 5 → 10%, 10 → 32%, 20 → 56%, 25 → 57%.
- **Observations (rollout analysis, 10 episodes per policy, eval seeds 0–9; script in the session scratchpad):** mean nearest-neighbour distance from visited states to 300K training states, by step range 0–50 / 50–200 / 200–500 / 500–1250: expert 1.22 / 0.94 / 0.91 / 0.89; chunk 25 executed fully 1.19 / 0.95 / 0.83 / 1.00 (stays in-distribution); chunk model executed 1 step 1.24 / 1.65 / 2.43 / 3.14; single-step model 1.43 / 2.56 / 3.87 / 5.23 (both drift further and further). Mean change between consecutive actions: expert 0.036, chunk-25 0.014 (smoother than the expert), chunk exec-1 0.053 (jittery: consecutive chunks disagree), single-step 0.029. Stalled steps (end-effector < 0.5 mm/step) and longest stall: expert 12.6% / 60 steps, chunk-25 14.6% / 63, chunk exec-1 14.2% / 112, single-step 28.4% / 306.
- **Conclusion / next step:** hypothesis confirmed: almost all of the chunking gain comes from executing the chunk open-loop, not from training on 25-step targets (the per-step accuracy of both models is also similar: first-action valid loss 0.0155 vs. 0.0184). Closed-loop execution re-decides every step from the state alone, while the expert's actions also depend on hidden per-episode variables (speed, detour side, time in its plan). The re-decisions are inconsistent (jitter), stall at the expert's pauses (when to resume is not visible in the state), and push the arm off the training distribution, where predictions get worse: compounding error. A 25-step chunk commits to one consistent segment, passes through pauses, and cuts the number of decisions per episode from ~1250 to ~50. The MSE chunk is an average over modes (smoother than the expert); flow matching could commit to a single mode instead.

## exp028_chunk25_basic_flow / exp029_mse_d10_10k
- **Date:** 2026-10-12
- **Compared with:** each other (same backbone: input projection + 9 pre-LN residual blocks + LayerNorm + Linear, width 512, lr 5e-4, chunk 25, 10K steps)
- **Change:** exp028 trains a flow matching policy (`train_flow.py`: velocity $v_\theta(o, x_t, t)$ on $x_t=(1-t)x_0+t\,x_1$, target $x_1-x_0$, $t\sim U[0,1]$ per sample; `eval_flow.py`: 10 Euler steps from $x_0\sim\mathcal N(0,I)$ per chunk). exp029 is the MSE regression baseline with the same backbone and steps.
- **Hypothesis:** at 10K steps flow is not better and may be worse than MSE: the flow loss was still falling at 10K steps (valid 0.085), and the seed-0 checkpoint samples chunks with error 0.075 to the ground truth vs. ~0.046 expected for an ideal sampler (2× the MSE model's 0.023). A quick 5-episode check of that checkpoint gave 0/5. This pair is a smoke-scale check of the pipeline; exp030 is the real comparison.
- **Note:** `valid_loss` of flow runs is the velocity-matching loss on fixed noise, not comparable with MSE runs.
- **Results:** exp028 (flow): success 0.0% ± 0.0% (0/0/0), every episode times out at 1250 steps; valid (velocity) loss 0.0857. exp029 (MSE): success 20.0% ± 11.4% (34/20/6); valid loss 0.0259.
- **Conclusion:** hypothesis confirmed: at 10K steps flow is clearly worse than MSE with the same backbone, so the failure is not just "10K steps is too short for any model".

## exp030_flow_d40_50k
- **Date:** 2026-10-12
- **Compared with:** exp025_depth40_preln (MSE, 65.3% ± 9.3%)
- **Change:** flow matching instead of MSE regression; same backbone (width 512, depth 40, pre-LN), lr 5e-4, chunk 25, 50K steps; 10 Euler steps at eval.
- **Hypothesis:** better than exp025: the MSE chunk is an average over modes (exp027 analysis: smoother than the expert, stalls at pauses), while a sampled chunk commits to one mode, e.g. one resume time after a pause. Expect higher success and shorter mean episode length. Risk: flow matching usually needs more steps than MSE to converge, so 50K may not be enough.
- **Results:** success 1.3% ± 0.9% (seeds: 2%, 0%, 2%), mean length 1245 (almost every episode times out); valid (velocity) loss 0.0513 ± 0.0008 (exp025: 65.3% ± 9.3%).
- **Observations (diagnosis on seed 0; scripts `diag_*.py` in the session scratchpad):**
  - Offline, a single sampled chunk has error 0.054 to the ground truth vs. 0.022 for the MSE policy. Over 32 samples per state: bias² (sample mean vs. ground truth) 0.040, variance 0.017. An ideal sampler would have bias² ≈ 0.022 (the MSE error) and variance ≈ 0.022. So the learned conditional distribution is off-center and too narrow, not too noisy.
  - Closed loop (10 episodes): averaging 4 or 16 samples per chunk still gives 0%, so sampling noise is not the cause. Also 0% for 1 Euler step, 50 steps, and noise temperature 0.
  - Not the gripper: MSE arm + flow gripper 80% (MSE alone 90%); flow arm + MSE gripper 0%. The flow arm gets from ~70 cm to ~10 cm of the block, then wanders at 5–30 cm and never gets closer than 4.5 cm; the MSE arm converges to < 1 cm.
  - Under-trained: the flow valid loss is still falling at 50K steps (0.0536 → 0.0510 over the last 5K steps), while the MSE valid loss is flat from ~20K steps.
- **Conclusion / next step:** hypothesis rejected: flow is far worse than MSE here. The failure is a biased conditional model (wrong mean action, especially for fine positioning), not multimodality or sampling noise. Likely causes: (1) noisy optimization: the velocity target has large irreducible noise (per-dim variance ~1 near $t=1$), so gradients are much noisier than for MSE and the final weights are noisy; (2) under-training; (3) weak conditioning: obs and $t$ enter only at the input layer, next to the 175-d $x_t$. Ablations exp031–exp033 test one each.


## exp031_flow_ema / exp032_flow_film / exp033_flow_200k
- **Date:** 2026-10-12
- **Compared with:** exp030 (flow, width 512, depth 40, 50K steps: 1.3%)
- **Change (one each):**
  - exp031: exponential moving average of the weights (`train.ema_decay: 0.999`); validation and the saved checkpoint use the EMA weights. Standard in diffusion / flow policies.
  - exp032: FiLM conditioning (`model.film: true`): $[o, t]$ also produces a scale and shift after the LayerNorm of every residual block (zero-initialized, so training starts from the exp030 network).
  - exp033: `train.n_steps: 50000 → 200000`.
- **Hypotheses:**
  - exp031: large gain (the biggest of the three): averaging over ~1000 steps of noisy updates removes most of the weight noise, so the conditional mean becomes accurate. Expect lower valid loss and lower bias².
  - exp032: moderate gain: obs and $t$ can steer every layer instead of being mixed once with the 175-d $x_t$.
  - exp033: gain, since the valid loss is still falling at 50K steps; but noisy final weights remain without EMA.
- **Results (exp031, seed 0 only):** success 2% (1/50); valid (velocity) loss 0.0434 vs. 0.0510 for exp030 seed 0. Seeds 1–2 were stopped: the diagnosis below showed the bias unchanged, so more seeds were not worth the GPU time.
- **Observations (exp031):** EMA lowers the velocity loss by 15%, but not the error that matters. At $t=0$ the ideal velocity is $\mathbb E[x_1\mid o]-x_0$, so the mean over $x_0$ of $x_0+v(x_0,0,o)$ is the network's estimate of the conditional mean action. Its squared distance to the MSE policy's prediction is 0.0246 (exp030) and 0.0250 (exp031), as large as the MSE policy's whole error (0.022). The noise the network fails to cancel at $t=0$ is small (variance 0.006 / 0.004).
- **Conclusion (exp031):** hypothesis rejected: weight noise is not the cause. The flow network learns the observation-to-mean-action mapping much worse than the MSE network with the same backbone. In the velocity loss this mapping is learned mainly from samples at small $t$ (at large $t$, $x_t$ already reveals $x_1$), and the network must also carry the 175-d noise through every layer to output $-x_0$. Two new ablations target this directly (exp034, exp035).
- **Results (exp033):** success 50.7% ± 0.9% (seeds: 52%, 50%, 50%); valid (velocity) loss 0.0362 ± 0.0000 (exp030: 1.3%, 0.0513). The valid loss flattens after ~140K steps (0.0379 at 140K, 0.0362 at 200K).
- **Observations (exp033, seed 0):** bias² 0.023 (exp030: 0.040; MSE policy error 0.022), variance 0.014, squared distance between the sample mean and the MSE prediction 0.010 (exp030: 0.025). Velocity loss by $t$ bin is lower everywhere than exp030, e.g. 0.028 vs. 0.041 at $t<0.1$.
- **Conclusion (exp033):** hypothesis confirmed: v-prediction with uniform $t$ is not broken, only slow; with 4× the steps it learns the conditional mean about as well as x-prediction does at 50K steps (exp034). Still ~15 points below MSE (exp025: 65.3%).
- **Results (exp032):** success 51.3% ± 2.5% (seeds: 54%, 48%, 52%); valid (velocity) loss 0.0445 ± 0.0005 (exp030: 1.3%, 0.0513).
- **Observations (exp032, seed 0):** overall bias² 0.033 (exp030: 0.040), smaller than for the other fixes, but split by task phase: far from the block (> 10 cm) 0.041 (exp030: 0.044, i.e. not improved), final approach (1–10 cm) 0.023 (exp030: 0.028), grasp/carry 0.019 (exp030: 0.023). Velocity loss is lower than exp030 in every $t$ bin (0.033 vs. 0.041 at $t<0.1$).
- **Conclusion (exp032):** hypothesis exceeded: FiLM is the best single fix (1.3% → 51%, the most stable across seeds), although it barely improves the coarse far-phase actions. Injecting $[o,t]$ into every block makes the velocity field sensitive to the small observation differences that matter for the final approach.

## exp034_flow_xpred / exp035_flow_tbeta
- **Date:** 2026-10-12
- **Compared with:** exp030 (flow, v-prediction, uniform $t$: 1.3%)
- **Change (one each):**
  - exp034: x-prediction (`train.pred: x`): the network outputs the clean chunk $\hat x_1(o,x_t,t)$, trained with $\|\hat x_1-x_1\|^2$ (= the velocity loss weighted by $(1-t)^2$); sampling uses $v=(\hat x_1-x_t)/(1-t)$. At $t=0$ the target is exactly the MSE target, and the network never has to output the noise.
  - exp035: $t$ from the $\pi_0$ Beta schedule (`train.t_dist: beta`): $t=0.999(1-u)$, $u\sim\mathrm{Beta}(1.5,1)$; mean $t$ 0.40, 29% of samples at $t<0.2$ (vs. 20% uniform), 9% at $t>0.8$ (vs. 20%). Validation still uses uniform $t$.
- **Hypotheses:**
  - exp034: large gain, close to or above exp025 (65%): the observation-to-action mapping is learned like in the MSE model, and sampling can still choose a mode. Offline, the sample mean should be about as close to the ground truth as the MSE prediction.
  - exp035: moderate gain: more training signal on the noisy $t$ where the observation matters most, but the network still predicts velocity.
- **Run order:** exp034, exp035, then exp032 (FiLM), next to exp033 (200K steps) on the same GPU.
- **Results (exp034):** success 44.0% ± 8.6% (seeds: 40%, 56%, 36%), vs. 1.3% for exp030 and 65.3% for the MSE policy exp025. Valid loss 0.0090 (x-space loss, not comparable with the velocity loss).
- **Observations (exp034, seed 0, 4000 valid states, 32 samples each):** single-sample error 0.035 (exp030: 0.054), bias² 0.026 (exp030: 0.040; MSE policy error 0.022), variance 0.011 (exp030: 0.017), squared distance between the sample mean and the MSE prediction 0.010 (exp030: 0.025). For comparison, exp033 seed 0 (v-prediction, 200K steps, 52% success) reaches almost the same numbers: 0.036 / 0.023 / 0.014 / 0.010.
- **Conclusion (exp034):** hypothesis mostly confirmed: x-prediction removes most of the failure (1.3% → 44%) by learning the observation-to-mean-action mapping as fast as the MSE model; at 50K steps it matches v-prediction at 200K steps. Still ~20 points below MSE; the samples are too narrow (variance 0.011 vs. ~0.022 expected), and some bias remains.
- **Results (exp035):** success 33.3% ± 7.7% (seeds: 44%, 26%, 30%); valid (velocity, uniform $t$) loss 0.0512 ± 0.0006, the same as exp030 (0.0513).
- **Observations (exp035, seed 0):** bias² 0.026 (exp030: 0.040), variance 0.016, squared distance to the MSE prediction 0.013. Velocity loss by $t$ bin vs. exp030: lower at small $t$ (0.031 vs. 0.041 at $t<0.1$; 0.030 vs. 0.037 at 0.1–0.2) and higher near the data end (0.222 vs. 0.173 at $t>0.9$); averaged over uniform $t$ the two cancel. So the uniform-$t$ valid loss does not predict closed-loop success for flow policies.
- **Conclusion (exp035):** hypothesis confirmed (moderate gain, 1.3% → 33%): putting more training on noisy $t$ fixes much of the bias, at the cost of accuracy near $t=1$; weaker and less stable across seeds than x-prediction (44%) or 4× longer training (51%).

## Summary: why the first flow policy (exp030) failed
- **Date:** 2026-10-12
- **Results (width 512, depth 40, chunk 25, 10 Euler steps; 3 seeds unless noted):**

| exp | change vs. exp030 | success | approach-phase bias² (seed 0) |
|---|---|---|---|
| exp025 | MSE policy (reference) | 65.3% ± 9.3% | 0.022 (MSE error) |
| exp030 | – (v-prediction, uniform $t$, 50K steps) | 1.3% ± 0.9% | 0.028 |
| exp031 | EMA 0.999 (seed 0 only) | 2% | 0.027 |
| exp035 | $t$ from the $\pi_0$ Beta schedule | 33.3% ± 7.7% | 0.024 |
| exp034 | x-prediction | 44.0% ± 8.6% | 0.020 |
| exp033 | 200K steps | 50.7% ± 0.9% | 0.021 |
| exp032 | FiLM conditioning | 51.3% ± 2.5% | 0.023 |

- **Cause:** not sampling noise, Euler steps, the gripper, or weight noise. The velocity network had not yet learned an accurate conditional mean $\mathbb E[x_1\mid o]$ for the final approach (effector 1–10 cm from the block), so the arm reached ~10 cm and then wandered. Every fix that brought the approach-phase bias² down to ≈ 0.023 (the MSE level) raised success to 33–51%; the two that did not (exp030, exp031: ≥ 0.027) stayed at ~0%. Errors far from the block hardly matter (exp032 keeps exp030's far-phase bias and still succeeds), because every new chunk replans from the current state.
- **Why v-prediction with uniform $t$ learns this slowly:** the observation matters most at small $t$, where $x_t$ carries little information about $x_1$; there the target $\mathbb E[x_1\mid o]-x_0$ also requires the network to carry the 175-d noise to its output; and uniform $t$ spends most of the gradient on large $t$, where the loss is dominated by irreducible noise (0.17 at $t>0.9$ vs. 0.04 at $t<0.1$). Each fix addresses one part: more weight on small $t$ (exp035, and implicitly exp034 via the $(1-t)^2$ weight), no noise in the output (exp034), stronger conditioning (exp032), or simply more steps (exp033).
- **Lesson for evaluation:** the uniform-$t$ velocity loss does not predict success (exp035 has the same valid loss as exp030 but 33% vs. 1%). Use the loss at small $t$ or the bias of the sample mean in the approach phase instead.
- **Remaining gap:** all flow variants are still 14–32 points below MSE, and their samples are narrower than the data (variance 0.011–0.016 vs. ≈ 0.022 expected).
- **Next steps:** (1) combine the fixes (x-prediction + FiLM, then longer training) to see whether they add up and close the gap to MSE; (2) separate the two effects inside exp034 (x-prediction output with the velocity loss, $1-t$ clipped at 0.05); (3) log the small-$t$ loss during training as the model-selection metric.

## exp036_flow_xpred_film
- **Date:** 2026-10-12
- **Compared with:** exp034 (x-prediction only: 44.0% ± 8.6%), exp032 (FiLM only: 51.3% ± 2.5%); reference exp025 (MSE: 65.3% ± 9.3%)
- **Change:** `train.pred: x` and `model.film: true` together (width 512, depth 40, 50K steps, uniform $t$, 10 Euler steps).
- **Hypothesis:** better than either alone (> 51%), closing part of the gap to MSE: the two act through different mechanisms (x-prediction: no noise in the output and more weight on small $t$; FiLM: the observation reaches every block), and both lower the approach-phase bias. Risk: they overlap, since both make the observation-dependence easier to learn, so the gain may be small.
- **Results:** success 62.7% ± 9.0% (seeds: 50%, 70%, 68%) vs. 51.3% ± 2.5% (exp032, FiLM only), 44.0% ± 8.6% (exp034, x-prediction only) and 65.3% ± 9.3% (exp025, MSE). Valid loss 0.0083 (x-space).
- **Conclusion / next step:** hypothesis confirmed: the two fixes add up (+11 points over the better single fix), and the flow policy now matches the MSE policy within seed noise at 50K steps. Chosen as the recipe for the 1M-step runs (exp039/exp040).

## exp037_flow_film_tsin
- **Date:** 2026-10-12
- **Compared with:** exp032 (FiLM, raw scalar $t$: 51.3% ± 2.5%)
- **Change:** `model.t_embed_dim: 0 → 32`: $t$ enters the input and every FiLM layer as 32 sinusoidal features $[\sin(1000\,t\,w_k), \cos(1000\,t\,w_k)]$, $w_k=10000^{-k/16}$, $k=0..15$ (the DiT timestep embedding), instead of one scalar. Also new in the code (no effect on results): validation is computed in chunks, so a run needs ~4 GiB instead of ~16 GiB of GPU memory and the 3 seeds run in parallel.
- **Hypothesis:** small gain or no change. With the raw scalar, the network must build every $t$-dependent switch from one input through ReLUs; sinusoidal features give it many smooth and sharp functions of $t$ directly, which should help most where the velocity changes fast with $t$ (near $t=1$). FiLM already lets $t$ scale every block, which removes much of the need, so the gain in success is expected to be small (within ~5 points).
- **Results:** success 40.7% ± 15.1% (seeds: 30%, 62%, 30%) vs. 51.3% ± 2.5% for exp032; valid (velocity) loss 0.0425 ± 0.0002 (exp032: 0.0445).
- **Observations:** lower valid loss than exp032, but lower and much less stable success. Offline, the approach-phase bias² is 0.022–0.024 for all three seeds (exp032: 0.023), so it does not explain the 30% vs. 62% spread: at this level of accuracy the phase bias no longer predicts which seed succeeds. The 32-point spread is far above the evaluation noise (binomial std ≈ 7 points at 50 episodes), so it is real seed-to-seed variation.
- **Conclusion / next step:** hypothesis partly confirmed: no gain from sinusoidal $t$ features with FiLM; the mean is ~10 points lower with a large spread, which 3 seeds cannot separate from no change. Keep the raw scalar $t$ for the long runs. A possible cause of the extra variance, not tested: the DiT scaling (angles up to $1000\,t$) gives features that oscillate many times between the 10 Euler time points, so the network can fit a rough function of $t$.

## exp038_flow_film_only
- **Date:** 2026-10-12
- **Compared with:** exp032 (concat $[o,x_t,t]$ at the input + FiLM: 51.3% ± 2.5%)
- **Change:** `model.cond_input: true → false`: the input layer sees only $x_t$; $o$ and $t$ enter only through FiLM (standard FiLM conditioning, as in Diffusion Policy; DiT's adaLN does the same for $t$).
- **Hypothesis:** about the same as exp032 (within ~5 points), i.e. the modulation is the channel that matters, and the input concat is redundant once every block is modulated. Risk: FiLM is zero-initialized, so at the start of training the network ignores $o$ and $t$ completely and must learn the conditioning from zero, which may slow early learning and cost a few points at 50K steps.
- **Results:** success 34.0% ± 7.1% (seeds: 38%, 40%, 24%) vs. 51.3% ± 2.5% for exp032; valid (velocity) loss 0.0447 ± 0.0000 (exp032: 0.0445).
- **Conclusion / next step:** hypothesis rejected: dropping the input concatenation costs ~17 points at the same validation loss, so the input concat is not redundant with FiLM here. Possible reasons (not tested): with zero-initialized FiLM the network starts with no access to $o$ or $t$ at all and must learn the conditioning from scratch; and only the input layer can mix $o$ with $x_t$ before the first nonlinearity. Keep the input concat.

## exp039_flow_long_10k / exp040_flow_long_50k
- **Date:** 2026-10-12
- **Compared with:** each other, and with the 50K-step runs of the same recipe; reference exp025 (MSE: 65.3% ± 9.3%). Modeled on the blog's figure (success / training loss / validation loss vs. steps for 10K vs. 50K episodes).
- **Change:** recipe = the best tested flow configuration among exp032 (FiLM), exp036 (x-prediction + FiLM) and exp038 (FiLM only), chosen by mean success when they finish (no untested combination): exp036 (62.7% vs. 51.3% and 34.0%), i.e. x-prediction + FiLM with the input concat, raw scalar $t$, uniform $t$. Trained for 1M steps on 10K episodes (exp039) and 50K episodes (exp040; `data/block-single-task1-v0-50k.npz`, checked to share no episode with the validation set); same validation set (10K-val) for both. `keep_every: 100000` keeps a checkpoint every 100K steps (and at step 0), each evaluated on the standard 50 episodes by `eval_steps.py`; `valid_every: 10000`. One seed each. Speed only (same method): TF32 matmuls and a CUDA-graph-compiled training step (`tf32`, `compile`); checked on 3K steps to give the same validation loss as exp036 (0.0152 vs. 0.0154 at step 3000), and 1.7× the throughput with two runs on the GPU (232 vs. 138 steps/s in total). A first launch without them (16:43) was stopped after ~10 min and removed.
- **Hypotheses:**
  - Success keeps rising past 50K steps (as for v-prediction, exp033: 1.3% → 50.7% at 200K) and ends above the MSE policy.
  - The 10K run overfits: validation loss bottoms out and then rises while training loss keeps falling, earlier than for 50K; as in the blog, success does not drop with it ("overfitting is not necessarily bad").
  - 50K episodes ≥ 10K episodes in success at 1M steps, with a lower validation loss.
- **Results (one seed each; success at 100K, 200K, ..., 1M steps; figure `flow_long_steps.png`):**
  - exp039 (10K episodes): 74, 64, 76, 74, 62, 72, 78, 92, 68, 82% → mean over the ten checkpoints 74.2%, final 82%. Validation loss: minimum ≈ 0.0082 around 50K steps, then rises to 0.0116 by 400K and stays at ~0.0113; training loss keeps falling (0.0059 at 100K → 0.0031 at 1M).
  - exp040 (50K episodes): 48, 80, 66, 40, 62, 70, 68, 50, 74, 36% → mean 59.4%, final 36%. Validation loss flat at ~0.006 (0.0057–0.0063), half of exp039's; training loss 0.0077 → 0.0059.
- **Observations:**
  - 10K episodes reproduces the blog's "overfitting is not necessarily bad": the validation loss rises by ~40% from its minimum and stays high, while success stays at 62–92% and is higher on average (74%) than any 50K-step run (exp036: 62.7%; MSE exp025: 65.3%).
  - 50K episodes generalizes better in loss (no overfitting, validation loss half of exp039's) but not in success: lower mean and much larger swings between checkpoints (36–80%).
  - Success jumps by 10–30 points between neighbouring checkpoints in both runs (evaluation noise alone is ±7 points at 50 episodes), so single checkpoints are unreliable; the training is constant-lr (5e-4), no EMA, so the final weights keep moving.
- **Conclusion / next step:** hypotheses 1 and 2 confirmed for 10K episodes (longer training helps; overfitting in validation loss without a drop in success); hypothesis 3 rejected with one seed (50K episodes is not better here). Correction after re-reading the blog: it reports the same effect ("A smaller dataset can be better than a larger dataset (block-single with {10K, 50K} episodes)", "using a larger dataset can hurt(!) performance"), so hypothesis 3 was against the blog and the 10K > 50K result reproduces it. Validation loss again fails to predict success. Before reading more into 10K vs. 50K: reduce the checkpoint-to-checkpoint noise (lr decay or EMA of the weights over long training, more evaluation episodes) and add seeds. Possible reason for 50K being worse, not tested: the 10.5M-parameter network underfits 5× more data in 1M steps (training loss 2× higher than exp039), while the blog's 50K-episode runs use much larger networks.

## exp041_mse_long_10k / exp042_mse_long_50k
- **Date:** 2026-10-12
- **Compared with:** exp039 / exp040 (flow, x-prediction + FiLM, same 1M-step protocol); exp025 (the same MSE recipe at 50K steps: 65.3% ± 9.3%)
- **Change:** the exp025 MSE recipe (width 512, depth 40, pre-LN, lr 5e-4, chunk 25) trained for 1M steps on 10K (exp041) and 50K (exp042) episodes; checkpoints every 100K steps evaluated on the standard 50 episodes; same validation set, `valid_every`, TF32 and compiled training step as exp039/exp040 (speed only; 3K-step check gives exp025's validation losses). One seed each.
- **Hypotheses:**
  - MSE does not benefit from long training the way flow does: success peaks early (≤ 200K steps) and then drops as it overfits, as for the small plain MSE model (exp004 → exp005: 56.7% at 50K → 38.0% at 100K). The blog's "overfitting is not necessarily bad" is a flow-matching result.
  - With 10K episodes, flow (exp039: mean 74%) ends clearly above MSE; the gap is the benefit of a generative policy at long training.
  - 50K episodes delays the MSE overfitting (validation loss rises later and less), so MSE degrades less with 50K than with 10K.
- **Results (one seed each; success at 100K, ..., 1M steps; figures `mse_long_steps.png`, `flow_vs_mse_steps.png`):**
  - exp041 (MSE, 10K): 64, 52, 40, 72, 38, 58, 66, 40, 74, 62% → mean 56.6%, final 62%. Validation MSE rises slowly from 0.0226 (100K) to 0.0253 (1M); training MSE 0.0157 → 0.0120.
  - exp042 (MSE, 50K): 38, 10, 58, 62, 48, 36, 50, 78, 26, 52% → mean 45.8%, final 52%. Validation MSE flat at ~0.016; training MSE ~0.016–0.020.
  - Flow at the same checkpoints (exp039 / exp040): mean 74.2% (10K) and 59.4% (50K).
- **Conclusion:** hypothesis 1 rejected: MSE does not collapse with long training (no peak-then-drop; it stays around 40–75%, mean 57% vs. 65.3% for exp025 at 50K steps), so the exp004 → exp005 drop of the small plain model does not carry over to the depth-40 pre-LN model. Hypothesis 2 confirmed: flow is above MSE by 15–17 points on average at both dataset sizes, at every data size and most checkpoints. Hypothesis 3 rejected: 50K episodes is worse than 10K for MSE too (45.8% vs. 56.6%), so the "larger dataset can hurt" effect is not specific to flow matching. Both methods fluctuate by 20–40 points between neighbouring checkpoints, so differences of a few points are not meaningful with one seed.

## exp043_flow_w1024_d10_50k / exp044_flow_w1024_d40_50k
- **Date:** 2026-10-13
- **Compared with:** exp040 (flow, x-prediction + FiLM, width 512 × depth 40, 11.7M parameters, 50K episodes, 1M steps: mean 59.4% over the ten checkpoints)
- **Change (one each vs. exp040):**
  - exp043: shape at about the same parameter count: width 512 → 1024, depth 40 → 10 (~10M parameters, about the same compute per step).
  - exp044: capacity: width 512 → 1024 at depth 40 (~43M parameters, ~4× compute per step).
  - Same lr (5e-4), 50K episodes, 1M steps, checkpoints every 100K steps evaluated on 50 episodes; one seed each.
- **Background:** the blog needs "at least [4096]*8-sized residual MLPs" (≈ 0.5B parameters with 8192 width) on block-double with infinite data, and says previous benchmarks used "[1024]*4 at most". Depth beyond 40 did not help us (exp026: depth 80 worse), and width 4096 did not help at 50K steps with MSE (exp013/exp020), but width was never tested with flow and long training. The blog also finds 10K > 50K episodes with its large models, so lack of capacity is unlikely to be why 50K is worse than 10K for us.
- **Hypotheses:**
  - exp044 > exp040: 4× capacity helps on 50K episodes (higher mean success, lower training loss), but it does not make 50K better than 10K (exp039: 74.2%).
  - exp043 ≈ exp040: at the same parameter count, a wide shallow residual MLP is about as good as a narrow deep one (within the large checkpoint-to-checkpoint noise); if anything, wider is better, as in the blog and in the RL literature on MLP scaling.
- **Results (one seed each; success at 100K, ..., 1M steps; figure `flow_width_steps.png`):**
  - exp043 (1024 × 10, 10.4M): 62, 54, 52, 30, 68, 58, 70, 62, 48, 40% → mean 54.4% (std over checkpoints 11.9); training loss 0.0076 → 0.0058; validation loss ~0.0058.
  - exp044 (1024 × 40, 43.8M): 62, 58, 72, 66, 72, 66, 52, 62, 68, 42% → mean 62.0% (std 8.9); training loss 0.0072 → 0.0051 (lowest); validation loss ~0.0060.
  - exp040 (512 × 40, 11.7M): mean 59.4% (std 14.2); training loss 0.0077 → 0.0059. exp039 (512 × 40, 10K episodes): mean 74.2% (std 8.3), training loss 0.0031.
- **Observations:** at equal parameter count, wide-shallow and narrow-deep have the same training and validation loss and success within noise. 4× capacity fits the training data better and gives the most stable checkpoints, but only +2.6 points of mean success. The 1M-step checkpoints of all three 50K runs are low (36%, 40%, 42%). All three used seed 0, and the minibatch indices, $x_0$ and $t$ are drawn from the CUDA generator, which model initialization (on CPU) does not touch, so the three runs saw exactly the same sequence of minibatches and noise; with a constant lr, the most recent minibatches may move all three in the same direction (not verified).
- **Conclusion / next step:** exp043 hypothesis confirmed (shape does not matter at ~10M parameters). exp044 hypothesis weakly supported: capacity helps fit and stability, but the gain in success is within noise, and 50K episodes stays far below 10K (62% vs. 74%), so lack of capacity is not why more data is worse, as the blog's large models also suggest. The main limit on reading these curves is checkpoint-to-checkpoint noise; next, reduce it (lr decay to 0 at the end of training, EMA of the weights, more seeds / eval episodes) before comparing data sizes or methods further.

## exp040 seed 1 / exp045_flow_long_50k_cos_ema (seeds 0, 1, 2)
- **Date:** 2026-10-13
- **Compared with:** exp040 seed 0 (512 × 40, 50K episodes, constant lr 5e-4, no EMA: mean 59.4%, std over checkpoints 14.2, 36% at 1M) and exp043 / exp044 seed 0 (same minibatch order).
- **Change:**
  - exp040 seed 1: the same configuration with another seed, i.e. another minibatch order, noise and initialization.
  - exp045: exp040 + cosine lr decay from 5e-4 to 0 over the 1M steps (`train.lr_schedule: cosine`) + EMA of the weights with decay 0.9999 (~10K-step window; validation, kept checkpoints and evaluation use the EMA weights). Two changes at once, as asked; separating them would need two more runs. Seeds 0, 1, 2; seed 0 shares the minibatch order with exp040 / exp043 / exp044 seed 0.
  - Code (speed only): the EMA update is now one fused `torch._foreach_lerp_` instead of a per-tensor loop (checked equal).
- **Purpose:** test whether the checkpoint-to-checkpoint swings (and the common drop of A/B/C to 36–42% at 1M) come from the most recent minibatches under a constant lr.
- **Hypotheses:**
  - exp040 seed 1 does not follow seed 0's ups and downs (its curve is uncorrelated with exp040/043/044 seed 0) and is not specifically low at 1M; mean success similar (~60%).
  - exp045 is much smoother: std over checkpoints well below exp040's 14.2, close to the ±7-point evaluation noise of 50 episodes; success rises toward the end as the lr goes to 0; seed 0 does not drop at 1M although it saw the same last minibatches as A/B/C; the three seeds agree within ~10 points at 1M; mean over the last checkpoints at or above exp040's 59%.
