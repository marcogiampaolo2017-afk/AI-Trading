# Model Card — TITANY AI EURUSD H1 (RecurrentPPO)

Baseline model produced by the 3-phase training run of 2026-10-21/22.
**Status: trained and evaluated, NOT approved for live trading** — see [Verdict](#verdict).

## Artifacts

Binary artifacts are intentionally **not** versioned in Git (see `.gitignore`: `models/*.zip`, `models/*.pkl`).
Distribute the model and the normalizer together: a mismatched pair silently distorts every observation.

| File | Size (bytes) | SHA-256 | Role |
|---|---|---|---|
| `models/model_eurusd_titany_v12_sniper.zip` | 24,264,488 | `9a4b619df48fc8608adcecc6d7b720ae2614d3b93d5ca5e059c9cc1f8da303da` | Policy weights (RecurrentPPO / MlpLstmPolicy) |
| `models/vec_normalize.pkl` | 46,480 | `121c088867428c0c1bc6ec2a2e56d54163067169874139151e906af7b76ecfad` | Observation normalizer, `(30, 30)` obs, 3 actions. Restored from the last checkpoint (`titany_ckpt_vecnormalize_2999952_steps.pkl`) — see note below. |
| `models/best_models/best_model.zip` | 24,264,461 | — | EvalCallback "best" @ 420k steps. Also backtested: **-31.1% in 168 bars, 8 trades, 0% win rate** — worse than the final model, which confirms the selection was noise. |

> **Normalizer provenance.** When the generated artifacts were untracked (commit `7543ce7`),
> the fast-forward merge to `main` removed the final `vec_normalize.pkl` from the working tree —
> untracking a file only keeps it on disk while no branch switch touches it.
> The file above was restored from the last checkpoint (`models/checkpoints/titany_ckpt_vecnormalize_2999952_steps.pkl`,
> 2,999,952 steps vs the 3,013,632 of the final save): same `(30, 30)` space and statistically
> equivalent, but not byte-identical to the lost pickle. The trade list is sensitive to this
> difference (see *Normalizer sensitivity* below), so comparisons with a freshly retrained pair
> should be made pair-by-pair.

## Training run

| Item | Value |
|---|---|
| Command | `python research/train_agent.py --headless` |
| Dates | 2026-10-21 22:36 → 2026-10-22 13:57 (America/Asuncion) |
| Steps | 3,000,000 target / 3,013,632 performed |
| Phases | 1M exploration → 1.2M refinement → 800k hardening |
| Parallel envs | 6 `SubprocVecEnv`, CPU only |
| Data | `data/EURUSD_H1_201005121400_202605181400.csv` |
| Split | walk-forward window 2: train 79,536 / test 19,886 H1 bars |
| Features | 27 (correlation filter dropped 0) + 3 state = obs `(30, 30)` |
| Actions | `Discrete(3)` = HOLD, OPEN-short, OPEN-long (`SL_TP_PAIRS=[(20, 60)]`, `ALLOW_MANUAL_CLOSE=False`) |
| Policy | LSTM 128×2 (critic LSTM on), pi `[256, 256]`, vf `[512, 512, 512]`, Tanh |
| PPO | n_steps 1536, batch 384, γ 0.995, lr 3e-5 linear, clip 0.15, target_kl 0.03, n_epochs 5, ent_coef 0.05 → 0.03 → 0.02 |
| Normalizer | legacy `(30, 29)` / 20-action pickle rejected (space mismatch) → a new one was created |
| Wall time | ~15 h 20 min (≈54 fps including evaluation) |

## Out-of-sample backtest

`python research/test_agent.py --headless` on the last 20% of the data (never seen in training,
teacher mode off, 1 pip spread, 0.2 pip slippage, 30% drawdown breaker):

Measured with the artifact pair as it exists today (model + restored normalizer):

| Metric | Value (restored pair) | Value (original final pair, lost) |
|---|---|---|
| Initial / final equity | $110.00 → $76.66 | $110.00 → $76.02 |
| Total return | **-30.31%** | -30.89% |
| Max drawdown | 43.12% | 48.77% |
| Trades | 27 | 125 |
| Win rate | **18.5%** (break-even at 3:1 R:R is 25%) | 22.4% |
| Avg win / avg loss | +138.2 / -46.6 pips | +110.3 / -35.2 pips |
| Profit factor | 0.67 | 0.90 |

Both runs stopped early because the equity circuit breaker
(`MAX_DRAWDOWN_FRACTION = 0.30`) fired — exactly what would happen on a live account.

> **Normalizer sensitivity.** A ~0.5% difference in the observation statistics (13.7k steps of
> running RMS) turns 125 trades into 27 and profit factor 0.90 into 0.67. The policy is brittle
> with respect to normalization — another reason not to run it live, and a reason to store the
> model+normalizer as an immutable pair.

## Verdict

**Not deployable.** The policy does not clear the 25% win rate that a 3:1 R:R requires, and the
1-pip round-trip cost turns the remaining edge negative. `ep_rew_mean` stayed negative for the whole
run (-2.6k → -1.5k per episode) and the evaluation history is non-monotonic.

Keep it as a **reproducible baseline** to compare future training runs against, not as a trading model.

## Issues found while evaluating

1. **Eval episodes were unbounded** (fixed on this branch). An evaluation episode ran until the
   drawdown breaker tripped or the test set ended, so `EvalCallback` compared total rewards of
   episodes with different lengths; `best_model.zip` was selected essentially at random.
   Now: `episode_max_steps=2000` and `n_eval_episodes=3`.
2. **Train/eval objective mismatch** (open). The training environment runs with
   `teacher_mode=True` (filter-based shaping rewards) while `test_agent.py` evaluates with
   `teacher_mode=False`. The policy may be optimising part of the shaping signal instead of P&L.
3. **Single seed** (open). None of the numbers above has a confidence interval.
4. Progress chart title/filename hardcoded "600k" (fixed on this branch).

## Reward audit (2026-10-22)

`python research/reward_audit.py --bars 3000` decomposes the environment reward term by term
(`env.step()` now returns `info["reward_breakdown"]`) on a 3,000-bar out-of-sample window:

| Policy | teacher | Trades | Equity | P&L terms | Shaping terms | Total |
|---|---|---|---|---|---|---|
| hold   | off | 0  | 110.00 | 0.00    | -116.00    | -116.00 |
| random | off | 79 | 137.22 | 1,825.57 | -3,751.49 | -1,925.92 |
| model  | off | 88 | 110.98 | 1,326.15 | -4,911.11 | -3,584.97 |
| hold   | on  | 0  | 110.00 | 0.00    | -116.00    | -116.00 |
| random | on  | 80 | 111.55 | 1,365.05 | -12,060.87 | -10,695.82 |
| model  | on  | 88 | 117.98 | 1,443.96 | -7,498.18 | -6,054.22 |

Findings:

1. **The per-step drawdown penalty dominates.** `-(dd_ratio * 10)` is charged on every bar below
   the equity peak and accumulates -3.8k to -6.7k reward units over 3k bars — 2-5x the P&L terms.
2. **`teacher_mode` doubles the noise**: its +-5/+-10/+-15 bonuses add another -3k (model) / -5.4k
   (random).
3. **Doing nothing wins the reward.** Flat `hold` collects only -116 (the `fer_block` term), so the
   reward-maximising policy is to not trade — even when trading was profitable (random made +$27 in
   the same window). That is exactly the behaviour the 3M model exhibits.
4. `realized` is not raw pips either: `_close_position` doubles TP hits and multiplies losses by
   1.5, adding another shaping layer on top.

Re-running the audit with the drawdown penalty disabled (`--dd-penalty-scale 0`) isolates the two
effects:

| dd_scale | teacher | policy | Trades | Equity | P&L terms | Shaping terms | Total |
|---|---|---|---|---|---|---|---|
| 1.0 | off | random | 79 | 137.22 | +1,825.6 | -3,751.5 | -1,925.9 |
| 0.0 | off | random | 77 | 125.80 | +1,549.0 | **+6.8** | **+1,555.8** |
| 0.0 | off | model  | 83 | 101.41 | +1,064.9 | -14.6 | +1,050.3 |
| 0.0 | on  | random | 70 |  76.77 | +579.2 | -4,993.4 | -4,414.2 |
| 0.0 | on  | model  | 53 |  76.97 | +279.6 | -3,187.6 | -2,908.0 |

So the two fixes are independent and both required: disabling the drawdown penalty makes P&L the
dominant term (+1.5k vs +7 shaping), while `teacher_mode` alone still injects -3k to -5k of shaping.

Conclusion: the training objective is not P&L, and more steps cannot fix that. The audit tooling and
the diagnostic flags (`--steps`, `--teacher-mode`, `--dd-penalty-scale`, `--seed`, `--out-dir`,
`--fresh`) are in place to measure each fix in isolation.

## Diagnostic runs (500k steps)

With the reward audit in hand, two short runs isolate each fix. Same data split, seed 42, bounded
evaluation episodes, 6 workers, isolated output dirs under `models/diagnostics/`.

| Run | teacher | dd_scale | Steps | Wall time | Output |
|---|---|---|---|---|---|
| A | off | 1.0 | 516,096 | 2 h 07 min | `models/diagnostics/teacher_off/` |
| B | off | 0.0 | 516,096 | ~2 h | `models/diagnostics/teacher_off_dd0/` |

### Evaluation curves (3 bounded episodes every 60k steps)

| Run | 60k | 120k | 180k | 240k | 300k | 360k | 420k | 480k |
|---|---|---|---|---|---|---|---|---|
| A | -922 | -1,193 | -3,431 | -1,822 | +157 | -2,706 | -478 | **+1,537** |
| B | -484 | +74 | -327 | -214 | +1,340 | +1,232 | +1,299 | **+2,553** |

Both end positive, which the 3M reference run never did. But recall that a positive eval reward is
not profitability (see the B result below).

### Out-of-sample backtests, two disjoint 3,000-bar windows

| Training config | Window 1 | Window 2 | Mean | PF (w1 / w2) |
|---|---|---|---|---|
| 3M baseline (teacher on, dd on) | -30.03% | -31.83% | **-30.9%** | 0.82 / 0.68 |
| A — 500k, teacher off, dd on | **+42.52%** | **-4.03%** | **+19.2%** | 1.24 / 0.97 |
| B — 500k, teacher off, dd off | +30.29% | -29.81% | +0.2% | 1.17 / 0.47 |

Window details for the two diagnostic runs: A ran the full 2,999 bars in both windows (no drawdown
breaker trip), max DD 32.65% / 24.86%, win rate 27.1% / 26.0%. B ran 2,999 bars in window 1 but was
stopped by the drawdown breaker after 967 bars in window 2 (max DD 36.81%, win rate 12.5%).

### What this changes for the next long run

1. **Turn teacher shaping off.** Removing it is worth more than 5x the training budget of the
   reference run: both diagnostic arms beat the 3M baseline on both windows.
2. **Keep the drawdown penalty.** It dominates the reward decomposition, but it also acts as risk
   shaping: removing it makes the policy bimodal (+30% / -30%). A's config (teacher off, dd on) is
   the robust one.
3. **Do not trust the eval reward as a profitability proxy.** B ended with the best eval mean
   (+2,553) and then lost 30% on window 2.
4. Next long run: 3M steps, teacher off, dd on, seed replication (3 seeds), evaluated on at least two
   windows. Only then consider the live terminal.

Caveats: 3,000-bar windows (~4 months of H1 each), single seed per arm, and full 19,886-bar
backtests are pending (the host CPU was shared with a game during the comparison).
## Recommended plan for the next run

Nothing below is implemented yet; it is the shortlist to make the next run comparable and cheaper.

1. **Align objective and metric.** Train with the same reward the backtest measures (`teacher_mode=False`),
   or keep the shaping and evaluate with it too. Measure the delta before adding more compute.
2. **Audit the reward terms.** Log the per-term contributions (spread/slippage cost, time penalty,
   `unrealized_delta_weight`, teacher bonuses) for a fixed policy; if shaping dominates P&L the
   agent is optimising the wrong thing.
3. **More steps, cheap evals.** With bounded eval episodes (`episode_max_steps=2000`) the same machine
   should fit 6-10M steps in ~15 h. Keep 6-7 workers.
4. **Hyperparameter pass.** lr 1e-4 with cosine decay, `ent_coef=0.01`, `gamma=0.99`, `n_steps=2048`.
5. **Risk sweep as separate runs.** `SL_TP_PAIRS` defines the action space: (20, 60) vs (20, 40) vs (50, 100)
   are three different projects, not a hyperparameter. Do not mix them in one run.
6. **Seeds.** At least 3 runs; report mean ± std of the OOS backtest, not a single number.
7. **Walk-forward.** `train_agent.py` already defines 4 windows; report all folds instead of only window 2.

## Reproduce

```bash
python research/train_agent.py --headless     # ~15 h on this 8-vCPU machine, 3M steps
python research/test_agent.py --headless      # OOS backtest -> reports/
```

## Load the pair (live / inference)

```python
import config
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize

# The normalizer must be loaded with its matching env (same obs space).
model = RecurrentPPO.load("models/model_eurusd_titany_v12_sniper.zip", device="cpu")
vec   = VecNormalize.load("models/vec_normalize.pkl", env)   # env: ForexTradingEnv / DummyVecEnv
vec.training = False
vec.norm_reward = False
```

`core/live_trading_system.py` and `core/TITANY_AI_Terminal_Pro.py` already look for this exact
pair; see `config.py` for the model paths.
