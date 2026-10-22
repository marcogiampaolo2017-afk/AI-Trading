# Model Card — TITANY AI EURUSD H1 (RecurrentPPO)

Baseline model produced by the 3-phase training run of 2026-10-21/22.
**Status: trained and evaluated, NOT approved for live trading** — see [Verdict](#verdict).

## Artifacts

Binary artifacts are intentionally **not** versioned in Git (see `.gitignore`: `models/*.zip`, `models/*.pkl`).
Distribute the model and the normalizer together: a mismatched pair silently distorts every observation.

| File | Size (bytes) | SHA-256 | Role |
|---|---|---|---|
| `models/model_eurusd_titany_v12_sniper.zip` | 24,264,488 | `9a4b619df48fc8608adcecc6d7b720ae2614d3b93d5ca5e059c9cc1f8da303da` | Policy weights (RecurrentPPO / MlpLstmPolicy) |
| `models/vec_normalize.pkl` | 46,480 | `b6df84d7e23377f49c318ff9436180bc8253f602625b5a06bd472299fd81251b` | Observation normalizer, `(30, 30)` obs, 3 actions |
| `models/best_models/best_model.zip` | 24,264,461 | — | EvalCallback "best" @ 420k steps. Also backtested: **-31.1% in 168 bars, 8 trades, 0% win rate** — worse than the final model, which confirms the selection was noise. |

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

| Metric | Value |
|---|---|
| Initial / final equity | $110.00 → $76.02 |
| Total return | **-30.89%** |
| Max drawdown | 48.77% (measured from the $146 equity peak) |
| Trades | 125 |
| Win rate | **22.4%** (break-even at 3:1 R:R is 25%) |
| Avg win / avg loss | +110.3 / -35.2 pips |
| Profit factor | 0.90 |

The run stopped at bar ~4,168 of 19,886 because the equity circuit breaker
(`MAX_DRAWDOWN_FRACTION = 0.30`) fired — exactly what would happen on a live account.

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
