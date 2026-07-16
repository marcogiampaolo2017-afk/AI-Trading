# Pull Request

## Related Issue
Closes #<!-- issue number -->

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature / enhancement
- [ ] Filter threshold adjustment
- [ ] Model / training change
- [ ] Documentation update
- [ ] Refactor (no functional change)

## Summary of Changes
<!-- What did you change and why? -->

## Filter Impact (if applicable)
| Filter | Before | After |
|---|---|---|
| LAZARUS threshold | `___` | `___` |
| VSA threshold | `___` | `___` |
| META_DIARIA | `___` | `___` |

## Testing
- [ ] `pytest tests/ -v` passes (all 67 tests green)
- [ ] Bot ran in demo for ≥ 1 hour without exceptions
- [ ] Last 20 log lines show no unintended blocks

## Log Evidence (demo run)
```
[HH:MM:SS] 🔍 ESTADO: Meta=🟢libre | Día=YYYY-MM-DD | Equity=XXXXX.XX€
```

## Critical Conventions Checklist
- [ ] No hardcoded values — all constants via `config.py`
- [ ] No future data used in indicators
- [ ] `regime_proxy` still in `feature_cols`
- [ ] `ALLOW_MANUAL_CLOSE = False` unchanged (unless intentional)
- [ ] Comments/docstrings in Spanish
