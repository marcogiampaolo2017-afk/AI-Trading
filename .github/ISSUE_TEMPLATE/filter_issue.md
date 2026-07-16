---
name: Filter / Protection Issue
about: A protection filter is blocking valid trades or allowing bad ones
title: "[FILTER] "
labels: filter
assignees: marcogiampaolo2017-afk
---

## Which Filter?
- [ ] LAZARUS (FER threshold)
- [ ] VSA (volume filter)
- [ ] Anti-Parabólica
- [ ] FOMO dynamic
- [ ] Impulse Shield / Z10
- [ ] ICT / Vision 360
- [ ] Anti-Cohete
- [ ] Multiverso (Monte Carlo)
- [ ] Meta Diaria / TENDENCIA EXCEPCIONAL
- [ ] H4 Macro Override

## Problem
**Over-blocking** (good trades blocked) or **Under-blocking** (bad trades passed)?

## Market Context
| Param | Value |
|---|---|
| H4 gap | `___p` |
| FER | `___` |
| VSA | `___` |
| MA50 diff | `___p` |
| Direction | SELL / BUY |

## Log Lines
```
[HH:MM:SS] 🛡️ LAZARUS: FER=X.XX < X.XX (H4=XXp)
```

## Proposed Threshold Change
<!-- Example: "LAZARUS for H4>50p should be 0.03 not 0.04" -->
