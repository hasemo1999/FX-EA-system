# ATR Quantile Production Profile — 2025-08-15

- Profile: B (lo=0.30, hi=0.72), window=500, period=14
- Backup: #1 (0.30–0.71)
- Fallback: #2 (0.15–0.83)
- Config: `backtest_config_production.json`
- Gate order: session → MTF → quality → ATR-quantile (final gate)
- Counters: `pre_signal`, `atr_quantile_reject`, `enter`
- Metrics: `atrq_reject_rate`, `mtf_reject_rate`, PF/MaxDD/WF
- Friday close (UTC): 06:10 (JST 15:10)
- Slack: `slack_daily_summary.py` posts PF/MaxDD/WF/Trades/atrq

Validation snapshot (2023-2024):
- B: PF≈1.246, MaxDD≈-10.93, WF≈83.33%, Trades≈181, atrq_reject_rate≈0.593
- #1: PF≈1.241, MaxDD≈-10.93, WF≈83.33%, Trades≈184, atrq_reject_rate≈0.603
- #2: PF≈1.221, MaxDD≈-15.05, WF≈83.33%, Trades≈191 (rejected)

Runbook
- Daily Slack: `python slack_daily_summary.py --webhook <URL>`
- Watch:
  - atrq_reject_rate > 0.65 → switch to #2
  - MaxDD < -10.93 → reconsider (#1 or #2)
  - WF < 75% (weekly) → re-optimize
- Tweaks:
  - If samples low/WF drops → relax MTF thresholds by 5–10%
  - Regime response → try atr_quantile_window 300 / 1000
