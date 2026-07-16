# ============================================================
# setup_github_labels.ps1
# Crea labels, issues y milestone organizados en el repo.
# Uso: .\setup_github_labels.ps1 -Token "ghp_xxxxxxxxxxxx"
# ============================================================
param(
    [Parameter(Mandatory=$true)]
    [string]$Token
)

$Repo  = "marcogiampaolo2017-afk/AI-Trading"
$Api   = "https://api.github.com"
$Headers = @{
    Authorization = "Bearer $Token"
    Accept        = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}

function Invoke-GH($Method, $Path, $Body = $null) {
    $params = @{ Method = $Method; Uri = "$Api$Path"; Headers = $Headers }
    if ($Body) { $params.Body = ($Body | ConvertTo-Json -Depth 5) }
    try { Invoke-RestMethod @params } catch { Write-Warning "  SKIP: $($_.Exception.Message)" }
}

# ── 1. LABELS ────────────────────────────────────────────────
Write-Host "`n=== Creating Labels ===" -ForegroundColor Cyan
$labels = @(
    @{ name = "bug";             color = "d73a4a"; description = "Something is broken in production" }
    @{ name = "enhancement";     color = "a2eeef"; description = "New feature or improvement" }
    @{ name = "filter";          color = "f9d71c"; description = "Trading filter / threshold change" }
    @{ name = "model";           color = "e4e669"; description = "AI/LSTM model or training related" }
    @{ name = "protection";      color = "0075ca"; description = "Risk management / meta diaria" }
    @{ name = "documentation";   color = "0075ca"; description = "README, AGENTS.md, comments" }
    @{ name = "priority: high";  color = "b60205"; description = "Blocking live trading" }
    @{ name = "priority: medium";color = "e4b63d"; description = "Important but not blocking" }
    @{ name = "priority: low";   color = "cfd3d7"; description = "Nice to have" }
    @{ name = "good first issue";color = "7057ff"; description = "Simple, well-scoped task for newcomers" }
    @{ name = "help wanted";     color = "008672"; description = "Extra attention is needed" }
    @{ name = "wontfix";         color = "ffffff"; description = "This will not be worked on" }
    @{ name = "in progress";     color = "fbca04"; description = "Currently being worked on" }
    @{ name = "needs review";    color = "ee0701"; description = "PR ready for review" }
)

foreach ($label in $labels) {
    Write-Host "  + $($label.name)"
    Invoke-GH "POST" "/repos/$Repo/labels" $label | Out-Null
}

# ── 2. MILESTONE ─────────────────────────────────────────────
Write-Host "`n=== Creating Milestone ===" -ForegroundColor Cyan
$milestone = @{
    title       = "V14 — Stable Demo Profitability"
    description = "Target: consistent +20€/month on AvaTrade demo. Bot must run 4 weeks without crashes with positive weekly PnL."
    due_on      = "2026-08-31T00:00:00Z"
}
$ms = Invoke-GH "POST" "/repos/$Repo/milestones" $milestone
$msNumber = $ms.number
Write-Host "  Milestone created: #$msNumber"

# ── 3. ISSUES ────────────────────────────────────────────────
Write-Host "`n=== Creating Issues ===" -ForegroundColor Cyan
$issues = @(
    @{
        title = "[FILTER] LAZARUS blocks all SELLs during 05:00-08:00 CEST (Asian close / pre-London)"
        body  = "During the pre-London gap (05:00-08:00 CEST) FER drops to 0.00-0.03 while H4 maintains a strong directional bias. The adaptive threshold (0.04 for H4>50p) still blocks. Consider adding a time-of-day exemption for known low-FER windows when H4 gap > 60p."
        labels = @("filter", "priority: medium")
    }
    @{
        title = "[FEAT] Session filter — skip Asian session (22:00-07:00 CEST)"
        body  = "EUR/USD Asian session has structurally low volume and FER. Instead of lowering thresholds to compensate, add a trading-hours gate that restricts entries to London+NY sessions (07:00-20:00 CEST). This would eliminate most LAZARUS blocks without touching thresholds."
        labels = @("enhancement", "filter", "priority: medium")
    }
    @{
        title = "[FEAT] Train V14 on 2025-2026 EURUSD data with updated action space"
        body  = "V13 was trained on 2010-2025 data. The 2025-2026 period includes a strong USD weakening trend. Retrain V14 with: updated CSV, teacher_mode penalties for SPRING signals before SELL, higher chaos_penalty for H4>50p counter-trend entries."
        labels = @("model", "enhancement", "priority: medium")
    }
    @{
        title = "[BUG] Shadow Trainer can inject corrupted V13 if CSV has malformed obs rows"
        body  = "If `market_experience.csv` contains rows where the `obs` field has a different number of elements than expected (due to a mid-candle save or indicator change), the Shadow Trainer trains on corrupted observations. Add validation: skip rows where `len(obs.split('|')) % 30 != 0`."
        labels = @("bug", "priority: medium")
    }
    @{
        title = "[FEAT] Real account mode — proper Kelly for 100€ balance"
        body  = "Demo has ~10,000 USD which makes Kelly give lots of 0.22. Real account has 100€. Need a separate `REAL_ACCOUNT_MODE` flag in config.py that caps Kelly at `base_lot * 1.5` regardless of balance, to match 100€ risk constraints."
        labels = @("enhancement", "protection", "priority: high")
    }
    @{
        title = "[FEAT] Weekly performance report — auto-generated CSV + summary email"
        body  = "After each Friday 22:00 CEST, generate `reports/weekly_YYYY-WW.csv` with all closed trades and a summary: trades, win rate, gross PnL, max drawdown, Sharpe proxy. Optionally send via SMTP or Base44 notification."
        labels = @("enhancement", "documentation", "priority: low")
    }
    @{
        title = "[GOOD FIRST ISSUE] Add FER value to SYNC log line"
        body  = "The current SYNC log line shows: `Meta / Día / Equity / dd`. Add `FER=X.XX` and `H4=XXp` so the live status shows filter state at a glance without having to read separate LAZARUS messages. Change is in `core/TITANY_AI_Terminal_Pro.py` around the SYNC block."
        labels = @("good first issue", "documentation")
    }
    @{
        title = "[PROTECTION] TENDENCIA EXCEPCIONAL VSA requirement too low (1.1) — causes SL trades"
        body  = "The 3rd trade on 2026-07-13 opened via TENDENCIA EXCEPCIONAL with VSA=1.1 (minimum) during ALERTA SPRING (bullish reversal signals). The trade lost -3.87€. Consider raising VSA requirement to 1.5 for TENDENCIA trades, or blocking TENDENCIA when ALERTA SPRING fired in last 5 candles."
        labels = @("protection", "filter", "priority: high")
    }
)

foreach ($issue in $issues) {
    Write-Host "  + $($issue.title.Substring(0, [Math]::Min(60, $issue.title.Length)))..."
    $body = @{
        title     = $issue.title
        body      = $issue.body
        labels    = $issue.labels
        milestone = $msNumber
    }
    Invoke-GH "POST" "/repos/$Repo/issues" $body | Out-Null
}

Write-Host "`nDone! Visit https://github.com/$Repo/issues" -ForegroundColor Green
