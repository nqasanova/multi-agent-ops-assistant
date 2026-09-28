# Evaluation results

Provider: `offline`, 24 cases, 0.2s total.

| Metric | Result |
|---|---|
| Routing accuracy | 23/24 (96%) |
| Doc recall (expected document retrieved) | 14/15 (93%) |
| Answer correct (phrases + reference SQL values) | 22/24 (92%) |
| Passed grounding review | 24/24 (100%) |

| Case | Routing | Doc recall | Correct | Grounded |
|---|---|---|---|---|
| sla-helios | yes | yes | yes | yes |
| sla-lumen | no | no | no | yes |
| refund-window-mistral | yes | yes | yes | yes |
| express-vantara | yes | yes | yes | yes |
| escalation-nordlicht | yes | yes | yes | yes |
| open-by-priority | yes | n/a | yes | yes |
| open-by-tier | yes | n/a | yes | yes |
| top3-2025 | yes | n/a | yes | yes |
| corvid-2025 | yes | n/a | yes | yes |
| isar-refunded | yes | n/a | yes | yes |
| orders-status | yes | n/a | yes | yes |
| revenue-tier | yes | n/a | yes | yes |
| tessera-tickets | yes | n/a | yes | yes |
| refund-approval | yes | yes | yes | yes |
| refund-payout | yes | yes | yes | yes |
| damaged | yes | yes | yes | yes |
| shipping-france | yes | yes | yes | yes |
| shipping-cost | yes | yes | no | yes |
| breach | yes | yes | yes | yes |
| sev1 | yes | yes | yes | yes |
| ai-tools | yes | yes | yes | yes |
| prod-access | yes | yes | yes | yes |
| security-training | yes | yes | yes | yes |
| out-of-scope | yes | n/a | yes | yes |
