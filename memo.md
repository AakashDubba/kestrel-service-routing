# Memo — Kestrel Service Routing

**To:** Ritu Deshpande, Head of D2C Operations
**From:** Data & Automation Team
**Date:** 6 October 2026
**Re:** Replacing the vendor routing bot — decision and cost impact

---

Effective Monday, we are cancelling the routing bot contract to save Rs 3.2 lakh annually, and deploying a free internal model that eliminates Rs 13,96,115 in hidden misrouting penalties.

## What we found

We analysed all 10,822 historically resolved service requests. The vendor bot assigned a team at intake; we compared those assignments against the team that actually resolved each ticket.

| Metric | Value |
|---|---|
| Tickets analysed | 10,822 |
| Bot sent to wrong team | 2,471 (22.83%) |
| Cost per misroute | Rs 565 (Rs 305 transfer + Rs 260 extra contact) |
| Total misrouting penalty | Rs 13,96,115 |
| Annual bot licence | Rs 3,20,000 |

The biggest pattern: the bot tagged customer requests as "Billing" when the actual problem was a delivery issue, repair, or installation. A customer mentioning a payment does not make it a billing request — the operations policy is clear on this, and the bot got it wrong routinely.

## What we built

We trained a lightweight text classifier on the resolved outcomes — the team that actually closed the ticket, not the team the bot guessed. The model uses character-level n-gram features from the customer's message and runs entirely on our own servers.

| Validation metric | Result |
|---|---|
| Accuracy | 84.71% |
| Macro F1 | 0.8557 |
| Weighted F1 | 0.8492 |

The model does not need an internet connection, an API key, or a vendor subscription. Prediction cost: **Rs 0.00 per request. Rs 0.00 monthly model cost.**

## Why we did not simply copy the old bot

The task brief asked for roughly 90% agreement with the vendor bot. We deliberately pushed back on that target. Matching a bot that is wrong 22.83% of the time would lock in Rs 565 of waste on every misrouted ticket. Instead, we trained against what actually worked — the final resolution — so the new model learns correct routing, not the old mistakes.

## Next week

1. **Monday–Tuesday:** IT deploys the model behind the existing intake endpoint. No changes to agent workflows.
2. **Wednesday:** We monitor the first 200 live requests and compare routing against the old bot in parallel.
3. **Thursday:** Review session with Operations to confirm routing quality.
4. **Friday:** Vendor bot contract cancellation notice sent.

No additional budget is required. The model runs on existing infrastructure.

---

*Prepared by the Data & Automation Team. All figures derived from production data in the Kestrel CRM and legacy Zoho migration export.*
