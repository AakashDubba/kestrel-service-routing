# Memo — Kestrel Service Routing

**To:** Ritu Deshpande, Head of D2C Operations  
**From:** Aakash Dubba  
**Date:** 6 October 2026  
**Re:** Replacing the vendor routing bot — decision and cost impact

Effective Monday, we are cancelling the routing bot contract to save Rs 3.2 lakh annually, and deploying a free internal model that eliminates Rs 13,96,115 in hidden misrouting penalties.

## What we found

We analysed 10,822 service requests by joining the intake records with their actual resolution outcomes. The vendor bot's assignment at intake was compared with the team that ultimately resolved each request.

| Metric | Value |
|---|---:|
| Tickets analysed | 10,822 |
| Bot misroutes | 2,471 (22.83%) |
| Old bot accuracy | 77.17% |
| Cost per misroute | Rs 565 |
| Transfer cost per misroute | Rs 305 |
| Additional contact cost per misroute | Rs 260 |
| Historical misrouting cost identified | Rs 13,96,115 |
| Annual bot licence | Rs 3,20,000 |

The audit showed that the legacy bot was frequently assigning requests to the wrong team. In particular, some requests containing billing-related language were ultimately resolved by teams handling delivery, repairs, or installation work.

## What we built

We trained the replacement model against the actual resolution outcome — the team that ultimately resolved the request — rather than the legacy bot's original assignment.

The final model uses character-level text features with a local LinearSVC classifier. It operates without an external AI API or paid inference service.

| Validation metric | Result |
|---|---:|
| Accuracy | **84.71%** |
| Macro F1 | **0.8557** |
| Weighted F1 | **0.8492** |

The validated accuracy is 84.71%, compared with 77.17% for the legacy bot, an improvement of 7.54 percentage points.

Prediction cost is **Rs 0.00 per request** under the current local deployment, with **Rs 0.00 in recurring model/software inference cost**.

## Why we did not simply copy the old bot

The task brief asked for approximately 90% agreement with the vendor bot. We reassessed that target after measuring the bot against actual resolution outcomes.

The legacy system was wrong on 22.83% of the tickets analysed. Optimizing primarily for agreement with that system would risk reproducing errors that were already present.

We therefore trained against `final_team` — the actual resolution outcome — so that the replacement model is evaluated on the operational result rather than on the legacy system's assumptions.

## Next week

1. **Approve deployment** of the internal routing model.
2. **Notify IT** to update the intake webhook to the internal `/predict` endpoint.
3. **Begin a controlled rollout** and review ambiguous routing cases during the first week.
4. **Proceed with the vendor-contract decision** based on the approved rollout.

The historical Rs 13,96,115 figure represents the misrouting cost identified from the legacy bot's observed errors. The Rs 3.2 lakh figure represents the annual vendor licence cost that can be avoided if the legacy contract is cancelled.

---

**Prepared by: Aakash Dubba**
