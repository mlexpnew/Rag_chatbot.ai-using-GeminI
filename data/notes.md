# Ops Notes – Fulfillment, Cold Chain & SLAs (Amazon-Style)

## Service Levels
- Same-city delivery ETA: ≤ 45 min (instant) or ≤ 1 day (standard).
- First response from support: ≤ 15 min during business hours.
- Ticket resolution targets:
  - P1 (order not received / wrong item): < 24 h
  - P2 (delayed / partial): < 48 h
  - P3 (general queries): < 72 h

## Cold-Chain Items
- Pack to maintain 2–8°C for up to 6 hours.
- Use gel packs + insulated shippers; include temperature indicator where applicable.
- If temperature breach occurs:
  1) Quarantine item; do not deliver.
  2) Trigger replacement dispatch.
  3) Log incident in QA register; notify vendor if recurring.

## Inventory & Substitutions
- If stockout at packing:
  - Suggest closest substitute (equal or higher value).
  - If customer declines, offer refund or credit (₹100 or 10%, whichever is higher).

## Delivery Exception Playbook
1) Rider delay > 20 min vs. ETA → proactively notify with updated ETA and apology credit if applicable.
2) Address unreachable → attempt 3 calls + 1 message; hold for 24 h; auto-cancel with full refund if perishable.
3) Damaged on arrival → collect photo proof; schedule replacement or refund immediately.

## Returns & QC
- Photo/video evidence required for DOA or damaged claims within 24 h of delivery.
- Returned items inspected within 48 h; initiate refund within 24 h post-inspection.

## Compliance
- PII handling: mask customer phone/email in third-party tools; rotate access keys quarterly.
- Courier partners must maintain on-time delivery ≥ 95% and damage rate ≤ 0.5%.
