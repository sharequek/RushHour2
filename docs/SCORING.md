NYC Listings Scoring (0–100)

Overview
- Purpose: rank listings for your personal criteria with a simple, explainable score in [0,100].
- Core factors (weights): Commute (0.40), Price (0.35), Sqft (0.25).
- Calibration: recomputed each scoring run from the live dataset using Postgres percentiles.
- Execution: runs once per full scrape, after OCR, and updates `listings.score`.

Calibration
- Quantiles computed over non-null, >0 values:
  - Price: percentile_cont on `COALESCE(effective_price, price)` → [p10, p25, p50, p75, p90].
  - Commute: percentile_cont on `COALESCE(commute_duration_google_min, commute_duration_heur_min)`.
  - Sqft: CASE source: `sqft` else `ocr_sqft_extracted` if `ocr_sqft_confidence >= conf_min`.
- Small‑N fallback: if spans are degenerate, fall back to config floors.

Normalization
- Price (lower is better):
  - Anchors: config price filter (2000–4500) with dataset guardrails.
  - Bounds: lower=min(p10, 2000), upper=max(p75, 4500), soft tail to `hard_zero_at` (default 5000).
  - Scoring: 100 at/below lower, 0 at/above `hard_zero_at`, linear in between.
- Commute (lower is better):
  - Bounds: ideal=min(q10, 20), max=35 (hard cap). 100 at/below ideal, 0 at/above 35.
- Sqft (higher is better, OCR-aware):
  - Bounds: q10–q75 of trusted sqft; fallback to 350–750 if span < 50.
  - OCR weighting: if using OCR and conf in [conf_min, conf_full), multiply by `ocr_partial_weight`; if conf < conf_min, treat as missing. Ignore OCR sizes <250 or >2000 as artifacts.
- Combine: weighted average (with per‑listing weight redistribution for missing factors). Clamp [0,100], round to int.

Nudges (Additive)
- Broker fee: −8 if `has_broker_fee` or `broker_fee_pct > 0`.
- Application fee: −2 if `has_application_fee`.
- Availability now: +3 if `availability_now`.
- Days on market: −2 (>45), −4 (>75), −6 (>120) total.
- Rent stabilized: +3 if `is_rent_stabilized`.
- Amenities (cap +10 total):
  - +3 in‑unit laundry (has_laundry_in_unit or has_washer_dryer)
  - +1 dishwasher, +1 elevator, +1 gym
  - +1 any outdoor (balcony, terrace, private outdoor, roof deck)

Missing Data
- If a factor is missing/invalid, drop its weight and redistribute remaining weights for that listing.
- If all factors are missing, use `default_factor_score` (50 by default).

Configuration (config/config.json)
- scoring.enabled: turn scoring on/off.
- scoring.weights: { commute, price, sqft } (should sum to ~1, but are normalized per listing).
- scoring.price.hard_zero_at: soft tail target for price to reach score 0 (default 5000).
- scoring.commute: { ideal_min: 20, max_tolerable: 35 }.
- scoring.sqft: { ocr_confidence_min: 0.40, ocr_confidence_full: 0.75, ocr_partial_weight: 0.70, fallback_min: 350, fallback_max: 750 }.
- scoring.nudges: amenity bonus map, fees, rent stabilization, days_on_market thresholds.
- scoring.missing_values: { redistribute_weights: true, default_factor_score: 50 }.

Integration
- Call site: after OCR and before the script summary in `src/scrape.py`.
- Function: `await score_all_listings(pool, config)` from `src/scoring.py`.

Logging
- Calibration snapshot: chosen bounds for price/commute/sqft.
- Completion summary: number of rows updated, how many used OCR sqft, and how many lacked commute.

Notes
- Price and sqft are correlated; weights and a capped amenity bonus keep the model explainable and stable.
- All mapping functions are linear and clipped to remain intuitive and predictable.

