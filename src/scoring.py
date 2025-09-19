import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import asyncpg


log = logging.getLogger("streeteasy")


# ---------- Helpers: config access with safe defaults ----------

def _get(d: Dict[str, Any], path: List[str], default: Any) -> Any:
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _weights(cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    w_cfg = _get(cfg, ["scoring", "weights"], {})
    w_commute = float(w_cfg.get("commute", 0.40))
    w_price = float(w_cfg.get("price", 0.35))
    w_sqft = float(w_cfg.get("sqft", 0.25))
    total = w_commute + w_price + w_sqft
    if total <= 0:
        return 0.40, 0.35, 0.25
    return w_commute, w_price, w_sqft


# ---------- Calibration queries ----------

async def _fetch_price_quantiles(conn: asyncpg.Connection) -> Optional[List[float]]:
    row = await conn.fetchrow(
        """
        SELECT percentile_cont(ARRAY[0.10,0.25,0.50,0.75,0.90])
        WITHIN GROUP (ORDER BY COALESCE(effective_price, price)) AS qs
        FROM listings
        WHERE COALESCE(effective_price, price) IS NOT NULL
          AND COALESCE(effective_price, price) > 0
        """
    )
    return list(row["qs"]) if row and row["qs"] else None


async def _fetch_commute_quantiles(conn: asyncpg.Connection) -> Optional[List[float]]:
    row = await conn.fetchrow(
        """
        SELECT percentile_cont(ARRAY[0.10,0.25,0.50,0.75,0.90])
        WITHIN GROUP (ORDER BY COALESCE(commute_duration_google_min, commute_duration_heur_min)) AS qs
        FROM listings
        WHERE COALESCE(commute_duration_google_min, commute_duration_heur_min) IS NOT NULL
          AND COALESCE(commute_duration_google_min, commute_duration_heur_min) > 0
        """
    )
    return list(row["qs"]) if row and row["qs"] else None


async def _fetch_sqft_quantiles(conn: asyncpg.Connection, conf_min: float) -> Optional[List[float]]:
    row = await conn.fetchrow(
        """
        WITH sqft_source AS (
          SELECT CASE
                   WHEN sqft IS NOT NULL AND sqft > 0 THEN sqft
                   WHEN ocr_sqft_extracted IS NOT NULL AND ocr_sqft_extracted > 0 AND ocr_sqft_confidence IS NOT NULL AND ocr_sqft_confidence >= $1
                     THEN ocr_sqft_extracted
                 END AS s
          FROM listings
        )
        SELECT percentile_cont(ARRAY[0.10,0.25,0.50,0.75,0.90])
        WITHIN GROUP (ORDER BY s) AS qs
        FROM sqft_source
        WHERE s IS NOT NULL
        """,
        conf_min,
    )
    return list(row["qs"]) if row and row["qs"] else None


# ---------- Scoring math ----------

def _lin_map(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if hi <= lo:
        return 50.0
    if invert:
        # lower is better: value==lo -> 100, value==hi -> 0
        if value <= lo:
            return 100.0
        if value >= hi:
            return 0.0
        return 100.0 * (hi - value) / (hi - lo)
    else:
        # higher is better: value==lo -> 0, value==hi -> 100
        if value <= lo:
            return 0.0
        if value >= hi:
            return 100.0
        return 100.0 * (value - lo) / (hi - lo)


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return hi if x > hi else lo if x < lo else x


def _choose_sqft(row: Dict[str, Any], conf_min: float) -> Tuple[Optional[int], Optional[float], bool]:
    # Returns (sqft_value, conf_used, used_ocr)
    s = row.get("sqft")
    if s is not None and s > 0:
        return int(s), None, False
    # fallback to OCR
    ocr_s = row.get("ocr_sqft_extracted")
    conf = row.get("ocr_sqft_confidence")
    if ocr_s is not None and ocr_s > 0 and conf is not None and float(conf) >= conf_min:
        return int(ocr_s), float(conf), True
    return None, None, False


def _amenity_bonus(row: Dict[str, Any], cap: int, nudge_cfg: Dict[str, Any]) -> int:
    bonus = 0
    # in-unit laundry
    if row.get("has_laundry_in_unit") or row.get("has_washer_dryer"):
        bonus += int(_get(nudge_cfg, ["amenities", "in_unit_laundry"], 3))
    # dishwasher
    if row.get("has_dishwasher"):
        bonus += int(_get(nudge_cfg, ["amenities", "dishwasher"], 1))
    # elevator
    if row.get("has_elevator"):
        bonus += int(_get(nudge_cfg, ["amenities", "elevator"], 1))
    # gym
    if row.get("has_gym"):
        bonus += int(_get(nudge_cfg, ["amenities", "gym"], 1))
    # outdoor any
    if row.get("has_private_outdoor") or row.get("has_balcony") or row.get("has_terrace") or row.get("has_roof_deck"):
        bonus += int(_get(nudge_cfg, ["amenities", "outdoor_any"], 1))
    if bonus > cap:
        bonus = cap
    return bonus


def _days_on_market_penalty(days: Optional[int], thresholds: List[Dict[str, int]]) -> int:
    if days is None:
        return 0
    penalty = 0
    for t in thresholds:
        try:
            if days >= int(t.get("min", 0)):
                penalty = max(penalty, int(t.get("penalty", 0)))
        except Exception:
            continue
    return penalty


async def score_all_listings(pool: asyncpg.Pool, config: Dict[str, Any]) -> None:
    import time
    start_time = time.time()
    
    if not _get(config, ["scoring", "enabled"], True):
        log.info({"event": "scoring_skipped", "reason": "disabled_in_config"})
        return

    w_commute, w_price, w_sqft = _weights(config)

    # Config anchors and thresholds
    price_filter = _get(config, ["scraper", "filters", "price"], [2000, 4500])
    cfg_price_min = float(price_filter[0]) if isinstance(price_filter, list) and len(price_filter) >= 1 else 2000.0
    cfg_price_max = float(price_filter[1]) if isinstance(price_filter, list) and len(price_filter) >= 2 else 4500.0
    hard_zero_at = float(_get(config, ["scoring", "price", "hard_zero_at"], 6000.0))

    commute_ideal_floor = float(_get(config, ["scoring", "commute", "ideal_min"], 20.0))
    commute_max = float(_get(config, ["scoring", "commute", "max_tolerable"], 35.0))  # hard cap per user

    conf_min = float(_get(config, ["scoring", "sqft", "ocr_confidence_min"], _get(config, ["ocr", "hybrid", "sqft_confidence_threshold"], 0.40)))
    conf_full = float(_get(config, ["scoring", "sqft", "ocr_confidence_full"], 0.75))
    partial_weight = float(_get(config, ["scoring", "sqft", "ocr_partial_weight"], 0.70))

    amen_cap = int(_get(config, ["scoring", "nudges", "amenities", "max_total"], 10))
    nudge_cfg = _get(config, ["scoring", "nudges"], {})
    default_factor_score = float(_get(config, ["scoring", "missing_values", "default_factor_score"], 50.0))

    # Fallback spans
    sqft_fallback_min = float(_get(config, ["scoring", "sqft", "fallback_min"], 350.0))
    sqft_fallback_max = float(_get(config, ["scoring", "sqft", "fallback_max"], 750.0))

    async with pool.acquire() as conn:
        # Calibration
        price_qs = await _fetch_price_quantiles(conn)
        commute_qs = await _fetch_commute_quantiles(conn)
        sqft_qs = await _fetch_sqft_quantiles(conn, conf_min)

        # Derive bounds with guardrails
        # Price bounds anchored to config with quantile guardrails
        if price_qs:
            p10, p25, p50, p75, p90 = price_qs
        else:
            p10 = cfg_price_min
            p75 = cfg_price_max
            p90 = max(cfg_price_max, cfg_price_min + 1000)

        price_lower = min(p10 or cfg_price_min, cfg_price_min)
        price_upper = max(p75 or cfg_price_max, cfg_price_max)
        if price_upper <= price_lower:
            price_lower, price_upper = cfg_price_min, cfg_price_max

        # Commute bounds (lower is better)
        if commute_qs:
            c10, _, _, c75, _ = commute_qs
            commute_ideal = min(c10, commute_ideal_floor)
            commute_max_bound = commute_max
            if commute_max_bound <= commute_ideal + 1:
                commute_max_bound = commute_ideal + 15
        else:
            commute_ideal = commute_ideal_floor
            commute_max_bound = commute_max

        # Sqft bounds (higher is better)
        if sqft_qs:
            s10, _, _, s75, _ = sqft_qs
            sqft_lo = s10
            sqft_hi = s75
            if sqft_hi is None or sqft_lo is None or sqft_hi - sqft_lo < 50:
                sqft_lo, sqft_hi = sqft_fallback_min, sqft_fallback_max
        else:
            sqft_lo, sqft_hi = sqft_fallback_min, sqft_fallback_max

        log.info({
            "event": "scoring_calibration",
            "price_bounds": {"lower": price_lower, "upper": price_upper, "hard_zero_at": hard_zero_at},
            "commute_bounds": {"ideal": commute_ideal, "max": commute_max_bound},
            "sqft_bounds": {"lo": sqft_lo, "hi": sqft_hi},
        })

        # Fetch all listings for scoring
        rows = await conn.fetch(
            """
            SELECT id,
                   price, effective_price, price_per_sqft,
                   commute_duration_google_min, commute_duration_heur_min,
                   sqft, ocr_sqft_extracted, ocr_sqft_confidence,
                   has_broker_fee, broker_fee_pct, has_application_fee,
                   availability_now, days_on_market, is_rent_stabilized,
                   has_laundry_in_unit, has_washer_dryer, has_dishwasher, has_elevator, has_gym,
                   has_private_outdoor, has_balcony, has_terrace, has_roof_deck
            FROM listings
            """
        )

        updates: List[Tuple[int, int]] = []
        missing_commute_count = 0
        used_ocr_count = 0

        for row in rows:
            r = dict(row)
            # Factor extraction
            p = r.get("effective_price") or r.get("price")
            d = r.get("commute_duration_google_min") or r.get("commute_duration_heur_min")
            s, s_conf, used_ocr = _choose_sqft(r, conf_min)
            if used_ocr:
                used_ocr_count += 1

            # Individual factor scores
            price_score: Optional[float]
            if p is None or p <= 0:
                price_score = None
            else:
                # anchored price: within [price_lower, price_upper] then soft tail to hard_zero_at
                if p <= price_lower:
                    price_score = 100.0
                elif p <= price_upper:
                    price_score = _lin_map(float(p), price_lower, price_upper, invert=True)
                elif p >= hard_zero_at:
                    price_score = 0.0
                else:
                    # linear decay from value at price_upper down to 0 at hard_zero_at
                    upper_score = _lin_map(price_upper - 1e-6, price_lower, price_upper, invert=True)
                    price_score = max(0.0, upper_score * (hard_zero_at - float(p)) / (hard_zero_at - price_upper))

            commute_score: Optional[float]
            if d is None or d <= 0:
                commute_score = None
                missing_commute_count += 1
            else:
                d_val = float(d)
                if d_val <= commute_ideal:
                    commute_score = 100.0
                elif d_val >= commute_max_bound:
                    commute_score = 0.0
                else:
                    commute_score = 100.0 * (commute_max_bound - d_val) / (commute_max_bound - commute_ideal)

            sqft_score: Optional[float]
            if s is None or s <= 0:
                sqft_score = None
            else:
                s_val = float(s)
                # Ignore extreme OCR artifacts
                if used_ocr and (s_val < 250 or s_val > 2000):
                    sqft_score = None
                else:
                    base_sqft = _lin_map(s_val, sqft_lo, sqft_hi, invert=False)
                    if used_ocr and s_conf is not None and s_conf < conf_full:
                        base_sqft *= partial_weight
                    sqft_score = base_sqft

            # Weight redistribution
            parts = []
            w_parts = []
            if commute_score is not None:
                parts.append(_clamp(commute_score))
                w_parts.append(w_commute)
            if price_score is not None:
                parts.append(_clamp(price_score))
                w_parts.append(w_price)
            if sqft_score is not None:
                parts.append(_clamp(sqft_score))
                w_parts.append(w_sqft)

            if not parts:
                base = default_factor_score
            else:
                w_sum = sum(w_parts)
                if w_sum <= 0:
                    base = default_factor_score
                else:
                    base = sum(v * (w / w_sum) for v, w in zip(parts, w_parts))

            # Nudges
            score = base
            # Broker/application fees
            if r.get("has_broker_fee") or (r.get("broker_fee_pct") and float(r.get("broker_fee_pct") or 0) > 0):
                score -= float(_get(nudge_cfg, ["broker_fee_flat_penalty"], 8))
            if r.get("has_application_fee"):
                score -= float(_get(nudge_cfg, ["application_fee"], 2))
            # Availability now
            if r.get("availability_now"):
                score += float(_get(nudge_cfg, ["availability_now"], 3))
            # Days on market
            dom_thresholds = _get(nudge_cfg, ["days_on_market_penalty"], [
                {"min": 45, "penalty": 2},
                {"min": 75, "penalty": 4},
                {"min": 120, "penalty": 6},
            ])
            score -= _days_on_market_penalty(r.get("days_on_market"), dom_thresholds)
            # Rent stabilized
            if r.get("is_rent_stabilized"):
                score += float(_get(nudge_cfg, ["rent_stabilized"], 3))
            # Amenity cap
            score += _amenity_bonus(r, amen_cap, nudge_cfg)

            # Final clamp and round
            score = _clamp(score)
            score_int = int(round(score))
            updates.append((score_int, int(r["id"])))

        # Batch updates
        if updates:
            chunk = 2000
            for i in range(0, len(updates), chunk):
                batch = updates[i : i + chunk]
                await conn.executemany(
                    "UPDATE listings SET score = $1 WHERE id = $2",
                    batch,
                )

        log.info({
            "event": "scoring_complete",
            "total_listings": len(rows),
            "updated": len(updates),
            "used_ocr_sqft_count": used_ocr_count,
            "missing_commute_count": missing_commute_count,
            "total_sec": round(time.time() - start_time, 2),
            "total_min": round((time.time() - start_time) / 60, 2)
        })
