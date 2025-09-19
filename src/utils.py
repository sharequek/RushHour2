"""
Utility functions for StreetEasy scraper.
"""

import json
import re
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import date

from .constants import LOG_VALUE_TRUNCATE_LENGTH

def price_to_int(text: Optional[str]) -> Optional[int]:
    """Extract price as integer from text like '$2,500' or '$2500'."""
    if not text:
        return None
    m = re.search(r"(\$?\s*[\d,]+)", text)
    if not m:
        return None
    return int(re.sub(r"[^\d]", "", m.group(1)))


def num_from_text(text: Optional[str]) -> Optional[float]:
    """Extract first number from text."""
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else None


def extract_unit_from_listing_name(listing_name: str, street_prefix: str) -> Optional[str]:
    """Extract unit number from listing name given street prefix."""
    ln = listing_name.strip()
    sp = street_prefix.strip()
    if ln.lower().startswith(sp.lower()):
        rest = ln[len(sp):].strip()
        m = re.search(r"(?:#\s*\w+|(?:apt|apartment|unit)\s*\w+)", rest, flags=re.I)
        if m:
            return m.group(0).strip()
        toks = rest.split()
        if len(toks) <= 3 and rest:
            return rest
    return None


def combine_address(listing_name: Optional[str], building_address: Optional[str]) -> Optional[str]:
    """Combine listing name and building address intelligently."""
    if not building_address:
        return listing_name or None
    if not listing_name:
        return building_address
    street_prefix = building_address.split(",", 1)[0].strip()
    unit = extract_unit_from_listing_name(listing_name, street_prefix)
    if unit:
        parts = building_address.split(",", 1)
        return f"{street_prefix} {unit}, {parts[1].strip()}" if len(parts) > 1 else f"{street_prefix} {unit}"
    return listing_name if len(listing_name) > len(building_address) else building_address


def _durations(ms: int) -> dict:
    """Convert milliseconds to dict with ms, sec, min."""
    try:
        return {"ms": ms, "sec": round(ms / 1000.0, 2), "min": round(ms / 60000.0, 2)}
    except Exception:
        return {"ms": ms}


def _log_field(name: str, value, t_start: float, verbose_fields: bool) -> None:
    """Log field extraction with timing if verbose mode enabled."""
    if not verbose_fields:
        return
    try:
        ms_raw = (time.time() - t_start) * 1000
        ms = int(ms_raw) if ms_raw >= 1 else (1 if ms_raw > 0 else 0)
        payload = {"event": "field_extracted", "ms": ms, "name": name}
        
        if isinstance(value, (int, float, bool)):
            payload["value"] = value
        elif isinstance(value, str):
            payload["value"] = value[:LOG_VALUE_TRUNCATE_LENGTH]
        elif isinstance(value, date):
            payload["value"] = value.isoformat()
            
        log = logging.getLogger("streeteasy")
        log.info(payload)
    except Exception:
        pass


def load_config() -> dict:
    """Load config/config.json if present."""
    cfg_path = Path('config/config.json')
    if cfg_path.exists():
        try:
            return json.loads(cfg_path.read_text(encoding='utf-8'))
        except Exception:
            pass
    return {}


def build_streeteasy_url(config: Dict[str, Any]) -> str:
    """Build the full StreetEasy URL from the new filters configuration."""
    scraper_config = config.get("scraper", {})
    base_url = scraper_config.get("start_url", "https://streeteasy.com/for-rent/nyc")
    filters = scraper_config.get("filters", {})
    
    filter_parts = []
    
    # Price filter: price:min-max
    if "price" in filters and len(filters["price"]) == 2:
        min_price, max_price = filters["price"]
        filter_parts.append(f"price:{min_price}-{max_price}")
    
    # Areas filter: area:comma,separated,list
    if "areas" in filters and filters["areas"]:
        areas_str = ",".join(map(str, filters["areas"]))
        filter_parts.append(f"area:{areas_str}")
    
    # Beds filter: beds<=X
    if "beds_max" in filters:
        filter_parts.append(f"beds<={filters['beds_max']}")
    
    # Baths filter: baths>=X
    if "baths_min" in filters:
        filter_parts.append(f"baths>={filters['baths_min']}")
    
    # Amenities filter: amenities:comma,separated,list
    if "amenities" in filters and filters["amenities"]:
        amenities_str = ",".join(filters["amenities"])
        filter_parts.append(f"amenities:{amenities_str}")
    
    # Build the final URL
    if filter_parts:
        return f"{base_url}/{'|'.join(filter_parts)}"
    else:
        return base_url
