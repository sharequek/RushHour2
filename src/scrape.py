import asyncio
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Dict, Any
from urllib.parse import urlparse, urlunparse
from pathlib import Path
import json
from datetime import datetime, timezone, date

from playwright.async_api import async_playwright, Page, BrowserContext
from asyncpg import Pool

from .db import make_pool, ensure_schema, upsert_listing, upsert_listing_transaction, replace_listing_photos, replace_listing_photos_transaction, replace_listing_features, replace_listing_features_transaction, update_scrape_duration
from .utils import price_to_int, num_from_text, combine_address, _durations, _log_field, load_config, build_streeteasy_url
from .commute import calculate_commutes_for_listings
from .llm_enricher import enrich_listings_with_llm_optimized
from .feature_flags import populate_feature_flags_for_listing, populate_feature_flags_for_listing_transaction, populate_feature_flags_from_scraped_data
from .ocr_extractor import process_listing_sqft_ocr, process_all_listings_sqft_ocr
from .scoring import score_all_listings
from .constants import (
    BROWSER_VIEWPORT, USER_AGENT, DEFAULT_TEXT_TIMEOUT, EXTENDED_TEXT_TIMEOUT,
    PAGE_LOAD_TIMEOUT, NAVIGATION_TIMEOUT, SELECTOR_TIMEOUT, DESCRIPTION_TIMEOUT,
    SHORT_DESCRIPTION_TIMEOUT, CHALLENGE_COOLDOWN_SECONDS, PHOTO_SIZE_TRANSFORMATIONS
)
from dotenv import load_dotenv


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("streeteasy")


PX_SELECTORS = [
    '#px-captcha',
    '#px-captcha-wrapper',
    '.px-captcha-container',
    '.px-captcha-message',
    'iframe[src*="perimeterx"]',
    'div[id*="px-"]',
    'text=Press & Hold',
    'text=Please verify you are a human',
    'text=Verifying you are human',
]


# Global challenge coordination state - pauses entire script indefinitely
GLOBAL_CHALLENGE_STATE = {"solving": False, "solved_at": 0}
GLOBAL_CHALLENGE_LOCK = None  # Will be initialized when needed


# Verbose logging controls (enabled only for single-detail scrapes)
VERBOSE_FIELDS: bool = False            # Individual field_extracted logs
VERBOSE_DETAIL_EVENTS: bool = False     # detail_* and photos/features logs


def get_optimized_browser_args() -> List[str]:
    """Get optimized browser arguments for consistent DNS and performance settings."""
    return [
        "--disable-blink-features=AutomationControlled",
        "--dns-prefetch-disable=false",  # Enable DNS prefetching
        "--enable-features=AsyncDns",    # Use async DNS resolution
        "--disable-dev-shm-usage",       # Reduce memory issues
        "--disable-features=VizDisplayCompositor",  # Performance optimization
        "--disable-secure-dns",          # Disable DNS-over-HTTPS (secure DNS)
        "--dns-over-https-mode=off",     # Explicitly turn off DoH
    ] + (["--no-sandbox"] if os.name == 'posix' else [])








async def wait_if_challenge(page: Page, once_state: dict, suppress_logs: bool = False) -> None:
    async def is_visible(sel: str) -> bool:
        try:
            return await page.locator(sel).first.is_visible()
        except Exception:
            return False

    for sel in PX_SELECTORS:
        if await is_visible(sel):
            if not suppress_logs and not once_state.get("logged", False):
                log.warning({"event": "challenge_detected"})
                once_state["logged"] = True
            # Wait until all challenge selectors are hidden using locator API (supports text selectors)
            for s in PX_SELECTORS:
                try:
                    await page.locator(s).first.wait_for(state="hidden", timeout=0)
                except Exception:
                    pass
            await page.wait_for_load_state("domcontentloaded")
            if not suppress_logs:
                log.info({"event": "challenge_cleared"})
            break


async def check_global_challenge(page: Page, px_once: dict) -> bool:
    """Global challenge detection and coordination - pauses entire script indefinitely."""
    global GLOBAL_CHALLENGE_LOCK
    
    # Initialize lock if not done yet
    if GLOBAL_CHALLENGE_LOCK is None:
        GLOBAL_CHALLENGE_LOCK = asyncio.Lock()
    
    # Wait a moment for page to fully load
    await asyncio.sleep(0.5)
    
    # Quick check if challenge is visible
    challenge_found = False
    try:
        # Check title first (most reliable)
        title = await page.title()
        if "Access to this page has been denied" in title:
            log.info({"event": "global_challenge_detected_by_title", "title": title})
            challenge_found = True
        else:
            # Check selectors
            for sel in PX_SELECTORS[:4]:  # Check the most reliable selectors first
                try:
                    is_visible = await page.locator(sel).first.is_visible()
                    if is_visible:
                        log.info({"event": "global_challenge_detected_by_selector", "selector": sel})
                        challenge_found = True
                        break
                except Exception:
                    pass
    except Exception:
        pass
    
    if not challenge_found:
        return False
        
    # Challenge detected - coordinate globally (entire script pauses)
    async with GLOBAL_CHALLENGE_LOCK:
        current_time = time.time()
        
        # If someone else solved it recently, just reload and continue
        if GLOBAL_CHALLENGE_STATE["solved_at"] > 0 and (current_time - GLOBAL_CHALLENGE_STATE["solved_at"]) < CHALLENGE_COOLDOWN_SECONDS:
            log.info({"event": "global_challenge_recently_solved_reloading"})
            await page.reload(wait_until="domcontentloaded")
            return False
        
        # If no one is solving, become the solver
        if not GLOBAL_CHALLENGE_STATE["solving"]:
            GLOBAL_CHALLENGE_STATE["solving"] = True
            log.info({"event": "SCRIPT_PAUSED_challenge_detected_solve_to_continue"})
            
            try:
                await page.bring_to_front()
            except Exception:
                pass
            
            # Wait indefinitely for challenge to be solved (user interaction)
            await wait_if_challenge(page, px_once, suppress_logs=True)
            
            # Mark as solved
            GLOBAL_CHALLENGE_STATE["solving"] = False
            GLOBAL_CHALLENGE_STATE["solved_at"] = time.time()
            log.info({"event": "SCRIPT_RESUMED_challenge_solved_continuing"})
            return False
        else:
            # Someone else is solving, wait indefinitely
            log.info({"event": "SCRIPT_PAUSED_waiting_for_challenge_solve"})
    
    # Wait indefinitely for the other worker to solve it
    while GLOBAL_CHALLENGE_STATE["solving"]:
        await asyncio.sleep(0.5)
    
    # Reload to get the solved session
    log.info({"event": "SCRIPT_RESUMED_challenge_solved_reloading"})
    await page.reload(wait_until="domcontentloaded")
    return False


def is_allowed_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        if not parsed.netloc.endswith("streeteasy.com"):
            return False
        path = parsed.path or ""
        # Accept only canonical rental detail URLs like /rental/12345
        if path.startswith("/rental/"):
            segments = [seg for seg in path.split("/") if seg]
            # segments[0] == 'rental'; require an ID segment with at least one digit
            if len(segments) >= 2 and bool(re.search(r"\d", segments[1])):
                return True
            return False
        if path.startswith("/building/"):
            segments = [seg for seg in path.split("/") if seg]
            if len(segments) < 3:
                return False
            unit_segment = segments[2]
            return bool(re.search(r"\d", unit_segment))
        return False
    except Exception:
        return False


def normalize_url(url: str) -> str:
    try:
        parsed = urlparse(url)
        path = parsed.path or ""
        if path.startswith("/building/"):
            segments = [seg for seg in path.split("/") if seg]
            if len(segments) >= 3:
                # keep only /building/{slug}/{unit}
                new_path = f"/{segments[0]}/{segments[1]}/{segments[2]}"
                parsed = parsed._replace(path=new_path)
                return urlunparse(parsed)
        return url
    except Exception:
        return url





async def get_text(page: Page, selector: str, timeout_ms: int = DEFAULT_TEXT_TIMEOUT) -> Optional[str]:
    try:
        loc = page.locator(selector).first
        return await loc.text_content(timeout=timeout_ms)
    except Exception:
        return None


async def wait_for_selector_safe(page: Page, selector: str, timeout_ms: int) -> bool:
    try:
        await page.wait_for_selector(selector, timeout=timeout_ms)
        return True
    except Exception:
        return False


async def collect_links_from_results(page: Page, max_pages: Optional[int], px_once: dict) -> List[str]:
    seen: Set[str] = set()
    t0 = time.time()

    async def harvest(page_idx: int) -> None:
        await check_global_challenge(page, px_once)
        # Wait until anchors are present (DOM-agnostic)
        try:
            await page.wait_for_selector('a[href]', timeout=PAGE_LOAD_TIMEOUT)
        except Exception:
            return
        # Light lazy-load scrolling
        try:
            last_h = await page.evaluate("document.body.scrollHeight")
            for _ in range(10):
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await asyncio.sleep(0.4)
                h = await page.evaluate("document.body.scrollHeight")
                if h == last_h:
                    break
                last_h = h
        except Exception:
            pass

        # Collect anchors, absolutize in JS, then filter/normalize in Python
        th = time.time()
        hrefs: List[str] = await page.evaluate(
            """
            () => Array.from(new Set(
              Array.from(document.querySelectorAll('a[href]'))
                .map(a => { try { return new URL(a.getAttribute('href'), window.location.href).href } catch { return null } })
                .filter(Boolean)
            ))
            """
        )
        before = len(seen)
        for u in hrefs:
            if is_allowed_url(u):
                seen.add(normalize_url(u))
        added = len(seen) - before
        page_ms = int((time.time()-th)*1000)
        log.info({"event": "links_page", "ms": page_ms, "page": page_idx, "added": added, "total": len(seen)})

    page_index = 1
    await harvest(page_index)
    while True:
        if max_pages is not None and page_index >= max_pages:
            break

        # Try to go to next page if available (support multiple selectors)
        next_sel = 'a[aria-labelledby="next-arrow-label"], a[rel="next"], a[aria-label="Next"], a[aria-label*="Next"], button[aria-label="Next"], button[aria-label*="Next"]'
        next_candidates = page.locator(next_sel)
        if await next_candidates.count() == 0:
            break
        nxt = next_candidates.first
        old_url = page.url
        try:
            await nxt.click()
        except Exception:
            # Try to scroll into view and click again
            try:
                await nxt.scroll_into_view_if_needed()
                await asyncio.sleep(0.2)
                await nxt.click()
            except Exception:
                break
        # Wait for URL to change or new results to render
        try:
            await page.wait_for_function("old => location.href !== old", arg=old_url, timeout=NAVIGATION_TIMEOUT)
        except Exception:
            pass
        await page.wait_for_load_state("domcontentloaded")
        await check_global_challenge(page, px_once)
        await asyncio.sleep(0.8)
        page_index += 1
        await harvest(page_index)

    took_ms = int((time.time() - t0) * 1000)
    dur = _durations(took_ms)
    log.info({"event": "links_collected", "sec": dur["sec"], "min": dur["min"], "links": len(seen)})
    return sorted(seen)


@dataclass
class RawExtracted:
    """Raw data extracted from page DOM - minimal processing, browser-open phase."""
    url: str
    listing_name_text: Optional[str]
    building_address_text: Optional[str]
    price_text: Optional[str]
    details_texts: List[str]
    availability_text: Optional[str]
    leasing_starts_text: Optional[str]
    days_on_market_text: Optional[str]
    last_price_change_text: Optional[str]
    description_raw: Optional[str]
    google_maps_href: Optional[str]
    photo_items_raw: List[dict]  # Raw JS objects
    features_items_raw: List[dict]  # Raw JS objects


@dataclass
class Extracted:
    url: str
    listing_name: Optional[str]
    building_address: Optional[str]
    combined_address: str
    price: Optional[int]
    effective_price: Optional[int]
    free_months: Optional[int]
    lease_term_months: Optional[int]
    sqft: Optional[int]
    price_per_sqft: Optional[float]
    rooms: Optional[float]
    beds: Optional[float]
    baths: Optional[float]
    availability_date: Optional[date]
    availability_now: Optional[bool]
    days_on_market: Optional[int]
    last_change_amount: Optional[int]
    last_change_pct: Optional[float]
    last_change_date: Optional[date]
    photos: List[Tuple[str, str]]
    description: Optional[str]
    google_maps: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    raw: dict
    features: List[Tuple[str, Optional[str]]]


async def extract_raw_data(page: Page) -> RawExtracted:
    """Extract only raw strings/data from DOM - minimal processing, browser-open phase."""
    await page.wait_for_load_state("domcontentloaded")

    # Get canonical URL (minimal processing)
    try:
        url = await page.locator('link[rel="canonical"]').first.get_attribute('href')
        if not url:
            url = page.url
    except Exception:
        url = page.url

    # Raw text extraction only - no parsing/processing
    listing_name_text = await get_text(page, 'h1[data-testid="address"]') \
        or await get_text(page, 'h1[class*="PrimaryLarge_"]') \
        or await get_text(page, 'h1')
    if not listing_name_text:
        listing_name_text = await get_text(page, 'h1[data-testid="address"]', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, 'h1[class*="PrimaryLarge_"]', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, 'h1', EXTENDED_TEXT_TIMEOUT)

    building_address_text = await get_text(page, 'p.AboutBuildingSection_address__TdYEX') \
        or await get_text(page, 'p[class*="AboutBuildingSection_address"]') \
        or await get_text(page, 'address, [data-testid="address"]')
    if not building_address_text:
        building_address_text = await get_text(page, 'p.AboutBuildingSection_address__TdYEX', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, 'p[class*="AboutBuildingSection_address"]', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, 'address, [data-testid="address"]', EXTENDED_TEXT_TIMEOUT)

    price_text = await get_text(page, 'h4[class*="PriceInfo_price"]') \
        or await get_text(page, '[data-testid="price"]') \
        or await get_text(page, 'span:has-text("$")')
    if not price_text:
        price_text = await get_text(page, 'h4[class*="PriceInfo_price"]', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, '[data-testid="price"]', EXTENDED_TEXT_TIMEOUT) \
            or await get_text(page, 'span:has-text("$")', EXTENDED_TEXT_TIMEOUT)

    # Get raw property details texts
    details_texts = await page.eval_on_selector_all(
        '[data-testid="propertyDetails"] .PropertyDetails_item__4mGTQ p, [data-testid="propertyDetails"] p',
        "els => els.map(e => e.textContent?.trim() || '')",
    ) or []

    # Raw rental spec texts
    await wait_for_selector_safe(page, '[data-testid="rentalListingSpecSection"]', SELECTOR_TIMEOUT)
    availability_text = await get_text(page, '[data-testid="rentalListingSpec-available"] .RentalListingSpec_bodyWrapper__K_R5w', DESCRIPTION_TIMEOUT)
    leasing_starts_text = await get_text(page, '[data-testid="rentalListingSpec-leasingStarts"] .RentalListingSpec_bodyWrapper__K_R5w', DESCRIPTION_TIMEOUT)
    days_on_market_text = await get_text(page, '[data-testid="rentalListingSpec-daysOnMarket"] .RentalListingSpec_bodyWrapper__K_R5w', DESCRIPTION_TIMEOUT)
    last_price_change_text = await get_text(page, '[data-testid="priceChangeValue"], [data-testid="rentalListingSpec-latPriceChanged"] .RentalListingSpec_bodyWrapper__K_R5w', DESCRIPTION_TIMEOUT)

    # Raw description extraction
    description_raw = None
    try:
        about_sel = '[data-testid="about-section"]'
        present = await wait_for_selector_safe(page, about_sel, SELECTOR_TIMEOUT)
        if present:
            # Click expand button if present
            try:
                btn = page.locator(f"{about_sel} button:has-text('show full description')").first
                if await btn.count() > 0:
                    await btn.click()
                    await asyncio.sleep(0.2)
            except Exception:
                pass
            # Get raw description text
            long_loc = page.locator(f"{about_sel} .ListingDescription_longDescription__vvBTw").first
            if await long_loc.count() > 0:
                description_raw = await long_loc.inner_text(timeout=DESCRIPTION_TIMEOUT)
            else:
                try:
                    description_raw = await page.locator(f"{about_sel}").inner_text(timeout=SHORT_DESCRIPTION_TIMEOUT)
                except Exception:
                    pass
    except Exception:
        pass

    # Raw Google Maps href
    google_maps_href = None
    try:
        google_maps_href = await page.eval_on_selector(
            'a.GoogleMapsLink_link__Ao9ZV, a[href*="google.com/maps"]',
            "el => el ? el.getAttribute('href') : null"
        )
    except Exception:
        pass

    # Raw photo data (no processing) - only capture main listing photos, not past listing photos
    photo_items_raw = []
    try:
        # Target only the main photo carousel, not the thumbnail carousel or past listing photos
        # Based on inspection: there are 4 swiper wrappers:
        # 1. Main carousel (19 slides, large images)
        # 2. Main carousel thumbnails (19 slides, medium images) 
        # 3. Past listing carousel (29 slides, large images)
        # 4. Past listing thumbnails (29 slides, medium images)
        # We only want the first swiper wrapper (main carousel)
        photo_items_raw = await page.eval_on_selector_all(
            '[data-testid="media-section"] .swiper-wrapper:first-of-type .swiper-slide',
            """
            slides => slides.map(slide => {
              const img = slide.querySelector('img');
              const src = img?.getAttribute('src') || '';
              const alt = (img?.getAttribute('alt') || '').toLowerCase();
              const isVideo = !!slide.querySelector('.MediaCarousel_isVideo');
              return { src, alt, isVideo };
            })
            """
        ) or []
    except Exception:
        pass

    # Raw features data (no processing)
    features_items_raw = []
    try:
        features_items_raw = await page.eval_on_selector_all(
            '.Lists_listsWrapper__Mu9vG li.ListItem_item_pkZSl',
            """
            els => els.map(li => {
              const ps = li.querySelectorAll('p');
              const name = ps[0]?.textContent?.trim() || '';
              const sub = ps[1]?.textContent?.trim() || null;
              return { name, sub };
            })
            """
        ) or []
    except Exception:
        pass

    return RawExtracted(
        url=url,
        listing_name_text=listing_name_text,
        building_address_text=building_address_text,
        price_text=price_text,
        details_texts=details_texts,
        availability_text=availability_text,
        leasing_starts_text=leasing_starts_text,
        days_on_market_text=days_on_market_text,
        last_price_change_text=last_price_change_text,
        description_raw=description_raw,
        google_maps_href=google_maps_href,
        photo_items_raw=photo_items_raw,
        features_items_raw=features_items_raw,
    )


def process_raw_data(raw: RawExtracted) -> Extracted:
    """Process raw data into structured format - no page needed, browser-closed phase."""
    
    # Address processing
    combined = combine_address(raw.listing_name_text or "", raw.building_address_text or "")
    if not combined:
        combined = raw.listing_name_text or raw.building_address_text or raw.url

    # Price processing
    price = price_to_int(raw.price_text)

    # Property details processing
    effective_price = None
    free_months = None
    lease_term_months = None
    sqft = None
    price_per_sqft = None
    rooms = None
    beds = None
    baths = None

    for txt in raw.details_texts:
        low = txt.lower()
        if "net effective" in low:
            effective_price = price_to_int(txt)
        elif "months free" in low:
            fm = num_from_text(txt)
            free_months = int(fm) if fm is not None else None
        elif "month lease" in low:
            lt = num_from_text(txt)
            lease_term_months = int(lt) if lt is not None else None
        elif "ft" in low and ("per" not in low):
            v = num_from_text(txt)
            sqft = int(v) if v is not None else sqft
        elif "per" in low and "ft" in low:
            price_per_sqft = num_from_text(txt) or price_per_sqft
        elif "room" in low:
            rooms = num_from_text(txt) or rooms
        elif "bed" in low:
            if "studio" in low:
                beds = 0.0
            else:
                beds = num_from_text(txt) or beds
        elif "bath" in low:
            baths = num_from_text(txt) or baths

    # Default effective price
    if effective_price is None and (free_months is None or free_months == 0) and price is not None:
        effective_price = price

    # Rental spec processing
    availability_date: Optional[date] = None
    availability_now: Optional[bool] = None
    days_on_market: Optional[int] = None
    last_change_amount: Optional[int] = None
    last_change_pct: Optional[float] = None
    last_change_date: Optional[date] = None

    # Process availability
    if raw.availability_text:
        if 'available now' in raw.availability_text.lower():
            availability_now = True
            availability_date = None
        else:
            availability_now = False
            try:
                dt = datetime.strptime(raw.availability_text.strip(), '%m/%d/%Y')
                availability_date = dt.date()
            except Exception:
                availability_date = None

    # Process days on market / leasing starts
    today = datetime.now(timezone.utc).date()
    if raw.days_on_market_text:
        dm = num_from_text(raw.days_on_market_text)
        days_on_market = int(dm) if dm is not None else None
    elif raw.leasing_starts_text:
        try:
            dt = datetime.strptime(raw.leasing_starts_text.strip(), '%m/%d/%Y').date()
            if dt > today:
                days_on_market = 0
            else:
                days_on_market = (today - dt).days
            # Use leasing starts as availability if not set
            if availability_now is not True and availability_date is None:
                availability_date = dt
                availability_now = False
        except Exception:
            pass

    # Process price changes
    if raw.last_price_change_text and 'no changes' not in raw.last_price_change_text.lower():
        last_change_amount = price_to_int(raw.last_price_change_text)
        pct_match = re.search(r"([+-]?[\d.]+)%", raw.last_price_change_text)
        if pct_match:
            try:
                last_change_pct = round(float(pct_match.group(1)), 2)
            except Exception:
                pass
        date_match = re.search(r"on\s+(\d{1,2}/\d{1,2}/\d{2,4})", raw.last_price_change_text)
        if date_match:
            ds = date_match.group(1)
            try:
                fmt = '%m/%d/%y' if len(ds.split('/')[-1]) == 2 else '%m/%d/%Y'
                last_change_date = datetime.strptime(ds, fmt).date()
            except Exception:
                pass

    # Process Google Maps
    google_maps = raw.google_maps_href
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    if google_maps:
        m = re.search(r"([+-]?\d{1,3}(?:\.\d+)?),([+-]?\d{1,3}(?:\.\d+)?)", google_maps)
        if m:
            try:
                latitude = float(m.group(1))
                longitude = float(m.group(2))
            except Exception:
                pass

    # Process photos
    photos_tuples: List[Tuple[str, str]] = []
    def upgrade_url(u: str) -> str:
        return u.replace(PHOTO_SIZE_TRANSFORMATIONS['se_medium_500_250'], 'se_large_800_400') if 'se_medium_500_250' in u else u
    
    def get_base_url(u: str) -> str:
        """Get base URL without size parameters to detect duplicates"""
        # Remove size parameters to identify the same image in different sizes
        base = u.split('-se_')[0] if '-se_' in u else u
        return base
    
    seen: set[str] = set()
    seen_base: set[str] = set()
    
    for item in raw.photo_items_raw:
        if not isinstance(item, dict):
            continue
        src = item.get('src', '')
        if not src or not src.startswith('https://'):
            continue
        
        # Determine type
        alt = item.get('alt', '').lower()
        is_video = item.get('isVideo', False)
        if 'maps.googleapis.com' in src:
            type = 'map'
        elif is_video or 'img.youtube.com' in src:
            type = 'video_thumb'
        elif 'floor plan' in alt:
            type = 'floor_plan'
        else:
            type = 'photo'
        
        upgraded_src = upgrade_url(src)
        base_url = get_base_url(upgraded_src)
        
        # Only add if we haven't seen this exact URL or base URL before
        if upgraded_src not in seen and base_url not in seen_base:
            seen.add(upgraded_src)
            seen_base.add(base_url)
            photos_tuples.append((upgraded_src, type))

    # Process features
    features: List[Tuple[str, Optional[str]]] = []
    for item in raw.features_items_raw:
        if not isinstance(item, dict):
            continue
        name = (item.get('name') or '').strip()
        if not name:
            continue
        sub = item.get('sub')
        sub_clean = sub.strip() if isinstance(sub, str) and sub.strip() else None
        features.append((name, sub_clean))

    # Build raw dict for compatibility
    raw_dict = {
        "url": raw.url,
        "listing_name": raw.listing_name_text,
        "building_address": raw.building_address_text,
        "price_text": raw.price_text,
        "property_details": raw.details_texts,
        "photos": [u for (u, _k) in photos_tuples],
    }

    return Extracted(
        url=raw.url,
        listing_name=raw.listing_name_text,
        building_address=raw.building_address_text,
        combined_address=combined,
        price=price,
        effective_price=effective_price,
        free_months=free_months,
        lease_term_months=lease_term_months,
        sqft=sqft,
        price_per_sqft=price_per_sqft,
        rooms=rooms,
        beds=beds,
        baths=baths,
        availability_date=availability_date,
        availability_now=availability_now,
        days_on_market=days_on_market,
        last_change_amount=last_change_amount,
        last_change_pct=last_change_pct,
        last_change_date=last_change_date,
        photos=photos_tuples,
        description=raw.description_raw,
        google_maps=google_maps,
        latitude=latitude,
        longitude=longitude,
        raw=raw_dict,
        features=features,
    )


async def persist_extracted_data(pool, data: Extracted, config: Dict[str, Any] = None) -> Tuple[int, bool, bool]:
    """Save extracted data to database - no page needed, browser-closed phase."""
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Upsert the listing within the transaction
            listing_id, was_updated, was_new = await upsert_listing_transaction(conn, {
                "url": data.url,
                "combined_address": data.combined_address,
                "listing_name": data.listing_name,
                "building_address": data.building_address,
                "price": data.price,
                "effective_price": data.effective_price,
                "free_months": data.free_months,
                "lease_term_months": data.lease_term_months,
                "sqft": data.sqft,
                "price_per_sqft": data.price_per_sqft,
                "rooms": data.rooms,
                "beds": data.beds,
                "baths": data.baths,
                "availability_date": data.availability_date,
                "availability_now": data.availability_now,
                "days_on_market": data.days_on_market,
                "last_change_amount": data.last_change_amount,
                "last_change_pct": data.last_change_pct,
                "last_change_date": data.last_change_date,
                "description": data.description,
                "google_maps": data.google_maps,
                "latitude": data.latitude,
                "longitude": data.longitude,
                "raw": data.raw,
            })
            
            # Save photos (always update photos for fresh content)
            if data.photos:
                await replace_listing_photos_transaction(conn, listing_id, data.photos)
            
            # Save features (always update features for fresh content)
            if data.features:
                await replace_listing_features_transaction(conn, listing_id, data.features)
            
            # Populate feature flags from scraped features (always run for fresh features)
            await populate_feature_flags_for_listing_transaction(conn, listing_id, data.features)
            
            # Return the listing_id and update status for tracking
            return listing_id, was_updated, was_new


async def scrape_detail(page: Page, pool, url: str, px_once: dict, config: Dict[str, Any] = None) -> Tuple[bool, bool]:
    """Detail scraping with separate browser-open and browser-closed phases for optimal performance."""
    try:
        # Phase 1: Extract raw data (browser open) - FAST
        t_extract = time.time()
        raw_data = await extract_raw_data(page)
        extract_ms = int((time.time() - t_extract) * 1000)
        if VERBOSE_DETAIL_EVENTS:
            log.info({"event": "raw_extract", "ms": extract_ms, "url": url})
        
        # NOTE: Page can be closed here - no more browser interaction needed!
        
        # Phase 2: Process raw data (browser closed) - CPU work
        t_process = time.time()
        processed_data = process_raw_data(raw_data)
        process_ms = int((time.time() - t_process) * 1000)
        if VERBOSE_DETAIL_EVENTS:
            log.info({"event": "data_process", "ms": process_ms, "url": url})
        
        # Phase 3: Save to database (browser closed) - I/O work
        t_persist = time.time()
        listing_id, was_updated, was_new = await persist_extracted_data(pool, processed_data, config)
        persist_ms = int((time.time() - t_persist) * 1000)
        if VERBOSE_DETAIL_EVENTS:
            log.info({"event": "data_persist", "ms": persist_ms, "url": url})
            total_ms = extract_ms + process_ms + persist_ms
            log.info({"event": "detail_complete", "ms": total_ms, "url": url, 
                     "breakdown": {"extract": extract_ms, "process": process_ms, "persist": persist_ms}})
        
        # Update timing in database (combined scraping and feature flags)
        total_scrape_ms = extract_ms + process_ms + persist_ms
        await update_scrape_duration(pool, listing_id, total_scrape_ms)
        
        return was_updated, was_new
    
    except Exception as e:
        log.error({"event": "detail_error", "url": url, "error": str(e)})
        return False, False




async def collect_and_process(config: Dict[str, Any]) -> None:
    pool = await make_pool()
    await ensure_schema(pool)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=get_optimized_browser_args())
        context = await browser.new_context(
            viewport=BROWSER_VIEWPORT,
            user_agent=USER_AGENT,
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Connection": "keep-alive",
            }
        )
        page = await context.new_page()

        # Extract scraper configuration
        scraper_config = config.get("scraper", {})
        start_url = build_streeteasy_url(config)
        max_pages = scraper_config.get("max_pages")
        concurrency = scraper_config.get("concurrency", 20)
        max_links = scraper_config.get("max_links")
        
        t_start = time.time()
        log.info({
            "event": "scrape_start",
            "ms": 0,
            "start_url": start_url,
            "max_pages": max_pages,
            "concurrency": concurrency,
        })
        # Disable verbose detail logs for multi-page runs
        global VERBOSE_FIELDS, VERBOSE_DETAIL_EVENTS
        VERBOSE_FIELDS = False
        VERBOSE_DETAIL_EVENTS = False

        
        # Pre-nav gate for the first page too (if challenge shows during listing pages)
        await page.goto(start_url, wait_until="domcontentloaded")
        px_once = {"logged": False}
        await check_global_challenge(page, px_once)

        links = await collect_links_from_results(page, max_pages, px_once)
        if max_links is not None and max_links > 0 and len(links) > max_links:
            dur = _durations(int((time.time()-t_start)*1000))
            log.info({"event": "links_capped", **dur, "cap": max_links, "original": len(links)})
            links = links[:max_links]

        # Links are already absolute; keep as-is
        abs_links = links

        # Page pool with queue
        queue: asyncio.Queue[str] = asyncio.Queue()
        for u in abs_links:
            queue.put_nowait(u)

        processed_count = 0
        new_listings_count = 0
        updated_listings_count = 0
        count_lock = asyncio.Lock()
        


        async def worker(worker_id: int) -> None:
            # Stagger worker starts to avoid DNS burst
            if worker_id > 0:
                await asyncio.sleep(worker_id * 0.05)  # 50ms delay per worker ID
                
            page = await context.new_page()
            try:
                while True:
                    try:
                        u = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    
                    # Check for challenge before scraping
                    try:
                        await page.goto(u, wait_until="domcontentloaded")
                        # Global challenge detection - will pause entire script if challenge found
                        await check_global_challenge(page, px_once)
                    except Exception as e:
                        log.error({"event": "page_navigation_error", "url": u, "error": str(e)})
                        # Skip this URL if navigation fails
                        continue
                    
                    try:
                        was_updated, was_new = await scrape_detail(page, pool, u, px_once, config)
                    except Exception as e:
                        log.error({"event": "scrape_detail_error", "url": u, "error": str(e)})
                        # Skip this URL if scraping fails
                        continue
                    
                    async with count_lock:
                        nonlocal processed_count, new_listings_count, updated_listings_count
                        processed_count += 1
                        if was_new:
                            new_listings_count += 1
                        elif was_updated:
                            updated_listings_count += 1
                        # If neither was_new nor was_updated, it was skipped (no changes)
                        
                        if processed_count % 10 == 0:
                            elapsed = int((time.time()-t_start)*1000)
                            dur = _durations(elapsed)
                            log.info({
                                "event": "progress", 
                                **dur, 
                                "progress": f"{processed_count}/{len(abs_links)}",
                                "new_listings": new_listings_count,
                                "updated_listings": updated_listings_count
                            })
            finally:
                await page.close()

        await asyncio.gather(*(worker(i) for i in range(max(1, concurrency))))

        total_ms = int((time.time() - t_start) * 1000)
        dur = _durations(total_ms)
        log.info({
            "event": "scrape_finished", 
            **dur, 
            "processed": len(abs_links),
            "new_listings": new_listings_count,
            "updated_listings": updated_listings_count
        })

        await browser.close()

    # Populate feature flags from scraped data
    try:
        await populate_feature_flags_from_scraped_data(pool)
    except Exception as e:
        log.error({"event": "feature_flags_population_error", "error": str(e)})
    
    # Calculate commutes after scraping is complete
    try:
        if config.get("commute", {}).get("enabled", True):
            await calculate_commutes_for_listings(pool, config)
    except Exception as e:
        log.error({"event": "commute_calculation_error", "error": str(e)})

    # Enrich listings with LLM-extracted data after scraping
    try:
        if config.get("llm", {}).get("enabled", False):
            await enrich_listings_with_llm_optimized(pool, config)
    except Exception as e:
        log.error({"event": "llm_enrichment_error", "error": str(e)})

    # Process OCR for square footage extraction after LLM enrichment
    try:
        await process_all_listings_sqft_ocr(pool, config.get("ocr", {}))
    except Exception as e:
        log.error({"event": "ocr_sqft_error", "error": str(e)})

    # Score all listings (calibration-based) after OCR and before summary
    try:
        if config.get("scoring", {}).get("enabled", True):
            await score_all_listings(pool, config)
    except Exception as e:
        log.error({"event": "scoring_error", "error": str(e)})

    # Generate comprehensive script summary
    await generate_script_summary(pool, t_start, new_listings_count, updated_listings_count)

    await pool.close()


async def generate_script_summary(pool: Pool, start_time: float, new_listings: int, updated_listings: int) -> None:
    """Generate a comprehensive summary of the script execution."""
    try:
        async with pool.acquire() as conn:
            # Get total listings count
            total_listings = await conn.fetchval("SELECT COUNT(*) FROM listings")
            
            # Get LLM enrichment statistics
            llm_enriched = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE llm_enrichment_occurred = true")
            llm_completed = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE llm_enrichment_completed_at IS NOT NULL")
            llm_pending = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE llm_enrichment_completed_at IS NULL")
            
            # Get feature flags statistics
            feature_flags_populated = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE has_elevator IS NOT NULL OR has_doorman IS NOT NULL OR has_gym IS NOT NULL")
            
            # Get commute statistics
            commute_calculated = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE commute_calculated_at IS NOT NULL")
            
            # Get OCR statistics
            sqft_ocr_extracted = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE ocr_sqft_extracted IS NOT NULL")
            sqft_ocr_completed = await conn.fetchval("SELECT COUNT(*) FROM listings WHERE ocr_sqft_completed_at IS NOT NULL")
            
        # Calculate total time
        total_time = time.time() - start_time
        total_sec = round(total_time, 2)
        total_min = round(total_time / 60, 2)
        
        # Generate summary
        summary = {
            "event": "script_summary",
            "total_time_sec": total_sec,
            "total_time_min": total_min,
            "total_listings": total_listings,
            "new_listings": new_listings,
            "updated_listings": updated_listings,
            "llm_enriched": llm_enriched,
            "llm_completed": llm_completed,
            "llm_pending": llm_pending,
            "feature_flags_populated": feature_flags_populated,
            "commute_calculated": commute_calculated,
            "sqft_ocr_extracted": sqft_ocr_extracted,
            "sqft_ocr_completed": sqft_ocr_completed
        }
        
        log.info(summary)
        
        # Also print a human-readable summary
        print(f"\n🎉 Script Summary:")
        print(f"   ⏱️  Total time: {total_min} minutes ({total_sec} seconds)")
        print(f"   📊 Total listings: {total_listings}")
        print(f"   🆕 New listings: {new_listings}")
        print(f"   🔄 Updated listings: {updated_listings}")
        print(f"   🤖 LLM enriched: {llm_enriched}")
        print(f"   ✅ LLM completed: {llm_completed}")
        print(f"   ⏳ LLM pending: {llm_pending}")
        print(f"   🏷️  Feature flags populated: {feature_flags_populated}")
        print(f"   🚇 Commute times calculated: {commute_calculated}")
        print(f"   📐 Sqft OCR extracted: {sqft_ocr_extracted}")
        print(f"   ✅ Sqft OCR completed: {sqft_ocr_completed}")
        
    except Exception as e:
        log.error({"event": "script_summary_error", "error": str(e)})




def parse_args(argv: List[str]) -> tuple[Dict[str, Any], Optional[str]]:
    args = argv[1:]
    def val(flag: str, default: Optional[str] = None) -> Optional[str]:
        if flag in args:
            i = args.index(flag)
            return args[i + 1] if i + 1 < len(args) else default
        return default

    cfg = load_config()

    # Override config with command line args if provided
    if val("--start-url"):
        cfg["scraper"]["start_url"] = val("--start-url")
    if val("--max-pages"):
        max_pages_s = val("--max-pages")
        if max_pages_s and max_pages_s.isdigit() and int(max_pages_s) > 0:
            cfg["scraper"]["max_pages"] = int(max_pages_s)
    if val("--max-links"):
        max_links_s = val("--max-links")
        if max_links_s and max_links_s.isdigit() and int(max_links_s) > 0:
            cfg["scraper"]["max_links"] = int(max_links_s)
    if val("--concurrency"):
        concurrency_s = val("--concurrency")
        if concurrency_s and concurrency_s.isdigit():
            cfg["scraper"]["concurrency"] = max(1, int(concurrency_s))
    
    detail_url = val("--detail-url") or os.getenv("DETAIL_URL")
    return cfg, detail_url


async def run_single_detail(detail_url: str, concurrency: int = 1, config: Dict[str, Any] = None) -> None:
    # Validate URL before launching the browser
    t0 = time.time()
    if not is_allowed_url(detail_url):
        log.info({"event": "detail_skipped", "ms": int((time.time()-t0)*1000), "url": detail_url, "reason": "url_not_scrapable"})
        return

    if config is None:
        config = {}

    pool = await make_pool()
    await ensure_schema(pool)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False, args=get_optimized_browser_args())
        context = await browser.new_context(viewport=BROWSER_VIEWPORT)
        px_once = {"logged": False}
        global VERBOSE_FIELDS
        VERBOSE_FIELDS = True
        # Create page and scrape
        page = await context.new_page()
        try:
            await page.goto(detail_url, wait_until="domcontentloaded")
            await check_global_challenge(page, px_once)
            
            await scrape_detail(page, pool, detail_url, px_once, config)
        finally:
            await page.close()
        await browser.close()
    await pool.close()


if __name__ == "__main__":
    # Load .env once for DATABASE_URL, LOG_LEVEL, etc.
    load_dotenv()
    config, detail_url = parse_args(sys.argv)
    if detail_url:
        asyncio.run(run_single_detail(detail_url, config["scraper"]["concurrency"]))
    else:
        asyncio.run(collect_and_process(config))

