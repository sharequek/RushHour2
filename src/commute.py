import asyncio
import aiohttp
import logging
import time
import math
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

from .constants import EARTH_RADIUS_KM, MILES_PER_KM
from .db import update_commute_duration

log = logging.getLogger("commute")

@dataclass
class CommuteResult:
    distance_km: Optional[float]
    duration_min: Optional[int]
    method: str
    error: Optional[str] = None

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points in kilometers."""
    R = 6371  # Earth's radius in kilometers
    
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c

def calculate_nyc_transit_heuristic(from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> CommuteResult:
    """Calculate NYC transit commute using heuristic (no API calls)."""
    try:
        # Haversine formula for distance
        lat1, lon1 = math.radians(from_lat), math.radians(from_lng)
        lat2, lon2 = math.radians(to_lat), math.radians(to_lng)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        distance_km = EARTH_RADIUS_KM * c
        distance_miles = distance_km * MILES_PER_KM
        
        # NYC transit heuristic: ~15 mph average speed including stops
        # Add 5 minutes for initial wait + transfers
        base_time_minutes = (distance_miles / 15) * 60 + 5
        
        # Round to nearest minute
        duration_minutes = round(base_time_minutes)
        
        return CommuteResult(
            distance_km=round(distance_km, 2),
            duration_min=duration_minutes,
            method="heuristic",
            error=None
        )
    except Exception as e:
        return CommuteResult(None, None, "heuristic", str(e))

class CommuteCalculator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("commute", {})
        self.work_address = self.config.get("address", "33 Whitehall St, New York, NY 10004")
        self.work_coordinates = self.config.get("coordinates")  # [lat, lng] from config
        self.departure_time = self.config.get("departure", "08:30")
        self.heuristic_enabled = self.config.get("heuristic", True)
        self.google_enabled = self.config.get("google", {}).get("enabled", False)
        self.google_api_key = self.config.get("google", {}).get("api_key")
        
        # Cache work coordinates (geocode once if not in config)
        self._work_coords: Optional[Tuple[float, float]] = None
        
    async def calculate_commute(self, from_lat: float, from_lng: float) -> Tuple[Optional[CommuteResult], Optional[CommuteResult]]:
        """Calculate commute using heuristic and optionally Google."""
        
        # Get work coordinates (prefer config, fallback to geocoding)
        if not self._work_coords:
            if self.work_coordinates and len(self.work_coordinates) == 2:
                self._work_coords = (self.work_coordinates[0], self.work_coordinates[1])
            else:
                self._work_coords = await self._geocode_work_address()
                if not self._work_coords:
                    return None, None
                
        work_lat, work_lng = self._work_coords
        
        # Heuristic calculation (always enabled, instant)
        heuristic_result = None
        if self.heuristic_enabled:
            heuristic_result = calculate_nyc_transit_heuristic(from_lat, from_lng, work_lat, work_lng)
        
        # Google calculation (if enabled)
        google_result = None
        if self.google_enabled and self.google_api_key:
            google_result = await self._calculate_google_transit(from_lat, from_lng, work_lat, work_lng)
            
        return heuristic_result, google_result
    
    async def _geocode_work_address(self) -> Optional[Tuple[float, float]]:
        """Geocode work address using free Nominatim with timeout and retries."""
        
        try:
            timeout = aiohttp.ClientTimeout(total=10)  # 10 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                url = "https://nominatim.openstreetmap.org/search"
                params = {
                    "q": self.work_address,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1
                }
                headers = {"User-Agent": "StreetEasy-Scraper/1.0"}
                
                async with session.get(url, params=params, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data:
                            lat, lng = float(data[0]["lat"]), float(data[0]["lon"])
                            log.info(f"Work address geocoded: {lat}, {lng}")
                            return lat, lng
                    else:
                        log.warning(f"Geocoding failed with status {resp.status}")
                        
        except asyncio.TimeoutError:
            log.warning("Geocoding request timed out")
        except Exception as e:
            log.error(f"Work address geocoding failed: {e}")
        
        return None
    
    async def _calculate_google_transit(self, from_lat: float, from_lng: float, to_lat: float, to_lng: float) -> CommuteResult:
        """Calculate commute using Google Transit with departure time."""
        try:
            # Calculate departure timestamp for today at specified time
            now = datetime.now()
            departure_hour, departure_min = map(int, self.departure_time.split(':'))
            departure_dt = now.replace(hour=departure_hour, minute=departure_min, second=0, microsecond=0)
            
            # If time has passed today, use tomorrow
            if departure_dt < now:
                departure_dt += timedelta(days=1)
            
            departure_timestamp = int(departure_dt.timestamp())
            
            async with aiohttp.ClientSession() as session:
                url = "https://maps.googleapis.com/maps/api/directions/json"
                
                params = {
                    "origin": f"{from_lat},{from_lng}",
                    "destination": f"{to_lat},{to_lng}",
                    "mode": "transit",
                    "departure_time": departure_timestamp,
                    "units": "metric",
                    "key": self.google_api_key
                }
                
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        
                        if result.get("status") == "OK" and result.get("routes"):
                            route = result["routes"][0]
                            leg = route["legs"][0]
                            
                            distance_m = leg.get("distance", {}).get("value", 0)
                            duration_s = leg.get("duration", {}).get("value", 0)
                            
                            return CommuteResult(
                                distance_km=round(distance_m / 1000, 2),
                                duration_min=round(duration_s / 60),
                                method="google_transit"
                            )
                        else:
                            return CommuteResult(None, None, "google_transit", result.get("status", "No route found"))
                    else:
                        error_text = await resp.text()
                        return CommuteResult(None, None, "google_transit", f"HTTP {resp.status}: {error_text}")
                        
        except Exception as e:
            return CommuteResult(None, None, "google_transit", str(e))

async def calculate_commutes_for_listings(pool, config: Dict[str, Any]) -> None:
    """Calculate commutes efficiently - heuristic for all, Google only for new listings."""
    
    if not config.get("commute", {}).get("enabled", True):
        log.info("Commute calculation disabled in config")
        return
    
    calculator = CommuteCalculator(config)
    
    # Safety limits from config
    commute_config = config.get("commute", {})
    max_google_calls_per_run = commute_config.get("google", {}).get("max_calls", 200)
    google_batch_size = commute_config.get("google", {}).get("batch_size", 50)
    
    # Get listings that need heuristic calculation (always recalculate - it's free)
    async with pool.acquire() as conn:
        heuristic_listings = await conn.fetch("""
            SELECT id, latitude, longitude, combined_address 
            FROM listings 
            WHERE latitude IS NOT NULL 
              AND longitude IS NOT NULL 
              AND commute_calculated_at IS NULL
            ORDER BY id
        """)
    
    # Get listings that need Google calculation (never calculated before)
    google_listings = []
    if calculator.google_enabled and calculator.google_api_key:
        async with pool.acquire() as conn:
            google_listings = await conn.fetch(f"""
                SELECT id, latitude, longitude, combined_address 
                FROM listings 
                WHERE latitude IS NOT NULL 
                  AND longitude IS NOT NULL 
                  AND commute_distance_google_km IS NULL
                  AND commute_duration_google_min IS NULL
                ORDER BY id
                LIMIT {max_google_calls_per_run}
            """)
    
    if not heuristic_listings and not google_listings:
        log.info("No listings need commute calculation")
        return
    
    if heuristic_listings:
        # Check if using config coordinates
        using_config_coords = calculator.work_coordinates and len(calculator.work_coordinates) == 2
        coord_source = "config" if using_config_coords else "geocoded"
        coords_info = f"Using {coord_source} coordinates for {calculator.work_address}"
        
        log.info({"event": "heuristic_calculation_start", "count": len(heuristic_listings), 
                 "work_coordinates": coords_info})
    
    # Handle large Google batches with cost warnings
    if len(google_listings) >= 100:
        estimated_cost = (len(google_listings) / 1000) * 5
        log.warning({"event": "large_google_batch", "count": len(google_listings), 
                    "estimated_cost": f"${estimated_cost:.2f}", 
                    "estimated_time_min": round(len(google_listings) * 0.2 / 60, 1)})
    
    # Process heuristic calculations (fast, free)
    t_heuristic_start = time.time()
    for i, listing in enumerate(heuristic_listings):
        try:
            t_listing_start = time.time()
            heuristic_result, _ = await calculator.calculate_commute(
                listing["latitude"], listing["longitude"]
            )
            
            # Round the distance to avoid precision issues
            distance_km = round(heuristic_result.distance_km, 2) if heuristic_result and heuristic_result.distance_km else None
            
            async with pool.acquire() as conn:
                await conn.execute("""
                    UPDATE listings SET 
                        commute_distance_heur_km = $1,
                        commute_duration_heur_min = $2,
                        commute_calculated_at = NOW()
                    WHERE id = $3
                """, 
                    distance_km,
                    heuristic_result.duration_min if heuristic_result else None,
                    listing["id"]
                )
            
            # Update timing for this listing
            listing_ms = int((time.time() - t_listing_start) * 1000)
            await update_commute_duration(pool, listing["id"], listing_ms)
            
            if (i + 1) % 50 == 0:
                elapsed_ms = int((time.time() - t_heuristic_start) * 1000)
                log.info({"event": "heuristic_progress", "ms": elapsed_ms, 
                         "progress": f"{i + 1}/{len(heuristic_listings)}"})
            
        except Exception as e:
            log.error({"event": "heuristic_calculation_error", "listing_id": listing["id"], "error": str(e)})
    
    if heuristic_listings:
        heuristic_ms = int((time.time() - t_heuristic_start) * 1000)
        log.info({"event": "heuristic_calculation_complete", "ms": heuristic_ms, "count": len(heuristic_listings)})
    
    # Process Google calculations in batches (paid API)
    t_google_start = time.time()
    if google_listings and calculator.google_enabled:
        log.info({"event": "google_calculation_start", "count": len(google_listings)})
        
        # Process in smaller batches to avoid overwhelming the API
        for batch_start in range(0, len(google_listings), google_batch_size):
            batch_end = min(batch_start + google_batch_size, len(google_listings))
            batch = google_listings[batch_start:batch_end]
            batch_num = batch_start // google_batch_size + 1
            
            log.info({"event": "google_batch_start", "batch": batch_num, 
                     "range": f"{batch_start+1}-{batch_end}", "size": len(batch)})
            
            for i, listing in enumerate(batch):
                try:
                    t_listing_start = time.time()
                    _, google_result = await calculator.calculate_commute(
                        listing["latitude"], listing["longitude"]
                    )
                    
                    # Round the distance to avoid precision issues
                    distance_km = round(google_result.distance_km, 2) if google_result and google_result.distance_km else None
                    
                    async with pool.acquire() as conn:
                        await conn.execute("""
                            UPDATE listings SET 
                                commute_distance_google_km = $1,
                                commute_duration_google_min = $2
                            WHERE id = $3
                        """, 
                            distance_km,
                            google_result.duration_min if google_result else None,
                            listing["id"]
                        )
                    
                    # Update timing for this listing (Google calculation)
                    listing_ms = int((time.time() - t_listing_start) * 1000)
                    await update_commute_duration(pool, listing["id"], listing_ms)
                    
                    # Rate limiting for Google API
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    log.error({"event": "google_calculation_error", "listing_id": listing["id"], "error": str(e)})
                    continue
            
            # Pause between batches
            if batch_end < len(google_listings):
                log.info({"event": "google_batch_pause", "batch": batch_num, "pause_sec": 2})
                await asyncio.sleep(2)
        
        google_ms = int((time.time() - t_google_start) * 1000)
        log.info({"event": "google_calculation_complete", "ms": google_ms, "count": len(google_listings)})
    
    # Final summary
    total_ms = int((time.time() - t_heuristic_start) * 1000)
    log.info({"event": "commute_calculation_summary", "ms": total_ms, 
             "heuristic_count": len(heuristic_listings), "google_count": len(google_listings)})
