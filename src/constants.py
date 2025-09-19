"""
Constants and configuration values for the RushHour2 scraper.
"""

# Database Configuration
DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5432/rushhour2"

# Browser Configuration
BROWSER_VIEWPORT = {"width": 1400, "height": 900}
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Timeout Values (in milliseconds)
DEFAULT_TEXT_TIMEOUT = 800
EXTENDED_TEXT_TIMEOUT = 2000
PAGE_LOAD_TIMEOUT = 20000
NAVIGATION_TIMEOUT = 10000
SELECTOR_TIMEOUT = 4000
DESCRIPTION_TIMEOUT = 1500
SHORT_DESCRIPTION_TIMEOUT = 1000

# Challenge Detection
CHALLENGE_COOLDOWN_SECONDS = 120

# Earth's radius for distance calculations
EARTH_RADIUS_KM = 6371
MILES_PER_KM = 0.621371

# LLM Configuration Defaults
DEFAULT_CONTEXT_LENGTH = 2048
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_BATCH = 512
DEFAULT_TIMEOUT = 60
DEFAULT_RESPONSE_PREVIEW_LENGTH = 200

# Photo URL transformations
PHOTO_SIZE_TRANSFORMATIONS = {
    'se_medium_500_250': 'se_large_800_400'
}

# Logging Configuration
LOG_VALUE_TRUNCATE_LENGTH = 120
