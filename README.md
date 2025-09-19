# RushHour2 - NYC Apartment Hunter

A sophisticated apartment hunting tool that scrapes apartment listings from various sites (currently StreetEasy), analyzes commute times, extracts property details using OCR, and provides intelligent scoring to help you find the perfect NYC apartment.

## 🏠 Features

- **Smart Scraping**: Automated apartment listing collection with anti-bot protection handling (currently StreetEasy, expandable to other sites)
- **Commute Analysis**: Real-time commute time calculations to your workplace
- **OCR Processing**: Advanced text extraction from listing images using PaddleOCR and Tesseract
- **LLM Enhancement**: AI-powered data enrichment using Ollama for missing property details
- **Intelligent Scoring**: Multi-factor scoring system considering price, commute, square footage, and amenities
- **Modern Web Interface**: React + TypeScript frontend with responsive design
- **RESTful API**: FastAPI backend with comprehensive endpoints
- **Database Integration**: PostgreSQL with optimized schema and async operations

## 🚀 Quick Start

### Prerequisites

- **Docker Desktop** (for PostgreSQL)
- **Python 3.12+**
- **Node.js 18+** (for web interface)
- **PowerShell** (Windows) or **Bash** (Linux/macOS)

### 1. Start Database

```bash
docker compose up -d
```

### 2. One-Time Setup

```bash
# Windows
./scripts/setup.ps1

# Linux/macOS
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Set up Playwright browsers
- Install Node.js dependencies

### 3. Configure Settings

Edit `config/config.json` to customize:
- Search filters (price range, neighborhoods, amenities)
- Commute destination and preferences
- LLM model settings
- OCR processing parameters
- Scoring weights

### 4. Run the Scraper

```bash
# Windows
./scripts/scrape.ps1 -Concurrency 3 -MaxPages 2 -MaxLinks 50

# Linux/macOS
./scripts/scrape.sh --concurrency 3 --max-pages 2 --max-links 50
```

### 5. Start Web Interface

```bash
# Terminal 1: Start API server
cd api && python main.py

# Terminal 2: Start web interface
cd web && npm run dev
```

Visit `http://localhost:5173` to view the results!

## 📁 Project Structure

```
RushHour2/
├── api/                    # FastAPI backend
│   └── main.py            # API server and endpoints
├── config/                 # Configuration files
│   └── config.json        # Main configuration
├── docs/                   # Documentation
├── scripts/                # Automation scripts
│   ├── setup.ps1          # One-time setup
│   ├── scrape.ps1         # Main scraper script
│   ├── test.ps1           # Run tests
│   └── db.ps1             # Database utilities
├── src/                    # Core Python modules
│   ├── scrape.py          # Main scraping logic
│   ├── db.py              # Database operations
│   ├── commute.py         # Commute calculations
│   ├── llm_enricher.py    # AI data enrichment
│   ├── ocr_extractor.py   # OCR processing
│   ├── scoring.py         # Listing scoring
│   └── utils.py           # Utilities
├── tests/                  # Test suite
├── tools/                  # Development tools
├── web/                    # React frontend
│   ├── src/               # React components
│   ├── package.json       # Node.js dependencies
│   └── vite.config.ts     # Vite configuration
├── docker-compose.yml     # PostgreSQL setup
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Search Filters

Configure your apartment search criteria in `config/config.json`. The configuration is designed to work across multiple listing sites:

```json
{
  "scraper": {
    "site": "streeteasy",  // Currently supports "streeteasy", expandable
    "filters": {
      "price": [2000, 4500],
      "areas": [101, 102, 119],  // NYC neighborhood codes (site-specific)
      "beds_max": 1,
      "baths_min": 1,
      "amenities": ["washer_dryer", "dishwasher", "elevator"]
    }
  }
}
```

### Commute Settings

Set your workplace location and preferences:

```json
{
  "commute": {
    "address": "33 Whitehall St, New York, NY 10004",
    "coordinates": [40.7055, -74.0122],
    "departure": "08:30",
    "heuristic": true
  }
}
```

### Scoring Weights

Customize how listings are scored:

```json
{
  "scoring": {
    "weights": {
      "commute": 0.40,
      "price": 0.35,
      "sqft": 0.25
    }
  }
}
```

## 🛠️ Advanced Usage

### Scrape Single Listing

```bash
./scripts/scrape.ps1 -DetailUrl "https://streeteasy.com/rental/1234567"
```

### Database Management

```bash
# Clear all listings
./scripts/db.ps1 -Clear

# Recreate database
./scripts/recreate_db.ps1
```

### Run Tests

```bash
./scripts/test.ps1
```

### OCR Processing

```bash
# Process square footage from listing images
python -m src.ocr_extractor
```

## 🔍 API Endpoints

The FastAPI backend provides these endpoints:

- `GET /listings` - Get all listings with pagination
- `GET /listings/{id}` - Get specific listing details
- `GET /stats` - Get scraping statistics
- `POST /listings/search` - Search listings with filters

## 🧪 Testing

Run the test suite:

```bash
# All tests
./scripts/test.ps1

# Specific test modules
python -m pytest tests/test_llm_protection.py
python -m pytest tests/test_ocr_regression.py
```

## 🚨 Troubleshooting

### Common Issues

**Browsers not installed:**
```bash
./scripts/setup.ps1
```

**Database connection issues:**
- Ensure Docker Desktop is running
- Check `docker compose ps` for container status
- Verify `DATABASE_URL` in environment variables

**OCR processing errors:**
- Ensure PaddleOCR dependencies are installed
- Check image file permissions
- Verify OCR configuration in `config.json`

**LLM enrichment failures:**
- Ensure Ollama is running with the configured model
- Check model availability: `ollama list`
- Verify LLM configuration in `config.json`

### Environment Variables

Create a `.env` file for custom settings:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rushhour2
LOG_LEVEL=INFO
CORS_ALLOW_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## 🔌 Expandability

RushHour2 is designed with modularity in mind and can be easily extended to scrape other apartment listing sites:

### Current Support
- **StreetEasy** (NYC) - Fully implemented with comprehensive data extraction

### Planned Expansions
- **Apartments.com** - National apartment listings
- **Zillow Rentals** - Broader market coverage
- **RentSpider** - Additional NYC listings
- **Custom Sites** - Easy to add new scrapers

### Adding New Sites

The scraping architecture is modular and site-agnostic:

1. **Create a new scraper module** in `src/scrapers/`
2. **Implement the base scraper interface** with site-specific logic
3. **Add site configuration** to `config/config.json`
4. **Update the main scraper** to include your new site
5. **Test and validate** data extraction

Example structure for new sites:
```python
# src/scrapers/apartments_com.py
class ApartmentsComScraper(BaseScraper):
    def extract_listing_data(self, page):
        # Site-specific extraction logic
        pass
    
    def handle_anti_bot(self, page):
        # Site-specific anti-bot handling
        pass
```

The core infrastructure (database, OCR, LLM, scoring) works with any apartment listing data, making it easy to expand to new markets and sites.

## 📊 Data Flow

1. **Scraping**: Collect listings from supported sites with anti-bot protection
2. **Commute Analysis**: Calculate travel times to your workplace
3. **OCR Processing**: Extract square footage and dimensions from listing images
4. **LLM Enhancement**: Use AI to fill in missing property details
5. **Scoring**: Apply multi-factor scoring algorithm
6. **Storage**: Save enriched data to PostgreSQL
7. **Display**: Present results through web interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational and personal use. Please respect StreetEasy's terms of service and robots.txt when scraping.

## ⚠️ Disclaimer

This tool is designed for personal apartment hunting in NYC. Users are responsible for complying with all applicable terms of service and local laws. The authors are not responsible for any misuse of this software.

---

**Happy apartment hunting! 🏠✨**