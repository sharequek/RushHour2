## RushHour2 Streeteasy Scraper (Python)

Quick start without manually activating a venv each time:

1. Bring up Postgres (Docker Desktop must be running):

```powershell
docker compose up -d
```

2. One-time setup (creates `.venv`, installs deps and browsers):

```powershell
./scripts/setup.ps1
```

3. Configure defaults in `config/config.json` (already set to your long URL). Run the scraper without passing the URL each time:

```powershell
./scripts/scrape.ps1 -Concurrency 3 -MaxPages 2 -MaxLinks 50
```

Omit `-MaxPages` to crawl all pages by default. Configure DB via `DATABASE_URL` in a `.env` file if needed. The scraper runs fully headed and waits on PerimeterX challenges until you complete them.

### Scrape a single detail page (for validation)

```powershell
./scripts/scrape.ps1 -DetailUrl "https://streeteasy.com/building/eight80-880-atlantic-avenue-brooklyn/7f?featured=1"
# or
./scripts/scrape.ps1 -DetailUrl "https://streeteasy.com/rental/1234567"
```

### Run tests

```powershell
./scripts/test.ps1
```

### Troubleshooting

- If browsers aren’t installed, re-run:

```powershell
./scripts/setup.ps1
```

### Admin

Clear the listings table quickly during development:

```powershell
./scripts/db.ps1 -Clear
```



