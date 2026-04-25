import pathlib as _pl
# import os as _os # Removed for local-only focus

# API Configuration
# OSM_OVERPASS_URL = "https://overpass-api.de/api/interpreter"  # Public API
OSM_OVERPASS_URL = "http://127.0.0.1:12345/api"  # OSMnx 2.x appends /interpreter automatically

# Paths
_ROOT = _pl.Path(__file__).resolve().parent
CACHE_DIR = str(_ROOT / "data" / "cache")
LOG_DIR = str(_ROOT / "data" / "logs")

# OSMnx Settings
OVERPASS_TIMEOUT = 90
# We use a broader setting for local to avoid date mismatch errors with PBF snapshots
OVERPASS_SETTINGS = '[out:json][timeout:90]'
