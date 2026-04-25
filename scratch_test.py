import osmnx as ox
from pathlib import Path
import sys

# Add parent to sys.path to import config
sys.path.append(str(Path.cwd()))
import config

print(f"Testing Overpass at: {config.OSM_OVERPASS_URL}")

ox.settings.overpass_endpoint = config.OSM_OVERPASS_URL
ox.settings.timeout = 30
ox.settings.use_cache = False

try:
    # Test a point in Mumbai (Malabar Hill)
    G = ox.graph_from_point((18.9533, 72.7983), dist=500, network_type="drive")
    print("SUCCESS: Graph fetched!")
    print(f"Nodes: {len(G.nodes())}, Edges: {len(G.edges())}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
