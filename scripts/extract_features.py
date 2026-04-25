import osmnx as ox
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import logging
import warnings
from tqdm import tqdm
import sys
import io
import pickle
import os
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from contextlib import nullcontext

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import rasterio
    import rasterio.mask
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logging.warning("Rasterio not found. Water correction will be disabled.")

# ---------------------------------------------------------
# Add parent directory to sys.path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

# ---------------------------------------------------------
warnings.filterwarnings("ignore", category=UserWarning)
ox.settings.log_console = False
ox.settings.use_cache = True
_SCRIPT_DIR = Path(__file__).parent.absolute()
_DATA_DIR   = _SCRIPT_DIR.parent / "data"
ox.settings.cache_folder = str(_DATA_DIR / "cache")

# Lock every worker process to local Docker — called at module import AND inside each worker
def _configure_osmnx():
    """Lock OSMnx 2.x to the local Docker Overpass instance."""
    # OSMnx 2.x source: url = settings.overpass_url.rstrip("/") + "/interpreter"
    # So overpass_url must be the BASE (e.g. http://127.0.0.1:12345/api), NOT the full path.
    base_url = config.OSM_OVERPASS_URL.rstrip("/")
    if base_url.endswith("/interpreter"):
        base_url = base_url[: -len("/interpreter")]

    ox.settings.overpass_url = base_url

    # CRITICAL: Local Docker reports "Rate limit: 0" which OSMnx interprets as
    # "0 slots available" and polls /api/status forever. Disable rate limiting.
    try:
        ox.settings.overpass_rate_limit = False
    except AttributeError:
        pass

    try:
        ox.settings.timeout = config.OVERPASS_TIMEOUT
    except AttributeError:
        pass

    if not hasattr(_configure_osmnx, "_logged"):
        logging.info(f"Overpass → {base_url}/interpreter  [rate_limit=off]")
        _configure_osmnx._logged = True

_configure_osmnx()
logging.basicConfig(level=logging.INFO, format="%(message)s")

WATER_RASTER = _DATA_DIR / "input" / "merged_gsw_compressed.tif"
# Checkpoint logic refactored to use CSV instead of .pkl below

# ---------------------------------------------------------
# IRC-BASED LANE DEFAULTS
# ---------------------------------------------------------
IRC_LANES_BY_HW = {
    'motorway': 4, 'trunk': 4, 'primary': 2, 'secondary': 2,
    'tertiary': 2, 'residential': 2, 'unclassified': 1,
    'service': 1, 'living_street': 1,
    'motorway_link': 2, 'trunk_link': 2, 'primary_link': 1,
    'secondary_link': 1, 'tertiary_link': 1
}

IRC_LANE_WIDTH = {
    'motorway': 3.5, 'trunk': 3.5, 'primary': 3.5, 'secondary': 3.5,
    'tertiary': 3.5, 'residential': 3.5, 'unclassified': 3.75,
    'service': 3.5, 'living_street': 3.0
}

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def get_utm_crs(lat, lon):
    try:
        # Basic bounds checking
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return "EPSG:3857" # Fallback to Web Mercator if invalid
        zone = int((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{epsg}"
    except Exception:
        return "EPSG:3857"


def normalize_highway(val):
    if val is None: return []
    if isinstance(val, (list, tuple, set)): return [str(v) for v in val if v is not None]
    if isinstance(val, str):
        for delim in [";", ",", "|"]:
            if delim in val: return [p.strip() for p in val.split(delim) if p.strip()]
        return [val.strip()]
    return [str(val)]


def get_water_percent(buffer_wgs, src=None):
    try:
        if not HAS_RASTERIO: return 0.0
        if src is None:
            with rasterio.open(str(WATER_RASTER)) as s:
                return _do_water_pct(buffer_wgs, s)
        return _do_water_pct(buffer_wgs, src)
    except Exception: return 0.0
def _do_water_pct(buffer_wgs, src):
    try:
        buf_proj = gpd.GeoSeries([buffer_wgs], crs="EPSG:4326").to_crs(src.crs)
        out_img, _ = rasterio.mask.mask(src, buf_proj.geometry, crop=True, filled=False)
        data = out_img[0]
        valid = data[data >= 0]
        if valid.size == 0: return 0.0
        return (np.sum(valid >= 50) / valid.size) * 100.0
    except Exception: return 0.0
def get_land_area_features(buffer_wgs, utm_crs):
    try:
        buffer_utm = gpd.GeoSeries([buffer_wgs], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
        return max(buffer_utm.area / 1e6, 0.001)
    except Exception: return 0.001
# ---------------------------------------------------------
# Feature Computation
# ---------------------------------------------------------
def compute_features_from_gdfs(nodes_utm, edges_utm, buffer_utm, radius, area_km2, base_utm=None):
    empty_features = {
        f"total_road_km_{radius}": 0.0,
        f"Density_{radius}": 0.0,
        f"IntersectionDensity_{radius}": 0.0,
        f"AvgNodeDegree_{radius}": np.nan,
        f"DeadEndRatio_{radius}": np.nan,
        f"lane_km_per_km2_{radius}": 0.0,
        f"road_area_per_km2_{radius}": 0.0,
        f"avg_segment_length_{radius}": np.nan,
        f"pct_lanes_estimated_{radius}": np.nan,
        **{f"share_{t}_{radius}": 0.0 for t in [
            'motorway', 'trunk', 'primary', 'secondary', 'tertiary',
            'residential', 'living_street', 'service', 'unclassified'
        ]}
    }
    if area_km2 <= 0 or edges_utm is None or edges_utm.empty:
        return empty_features
    try:
        if 'geometry' in edges_utm.columns:
            edges_utm = edges_utm.set_geometry('geometry')
        else:
            return empty_features
    except Exception: return empty_features
    features = empty_features.copy()
    if "length" not in edges_utm.columns:
        edges_utm["length"] = edges_utm.geometry.length
    edges_utm = edges_utm[edges_utm["length"] > 0].copy()
    if edges_utm.empty: return features
    total_length_km = edges_utm["length"].sum() / 1000.0
    features[f"total_road_km_{radius}"] = total_length_km
    features[f"Density_{radius}"] = total_length_km / area_km2
    try:
        G = ox.graph_from_gdfs(nodes_utm, edges_utm)
        G_topo = ox.simplify_graph(G)
        G_final = ox.convert.to_undirected(G_topo)
    except Exception as e:
        logging.warning(f"Graph topology build failed for radius {radius}: {e}")
        G_final = None
    if G_final is not None:
        try:
            deg = dict(G_final.degree())
            deg_values = list(deg.values())
            intersec_nodes = [n for n, d in deg.items() if d >= 3]
            features[f"IntersectionDensity_{radius}"] = len(intersec_nodes) / area_km2
            features[f"AvgNodeDegree_{radius}"] = np.mean(deg_values) if deg_values else np.nan
            if base_utm is not None:
                nodes_topo_df = ox.graph_to_gdfs(G_topo, edges=False)
                dead_ends_internal = 0
                center_pt = buffer_utm.centroid
                for node, d in deg.items():
                    if d == 1:
                        node_geom = nodes_topo_df.loc[node, 'geometry']
                        if node_geom.distance(center_pt) <= (radius - 15):
                            dead_ends_internal += 1
                dead_ends = dead_ends_internal
            else:
                dead_ends = sum(1 for d in deg.values() if d == 1)
            features[f"DeadEndRatio_{radius}"] = dead_ends / len(deg_values) if deg_values else np.nan
        except Exception as e:
            logging.warning(f"Feature calculation error: {e}")
            features[f"IntersectionDensity_{radius}"] = np.nan
            features[f"AvgNodeDegree_{radius}"] = np.nan
            features[f"DeadEndRatio_{radius}"] = np.nan
    edges_utm["highway_norm"] = edges_utm.get("highway", np.nan).apply(
        lambda x: normalize_highway(x) if pd.notna(x) else []
    )
    def parse_lanes(x, hw_list):
        if pd.notna(x):
            val = x
            if isinstance(x, (list, tuple, set)): val = list(x)[0] if x else None
            if isinstance(val, str):
                for delim in [",", ";", "|", " "]:
                    if delim in val:
                        val = val.split(delim)[0].strip()
                        break
            try:
                n = int(float(val))
                if n > 0: return n, False
            except Exception: pass
        for hw in hw_list:
            if hw in IRC_LANES_BY_HW: return IRC_LANES_BY_HW[hw], True
        return 1, True
    lanes_parsed = []
    lanes_estimated_flags = []
    for idx, row in edges_utm.iterrows():
        hw_list = row["highway_norm"] if isinstance(row["highway_norm"], list) else []
        lanes_val, estimated_flag = parse_lanes(row.get("lanes"), hw_list)
        lanes_parsed.append(lanes_val)
        lanes_estimated_flags.append(1 if estimated_flag else 0)
    edges_utm["lanes_parsed"] = lanes_parsed
    edges_utm["lanes_estimated_flag"] = lanes_estimated_flags
    lane_km = (edges_utm["length"] * edges_utm["lanes_parsed"]).sum() / 1000.0
    features[f"lane_km_per_km2_{radius}"] = lane_km / area_km2
    features[f"pct_lanes_estimated_{radius}"] = (
        edges_utm["lanes_estimated_flag"].sum() / len(edges_utm) if len(edges_utm) > 0 else np.nan
    )
    def get_irc_width(hw_list):
        for hw in hw_list:
            if hw in IRC_LANE_WIDTH: return IRC_LANE_WIDTH[hw]
        return 3.5
    edges_utm["irc_width"] = edges_utm["highway_norm"].apply(get_irc_width)
    edges_utm["road_width"] = edges_utm["lanes_parsed"] * edges_utm["irc_width"]
    road_area = (edges_utm["length"] * edges_utm["road_width"]).sum() / 1e6
    features[f"road_area_per_km2_{radius}"] = road_area / area_km2
    features[f"avg_segment_length_{radius}"] = edges_utm["length"].mean()
    hierarchy_categories = {
        'motorway':    ['motorway',    'motorway_link'],
        'trunk':       ['trunk',       'trunk_link'],
        'primary':     ['primary',     'primary_link'],
        'secondary':   ['secondary',   'secondary_link'],
        'tertiary':    ['tertiary',    'tertiary_link'],
        'residential': ['residential'],
        'living_street':['living_street'],
        'service':     ['service'],
        'unclassified':['unclassified']
    }
    total_len = edges_utm["length"].sum()
    shares = {}
    for parent, types in hierarchy_categories.items():
        mask = edges_utm["highway_norm"].apply(lambda L: any(t in L for t in types))
        length_for_type = edges_utm.loc[mask, "length"].sum()
        shares[parent] = length_for_type / total_len if total_len > 0 else 0.0
    total_share = sum(shares.values())
    if total_share > 0:
        shares = {k: v / total_share for k, v in shares.items()}
    for parent, share_val in shares.items():
        features[f"share_{parent}_{radius}"] = share_val
    return features
def process_point(lat, lon, address=None, radii=[500, 1000, 2000], verbose=False):
    # Ensure local config in worker process
    _configure_osmnx()
    results = {
        "address": address, 
        "lat": lat, 
        "lon": lon, 
        "status": "Success",
        "extraction_error": ""
    }
    if pd.isna(lat) or pd.isna(lon):
        results["extraction_error"] = "Invalid coordinates: NaN"
        return results
    try:
        utm_crs = get_utm_crs(lat, lon)
        max_radius = max(radii)
        base_wgs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").iloc[0]
        base_utm = gpd.GeoSeries([base_wgs], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
        buffers_utm = {r: base_utm.buffer(r) for r in radii}
        buffers_wgs = {
            r: gpd.GeoSeries([buffers_utm[r]], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
            for r in radii
        }
    except Exception as e:
        results["extraction_error"] = f"Geometry initialization failed: {e}"
        results["status"] = "Error"
        # Return empty features for all radii
        for r in radii:
            results.update(compute_features_from_gdfs(gpd.GeoDataFrame(), gpd.GeoDataFrame(), None, r, 0.001, None))
        return results
    try:
        if verbose: logging.info(f"Fetching road graph at max radius {max_radius}m (single API call)...")
        G_max = ox.graph_from_polygon(buffers_wgs[max_radius], network_type="drive", simplify=False, retain_all=True)
        nodes_wgs_max, edges_wgs_max = ox.graph_to_gdfs(G_max)
        nodes_utm_max = nodes_wgs_max.to_crs(utm_crs)
        edges_utm_max = edges_wgs_max.to_crs(utm_crs)
        nodes_utm_max['x'] = nodes_utm_max.geometry.x
        nodes_utm_max['y'] = nodes_utm_max.geometry.y
        fetch_failed = False
    except Exception as e:
        err_msg = f"Graph fetch failed: {type(e).__name__}: {str(e)[:200]}"
        logging.error(f"Point ({lat},{lon}) {err_msg}")
        results["extraction_error"] = err_msg
        fetch_failed = True
    raster_ctx = rasterio.open(str(WATER_RASTER)) if HAS_RASTERIO and WATER_RASTER.exists() else nullcontext()
    with raster_ctx as src:
        for r in radii:
            buffer_utm = buffers_utm[r]
            buffer_wgs = buffers_wgs[r]
            water_pct = get_water_percent(buffer_wgs, src)
            results[f"water_percent_{r}"] = water_pct
            area_km2 = get_land_area_features(buffer_wgs, utm_crs)
            if water_pct > 25:
                area_km2 = max(area_km2 * (1 - water_pct / 100.0), 0.0001)
            results[f"effective_area_km2_{r}"] = area_km2
            if verbose: logging.info(f"  Radius {r}m — water%: {water_pct:.2f}, area={area_km2:.4f} km²")
            if fetch_failed:
                results.update(compute_features_from_gdfs(gpd.GeoDataFrame(), gpd.GeoDataFrame(), buffer_utm, r, 0.001, base_utm))
                continue
            try:
                edges_utm = edges_utm_max.copy()
                edges_utm["geometry"] = edges_utm.geometry.intersection(buffer_utm)
                edges_utm = edges_utm[~edges_utm.geometry.is_empty].copy()
                edges_utm["length"] = edges_utm.geometry.length
                if (isinstance(edges_utm.index, pd.MultiIndex) and list(edges_utm.index.names[:2]) == ['u', 'v']):
                    used_node_ids = set(edges_utm.index.get_level_values('u')) | set(edges_utm.index.get_level_values('v'))
                else: used_node_ids = set()
                nodes_utm = nodes_utm_max[nodes_utm_max.index.isin(used_node_ids)].copy()
                if nodes_utm.empty or edges_utm.empty:
                    results.update(compute_features_from_gdfs(gpd.GeoDataFrame(), gpd.GeoDataFrame(), buffer_utm, r, area_km2, base_utm))
                    continue
                results.update(compute_features_from_gdfs(nodes_utm, edges_utm, buffer_utm, r, area_km2, base_utm))
            except Exception as e:
                err_msg = f"clip/compute failed: {type(e).__name__}: {str(e)[:200]}"
                logging.error(f"({lat},{lon}) r={r} {err_msg}")
                if not results["extraction_error"]: results["extraction_error"] = err_msg
                results.update(compute_features_from_gdfs(gpd.GeoDataFrame(), gpd.GeoDataFrame(), buffer_utm, r, area_km2, base_utm))
    return results
# ---
def _load_processed_set(csv_path: Path) -> set:
    """Return set of (lat, lon) already in the output CSV."""
    if not csv_path.exists(): return set()
    try:
        # We use lat/lon as the unique key for resume
        df = pd.read_csv(csv_path, usecols=["lat", "lon"])
        return {(round(r.lat, 6), round(r.lon, 6)) for r in df.itertuples()}
    except Exception: return set()

def process_wrapper(args_tuple):
    """Worker wrapper for process_point."""
    try:
        return process_point(args_tuple[1], args_tuple[2], args_tuple[0], verbose=False)
    except Exception as e:
        return {
            "address": args_tuple[0], 
            "lat": args_tuple[1], 
            "lon": args_tuple[2], 
            "status": "Error",
            "extraction_error": str(e)
        }

def read_input_csv(csv_file):
    """Read coordinates from input CSV with encoding detection."""
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                break
            except Exception: continue
        if df is None: raise ValueError("Could not read CSV with any standard encoding")
        
        # Detect columns
        lat_col = lon_col = addr_col = None
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ['lat', 'latitude']: lat_col = c
            elif cl in ['lon', 'long', 'longitude']: lon_col = c
            elif cl in ['address', 'name', 'place', 'location']: addr_col = c
            
        if not lat_col or not lon_col:
            raise ValueError("CSV must have 'lat' and 'lon' columns.")
            
        points = []
        for i, row in df.iterrows():
            addr = str(row[addr_col]) if addr_col and pd.notna(row[addr_col]) else f"Point_{i}"
            points.append((addr, float(row[lat_col]), float(row[lon_col])))
        return points
    except Exception as e:
        logging.error(f"Failed to read input CSV: {e}")
        sys.exit(1)

def preprocess_road_data(df):
    """Clean, fill, and engineer features from extracted road data."""
    radii = [500, 1000, 2000]
    for r in radii:
        # Fill raw extracted columns
        df[f'total_road_km_{r}']       = df.get(f'total_road_km_{r}', 0).fillna(0.0)
        df[f'Density_{r}']             = df.get(f'Density_{r}', 0).fillna(0.0)
        df[f'IntersectionDensity_{r}'] = df.get(f'IntersectionDensity_{r}', 0).fillna(0.0)
        df[f'AvgNodeDegree_{r}']       = df.get(f'AvgNodeDegree_{r}', 0).fillna(0.0)
        df[f'DeadEndRatio_{r}']        = df.get(f'DeadEndRatio_{r}', 0).fillna(0.0)
        df[f'lane_km_per_km2_{r}']     = df.get(f'lane_km_per_km2_{r}', 0).fillna(0.0)
        df[f'road_area_per_km2_{r}']   = df.get(f'road_area_per_km2_{r}', 0).fillna(0.0)

        for road_type in ['motorway', 'trunk', 'primary', 'secondary', 'tertiary',
                          'residential', 'living_street', 'service', 'unclassified']:
            df[f'share_{road_type}_{r}'] = df.get(f'share_{road_type}_{r}', 0).fillna(0.0)

        # ── Derived features ──────────────────────────────────────────────────

        # Grid Complexity: well-connected intersections penalised by dead-ends
        df[f'Grid_Complexity_{r}'] = (
            df[f'IntersectionDensity_{r}'] / (1 + df[f'DeadEndRatio_{r}'])
        ).fillna(0.0)

        # Informal Proxy: unclassified + service roads dominate informal layouts
        df[f'Informal_Proxy_{r}'] = (
            df[f'share_unclassified_{r}'] + df[f'share_service_{r}']
        ).clip(0, 1).fillna(0.0)

        # Network Capacity: throughput proxy = density × connectivity
        df[f'Network_Capacity_{r}'] = (
            df[f'Density_{r}'] * df[f'AvgNodeDegree_{r}']
        ).fillna(0.0)

        # Major Road Connectivity: share of backbone roads (motorway→secondary)
        df[f'Major_Road_Connectivity_{r}'] = (
            df[f'share_motorway_{r}'] + df[f'share_trunk_{r}'] +
            df[f'share_primary_{r}']  + df[f'share_secondary_{r}']
        ).clip(0, 1).fillna(0.0)

        # Congestion Risk: high density with poor connectivity = congestion
        # Uses safe division — 0 density → 0 risk
        df[f'Congestion_Risk_{r}'] = (
            df[f'Density_{r}'] / (df[f'AvgNodeDegree_{r}'].replace(0, np.nan))
        ).fillna(0.0)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Feature Extraction")
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--preprocessed-output", "-pp", required=True)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    
    OUTPUT_CSV = Path(args.output)
    PP_CSV = Path(args.preprocessed_output)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Water raster: {WATER_RASTER} {'OK' if WATER_RASTER.exists() else 'NOT FOUND'}")
    inputs = read_input_csv(args.input)[:args.limit] if args.limit else read_input_csv(args.input)
    
    # RESUME LOGIC: Check CSV instead of .pkl
    processed = _load_processed_set(OUTPUT_CSV) if args.resume else set()
    remaining = [pt for pt in inputs if (round(pt[1], 6), round(pt[2], 6)) not in processed]
    
    if not remaining:
        logging.info("All locations already processed. Nothing to do.")
        # If we need to regenerate the preprocessed file, we'd load the full CSV here
        sys.exit(0)

    num_workers = args.workers if args.workers else (min(8, max(1, cpu_count() - 1)) if args.parallel else 1)
    logging.info(f"Processing {len(remaining)}/{len(inputs)} locations | workers={num_workers} | resume={args.resume}")

    all_results = []
    # On resume, load previous successful results for final preprocessing
    if args.resume and OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 5:
        try:
            all_results = pd.read_csv(OUTPUT_CSV).to_dict('records')
        except Exception:
            all_results = []

    _csv_header_written = [OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 5]

    def _append_result(result: dict):
        """Append a result dict to the CSV, writing the header on the first call."""
        row_df = pd.DataFrame([result])
        if not _csv_header_written[0]:
            row_df.to_csv(OUTPUT_CSV, mode='w', header=True, index=False)
            _csv_header_written[0] = True
        else:
            row_df.to_csv(OUTPUT_CSV, mode='a', header=False, index=False)

    try:
        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                for result in tqdm(pool.imap(process_wrapper, remaining),
                                   total=len(remaining), desc="Roads"):
                    all_results.append(result)
                    _append_result(result)
        else:
            for addr, lat, lon in tqdm(remaining, desc="Roads"):
                result = process_point(lat, lon, addr)
                all_results.append(result)
                _append_result(result)
    except KeyboardInterrupt:
        logging.info("\nInterrupted. Progress saved to CSV.")
        sys.exit(0)
                
    # Phase 2: Preprocessing
    df_raw = pd.DataFrame(all_results)
    logging.info("Running preprocessing on all features...")
    df_pp = preprocess_road_data(df_raw.copy())
    df_pp.to_csv(PP_CSV, index=False)
    
    logging.info(f"Done! Raw features: {OUTPUT_CSV}")
    logging.info(f"Done! Preprocessed features: {PP_CSV}")