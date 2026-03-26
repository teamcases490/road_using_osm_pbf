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
warnings.filterwarnings("ignore", category=UserWarning)
ox.settings.log_console = False
ox.settings.use_cache = True
_SCRIPT_DIR = Path(__file__).parent.absolute()
_DATA_DIR   = _SCRIPT_DIR.parent / "data"
ox.settings.cache_folder = str(_DATA_DIR / "cache")
ox.settings.overpass_settings = '[out:json][timeout:90][date:"2025-03-01T00:00:00Z"]'
logging.basicConfig(level=logging.INFO, format="%(message)s")

WATER_RASTER = _DATA_DIR / "input" / "merged_gsw_compressed.tif"
CHECKPOINT_FILE = "processing_checkpoint.pkl"
CHECKPOINT_INTERVAL = 10

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
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return f"EPSG:{epsg}"


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
    results = {"address": address, "lat": lat, "lon": lon, "extraction_error": ""}
    if pd.isna(lat) or pd.isna(lon):
        results["extraction_error"] = "Invalid coordinates: NaN"
        return results
    utm_crs = get_utm_crs(lat, lon)
    max_radius = max(radii)
    base_wgs = gpd.GeoSeries([Point(lon, lat)], crs="EPSG:4326").iloc[0]
    base_utm = gpd.GeoSeries([base_wgs], crs="EPSG:4326").to_crs(utm_crs).iloc[0]
    buffers_utm = {r: base_utm.buffer(r) for r in radii}
    buffers_wgs = {
        r: gpd.GeoSeries([buffers_utm[r]], crs=utm_crs).to_crs("EPSG:4326").iloc[0]
        for r in radii
    }
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
def save_checkpoint(results, index):
    with open(CHECKPOINT_FILE, 'wb') as f:
        pickle.dump({'results': results, 'last_index': index}, f)
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'rb') as f: data = pickle.load(f)
            return data['results'], data['last_index']
        except Exception as e: logging.warning(f"Could not load checkpoint: {e}")
    return [], -1
# ---------------------------------------------------------
# CSV Input Reader
# ---------------------------------------------------------
def read_input_csv(csv_file):
    try:
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                logging.info(f"Successfully read CSV with encoding: {encoding}")
                break
            except (UnicodeDecodeError, UnicodeError): continue
        if df is None: raise ValueError("Could not read CSV with any standard encoding")
        address_col = lat_col = lon_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ['address', 'location', 'name', 'place']: address_col = col
            elif col_lower in ['lat', 'latitude']: lat_col = col
            elif col_lower in ['lon', 'long', 'longitude']: lon_col = col
        if lat_col is None or lon_col is None:
            raise ValueError(f"CSV must contain latitude and longitude columns. Found: {list(df.columns)}")
        coords = []
        for idx, row in df.iterrows():
            try:
                lat, lon = float(row[lat_col]), float(row[lon_col])
                address = str(row[address_col]) if address_col and pd.notna(row[address_col]) else f"Location_{idx}"
                coords.append((address, lat, lon))
            except Exception as e:
                logging.warning(f"Row {idx}: Invalid coordinates - {e}")
                continue
        return coords
    except Exception as e:
        logging.error(f"Error reading CSV: {e}"); sys.exit(1)
# ---
def preprocess_road_data(df):
    print("\n" + "=" * 60 + "\nPREPROCESSING PIPELINE\n" + "=" * 60)
    initial_rows = len(df)
    df = df.dropna(how='all')
    if 'lat' in df.columns and 'lon' in df.columns: df = df.dropna(subset=['lat', 'lon'])
    df = df.reset_index(drop=True)
    if (removed := initial_rows - len(df)) > 0:
        print(f"  Cleaned: removed {removed} empty/invalid rows, {len(df)} remaining")
    radii = [col.split('_')[-1] for col in df.columns if col.startswith('total_road_km_')]
    if not radii:
        print("  WARNING: No radius columns found — skipping preprocessing.")
        return df
    print(f"  Detected radii: {radii}")
    print("\n[PHASE 1] Handling missing values...")
    for r in radii:
        cols = [f'total_road_km_{r}', f'Density_{r}', f'IntersectionDensity_{r}', f'AvgNodeDegree_{r}',
                f'lane_km_per_km2_{r}', f'road_area_per_km2_{r}', f'DeadEndRatio_{r}', f'avg_segment_length_{r}']
        missing = [c for c in cols if c not in df.columns]
        if missing: continue
        df[f'total_road_km_{r}'] = df[f'total_road_km_{r}'].fillna(0.0).clip(lower=0)
        df[f'is_void_{r}'] = (df[f'total_road_km_{r}'] == 0)
        for col in [f'Density_{r}', f'IntersectionDensity_{r}', f'lane_km_per_km2_{r}', f'road_area_per_km2_{r}']:
            df.loc[df[f'is_void_{r}'], col] = 0.0
            df[col] = df[col].fillna(0.0).clip(lower=0)
        for col, default in [(f'AvgNodeDegree_{r}', 0.0), (f'DeadEndRatio_{r}', 0.0), (f'avg_segment_length_{r}', 100.0)]:
            df.loc[df[f'is_void_{r}'], col] = 0.0
            med = df.loc[~df[f'is_void_{r}'], col].median()
            df[col] = df[col].fillna(med if not pd.isna(med) else default)
        print(f"  ✓ Radius {r}: {int(df[f'is_void_{r}'].sum())} void areas detected")
    print("\n[PHASE 2] Applying void masks...")
    for r in radii:
        if f'is_void_{r}' not in df.columns: continue
        mask_cols = [f'total_road_km_{r}', f'Density_{r}', f'IntersectionDensity_{r}', f'AvgNodeDegree_{r}',
                     f'lane_km_per_km2_{r}', f'road_area_per_km2_{r}', f'DeadEndRatio_{r}', f'avg_segment_length_{r}']
        for col in mask_cols:
            if col in df.columns: df.loc[df[f'is_void_{r}'], col] = 0.0
    print("\n[PHASE 3] Engineering advanced features...")
    for r in radii:
        if not all(c in df.columns for c in [f'Density_{r}', f'IntersectionDensity_{r}', f'AvgNodeDegree_{r}', f'DeadEndRatio_{r}']): continue
        df[f'Grid_Complexity_{r}'] = df[f'IntersectionDensity_{r}'] / (1 + df[f'DeadEndRatio_{r}'])
        df[f'Major_Road_Connectivity_{r}'] = sum(df[c] for c in [f'share_motorway_{r}', f'share_trunk_{r}', f'share_primary_{r}', f'share_secondary_{r}'] if c in df.columns)
        df[f'Local_Access_Intensity_{r}'] = sum(df[c] for c in [f'share_residential_{r}', f'share_living_street_{r}'] if c in df.columns)
        df[f'Informal_Proxy_{r}'] = sum(df[c] for c in [f'share_unclassified_{r}', f'share_service_{r}'] if c in df.columns)
        df[f'Network_Capacity_{r}'] = df[f'Density_{r}'] * df[f'AvgNodeDegree_{r}']
        df[f'Congestion_Risk_{r}'] = (df[f'Density_{r}'] * df[f'DeadEndRatio_{r}'] * df[f'Informal_Proxy_{r}'])
        for col in [f'Grid_Complexity_{r}', f'Major_Road_Connectivity_{r}', f'Local_Access_Intensity_{r}', f'Informal_Proxy_{r}', f'Network_Capacity_{r}', f'Congestion_Risk_{r}']:
            if col in df.columns: df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        # 7. POI Infrastructure Score (bus stops + traffic signals)
        poi_bus, poi_signals = f'poi_bus_stop_{r}', f'poi_traffic_signals_{r}'
        bus_series = df[poi_bus] if poi_bus in df.columns else pd.Series(0, index=df.index)
        sig_series = df[poi_signals] if poi_signals in df.columns else pd.Series(0, index=df.index)
        
        area_km2_pi = np.pi * (int(r)/1000.0)**2
        df[f'POI_Infrastructure_{r}'] = ((bus_series * 0.6 + sig_series * 0.4) / area_km2_pi).replace([np.inf, -np.inf], 0).fillna(0)
        for col in [f'AvgBetweenness_{r}', f'MaxBetweenness_{r}', f'AvgCloseness_{r}']:
            if col in df.columns: df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
    print("\n" + "=" * 60 + "\nPREPROCESSING COMPLETE\n" + "=" * 60)
    return df
def process_wrapper(args_tuple):
    try:
        return process_point(args_tuple[1], args_tuple[2], args_tuple[0], verbose=False)
    except Exception as e:
        return {"address": args_tuple[0], "lat": args_tuple[1], "lon": args_tuple[2], "error": str(e)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Feature Extraction + Preprocessing")
    parser.add_argument("--input",  "-i", type=str, default="data/input/locations.csv")
    parser.add_argument("--output", "-o", type=str, default="data/output/raw_features.csv")
    parser.add_argument("--preprocessed-output", "-pp", type=str, default="data/output/preprocessed_features.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-fresh", action="store_true")
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--skip-preprocessing", action="store_true")
    args = parser.parse_args()
    
    OUTPUT_CSV = args.output
    PREPROCESSED_CSV = args.preprocessed_output if args.preprocessed_output else str(Path(OUTPUT_CSV).parent / (Path(OUTPUT_CSV).stem + "_preprocessed" + Path(OUTPUT_CSV).suffix))
    
    logging.info(f"  Water raster: {WATER_RASTER} {'✓' if WATER_RASTER.exists() else 'NOT FOUND'}")
    inputs = read_input_csv(args.input)[:args.limit] if args.limit else read_input_csv(args.input)
    if not inputs: 
        logging.error("No valid coordinates found.")
        sys.exit(1)
        
    all_results, start_index = [], 0
    if os.path.exists(CHECKPOINT_FILE):
        should_resume = args.resume if args.resume or args.force_fresh else False
        if not (args.resume or args.force_fresh):
            try: should_resume = (input("\n🔄 Checkpoint found. Resume? (y/n): ").strip().lower() == 'y')
            except EOFError: should_resume = False
        if should_resume: 
            all_results, last_index = load_checkpoint()
            start_index = last_index + 1
        elif os.path.exists(CHECKPOINT_FILE): 
            os.remove(CHECKPOINT_FILE)
            
    logging.info("\n" + "="*60 + "\nRAW FEATURE EXTRACTION\n" + "="*60)
    num_workers = (args.workers if args.workers else max(1, cpu_count() - 1)) if args.parallel else 1
    
    if start_index == 0 and inputs:
        all_results.append(process_point(inputs[0][1], inputs[0][2], inputs[0][0], verbose=True))
        start_index = 1
        
    remaining = inputs[start_index:]
    
    if num_workers > 1 and remaining:
        with Pool(processes=num_workers) as pool:
            for i, result in enumerate(tqdm(pool.imap(process_wrapper, remaining), total=len(remaining), desc="Processing")):
                all_results.append(result)
                if ((start_index+i+1) % CHECKPOINT_INTERVAL == 0) or ((start_index+i+1) == len(inputs)):
                    pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
                    save_checkpoint(all_results, start_index+i)
    elif remaining:
        for i, (addr, lat, lon) in enumerate(tqdm(remaining, desc="Processing")):
            all_results.append(process_point(lat, lon, addr, verbose=False))
            if ((start_index+i+1) % CHECKPOINT_INTERVAL == 0) or ((start_index+i+1) == len(inputs)):
                pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
                save_checkpoint(all_results, start_index+i)
                
    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(OUTPUT_CSV, index=False)
        
    if not args.skip_preprocessing:
        df_pp = preprocess_road_data(df_raw.copy())
        df_pp.to_csv(PREPROCESSED_CSV, index=False)
        logging.info(f"✅ Saved → {PREPROCESSED_CSV}")
        
    logging.info(f"✅ Saved → {OUTPUT_CSV}\n⏱️ Time/point: ~10-30s")
    logging.info(f"⏱️  Est. total for {len(inputs)} points: ~{len(inputs) * 20 / (3600 * num_workers):.1f} hours")