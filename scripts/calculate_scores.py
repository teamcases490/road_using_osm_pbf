"""
SCRIPT 2: ROAD SCORE CALCULATION

Input:  Preprocessed CSV (Absolute Values)
Output: CSV with Road Scores and categorized results
"""

import pandas as pd
import numpy as np
import time
import warnings
import argparse
warnings.filterwarnings('ignore')

# ============================================================================
# FILE CONFIGURATION
# ============================================================================

INPUT_FILE = "data/output/preprocessed_features.csv"
OUTPUT_FILE = "data/output/road_scores.csv"

# Component weights (should sum to 1.0)
WEIGHT_U = 0.45  # Urbanicity (Increased from 0.40)
WEIGHT_H = 0.35  # Hierarchy (Reduced from 0.40)
WEIGHT_C = 0.10  # Capacity
WEIGHT_Q = 0.10  # Quality 

# Multi-scale fusion weights 
WEIGHT_500  = 0.50   # Local
WEIGHT_1000 = 0.30  # Neighborhood
WEIGHT_2000 = 0.20  # Regional

# Urbanicity sub-weights 
WEIGHT_DENSITY         = 0.35
WEIGHT_GRID_COMPLEXITY = 0.45  
WEIGHT_NODE_DEGREE     = 0.20

# NEW: ABSOLUTE THRESHOLDS 
# Any value above these thresholds gets a score of 1.0
# Calibrated from dataset of 1010 Indian locations
ABSOLUTE_THRESHOLDS = {
    'Density': 12.0,              # km/km² (REFINED: >12 is dense urban)
    'IntersectionDensity': 80.0,  # count/km² (approx 8-9 per 100m block)
    'AvgNodeDegree': 3.5,         # Max connectivity
    'lane_km_per_km2': 100.0,     # lane-km/km² (95th percentile from dataset)
    'road_area_per_km2': 0.15,    # 15% of land area is road
    'Grid_Complexity': 60.0,      # REFINED: Lowered from 100 (more realistic)
    'Network_Capacity': 100.0,    # NEW: Density × NodeDegree threshold
    'Congestion_Risk': 5.0        # Threshold for slum-like areas
}

# Hierarchy weights 
HIERARCHY_WEIGHTS = {
    'motorway': 1.00, 'trunk': 1.00, 'primary': 0.85,
    'secondary': 0.40, 'tertiary': 0.10, 'residential': 0.02,
    'living_street': 0.01, 'service': 0.01, 'unclassified': 0.01
}

# Category thresholds 
CATEGORIES = {
    'Rural': (0.00, 0.35),
    'Urban': (0.35, 0.59),
    'Metro': (0.59, 1.01)
}

# Gate Thresholds
# Scale Gate removed — was incorrectly capping urban residential areas
HIERARCHY_GATE_THRESHOLD = 0.05  # Min share of major roads to avoid penalty
HIERARCHY_GATE_PENALTY   = 0.5
SPRAWL_PENALTY_EXPONENT  = 0.3   # Reduced from 0.5 to help coastal/hilly areas

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_absolute(series, threshold):
    """Normalize absolute values against a defined physical threshold."""
    return (series / threshold).clip(0, 1)

def calculate_urbanicity(df, radius):
    """Calculate Urbanicity using Grid Complexity Integration
    
    Solves the 'Pune Paradox' by using Grid_Complexity instead of separate
    Intersection and DeadEnd terms. This prevents high-density slums with
    dead ends from scoring too high.
    """
    suffix = f"_{radius}"
    
    # 1. Density (35%)
    dens = normalize_absolute(df.get(f"Density{suffix}", 0), ABSOLUTE_THRESHOLDS['Density'])
    
    # 2. Grid Complexity (45%) 
    # Grid_Complexity = IntersectionDensity / (1 + DeadEndRatio)
    # Naturally penalizes dead-end heavy areas
    grid = normalize_absolute(df.get(f"Grid_Complexity{suffix}", 0), ABSOLUTE_THRESHOLDS['Grid_Complexity'])
    
    # 3. Node Degree (20%)
    degree = normalize_absolute(df.get(f"AvgNodeDegree{suffix}", 0), ABSOLUTE_THRESHOLDS['AvgNodeDegree'])
    
    U = (WEIGHT_DENSITY * dens +
         WEIGHT_GRID_COMPLEXITY * grid +
         WEIGHT_NODE_DEGREE * degree)
    
    return U.clip(0, 1), dens  # Return density separately for other calculations

def calculate_hierarchy(df, radius, density_score):
    """Calculate Hierarchy using Weighted Sum of Road Classes.
    
    Weights road classes by importance. Posh residential is GOOD (0.5).
    Ensures zero density = zero hierarchy.
    """
    suffix = f"_{radius}"
    weights = {
        'motorway': 1.0, 'trunk': 1.0, 'primary': 0.8,
        'secondary': 0.6, 'tertiary': 0.4, 'residential': 0.5,
        'living_street': 0.5, 'unclassified': 0.2, 'service': 0.1
    }
    H = pd.Series(0.0, index=df.index)
    for cat, weight in weights.items():
        col = f"share_{cat}{suffix}"
        if col in df.columns:
            H += df[col] * weight
    return (H * (density_score * 2).clip(0, 1)).clip(0, 1)

def calculate_capacity(df, radius):
    """Calculate Capacity using Network Capacity (REFINED)
    
    Incorporates Network_Capacity (Density × NodeDegree) which measures
    how much traffic the grid can actually distribute.
    """
    suffix = f"_{radius}"
    
    # 1. Network Capacity (40%) - NEW: Measures distribution ability
    network_cap = normalize_absolute(df.get(f"Network_Capacity{suffix}", 0), 
                                    ABSOLUTE_THRESHOLDS['Network_Capacity'])
    
    # 2. Lane Capacity (30%)
    lane_cap = pd.Series(0.0, index=df.index)
    if f"lane_km_per_km2{suffix}" in df.columns:
        lane_cap = normalize_absolute(df[f"lane_km_per_km2{suffix}"], 
                                     ABSOLUTE_THRESHOLDS['lane_km_per_km2'])
    
    # 3. Road Area (30%)
    road_area = pd.Series(0.0, index=df.index)
    if f"road_area_per_km2{suffix}" in df.columns:
        road_area = normalize_absolute(df[f"road_area_per_km2{suffix}"], 
                                      ABSOLUTE_THRESHOLDS['road_area_per_km2'])
    
    C = (0.4 * network_cap + 0.3 * lane_cap + 0.3 * road_area)
    
    return C.clip(0, 1)

def calculate_quality(df, radius):
    """Calculate Quality component using engineered features
    
    Distinguishes formal metros from informal settlements using:
    - Grid Complexity (urban grid vs dead ends)
    - Major Road Connectivity (backbone infrastructure)
    - Informal Proxy (unplanned layouts)
    - Congestion Risk (dense slums)
    """
    suffix = f"_{radius}"
    
    # 1. Grid Complexity (40% weight)
    # Normalize against threshold (100 = very high urban complexity)
    grid = normalize_absolute(df.get(f"Grid_Complexity{suffix}", 0), 
                              ABSOLUTE_THRESHOLDS['Grid_Complexity'])
    
    # 2. Major Road Connectivity (30% weight)
    # Already 0-1 (sum of road class shares)
    major = df.get(f"Major_Road_Connectivity{suffix}", 0).clip(0, 1)
    
    # 3. Informal Penalty (15% weight)
    # High informal proxy = low quality
    informal_proxy = df.get(f"Informal_Proxy{suffix}", 0).clip(0, 1)
    informal_penalty = 1 - informal_proxy
    
    # 4. Congestion Risk Penalty (15% weight)
    # High congestion risk = slum-like = low quality
    congestion = normalize_absolute(df.get(f"Congestion_Risk{suffix}", 0),
                                    ABSOLUTE_THRESHOLDS['Congestion_Risk'])
    congestion_penalty = 1 - congestion
    
    Q = (0.4 * grid + 
         0.3 * major + 
         0.15 * informal_penalty + 
         0.15 * congestion_penalty)
    
    return Q.clip(0, 1)

def apply_hierarchy_gate(df, H, radius, major_roads_2000=None):
    """Apply hierarchy gate with regional context"""
    # Calculate local major road share
    local_major = pd.Series(0.0, index=df.index)
    for rclass in ['motorway', 'trunk', 'primary']:
        col = f"share_{rclass}_{radius}"
        if col in df.columns:
            local_major += df[col]

    H_gated = H.copy()
    
    # Logic: Penalize if NO major roads locally AND NO major roads regionally
    lacks_local = local_major < HIERARCHY_GATE_THRESHOLD
    
    if major_roads_2000 is not None:
        lacks_regional = major_roads_2000 < HIERARCHY_GATE_THRESHOLD
        penalty_mask = lacks_local & lacks_regional
    else:
        penalty_mask = lacks_local
        
    H_gated[penalty_mask] *= HIERARCHY_GATE_PENALTY
    return H_gated

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()
    
    print(f"\n[STARTING ROAD SCORING (Absolute Thresholds)]")
    print(f"   Input: {args.input}")
    
    try:
        df = pd.read_csv(args.input)
    except Exception as e:
        print(f"Error reading input: {e}")
        return

    # Pre-calc 2000m context
    major_2000 = pd.Series(0.0, index=df.index)
    for rc in ['motorway', 'trunk', 'primary']:
        col = f"share_{rc}_2000"
        if col in df.columns: 
            major_2000 += df[col]
    
    # Calculate per radius
    results = df.copy()
    
    for r in [500, 1000, 2000]:
        print(f"   - Processing {r}m radius...")
        # Components
        U, dens_score = calculate_urbanicity(df, r)
        H = calculate_hierarchy(df, r, dens_score)
        
        # Gates
        H = apply_hierarchy_gate(df, H, r, major_2000)
        
        C = calculate_capacity(df, r)
        Q = calculate_quality(df, r)  # NEW: Quality component
        
        # Fusion (now includes Quality)
        RS = (WEIGHT_U * U) + (WEIGHT_H * H) + (WEIGHT_C * C) + (WEIGHT_Q * Q)
        
        results[f"U_{r}"] = U
        results[f"H_{r}"] = H
        results[f"C_{r}"] = C
        results[f"Q_{r}"] = Q  # NEW: Store Quality scores
        results[f"RS_{r}"] = RS
        
    # Multi-scale Fusion
    print(f"   - Fusing scales (Weighted: {WEIGHT_500}/{WEIGHT_1000}/{WEIGHT_2000})...")
    RoadScore = (WEIGHT_500 * results['RS_500'] +
                 WEIGHT_1000 * results['RS_1000'] +
                 WEIGHT_2000 * results['RS_2000'])

    # Sprawl Penalty: areas with very low 2km density are likely rural/isolated
    dens_2000 = normalize_absolute(df['Density_2000'], ABSOLUTE_THRESHOLDS['Density'])
    # Gentler sprawl: exponent 0.2 + floor of 0.8 to prevent coastal crushing
    sprawl_factor = (dens_2000 ** 0.2).clip(lower=0.8)
    
    # Store intermediate values for debugging
    results['RS_Fusion'] = RoadScore.copy()
    results['Sprawl_Factor'] = sprawl_factor
    
    # Apply sprawl penalty
    RoadScore = RoadScore * sprawl_factor

    # Final Zero-Mask: No roads within 500m → score must be 0
    no_roads_mask = df['Density_500'] == 0
    RoadScore.loc[no_roads_mask] = 0.0

    results['RoadScore'] = RoadScore
    
    # ========================================================================
    # QUALITY GATEKEEPER (REFINED)
    # ========================================================================
    print(f"   - Applying Quality gatekeeper...")
    
    # Only flag if it's TRULY informal (High informal proxy + Low Quality)
    informal_proxy_500 = df.get('Informal_Proxy_500', 0)
    major_roads_500 = df.get('Major_Road_Connectivity_500', 0)
    quality_500 = results.get('Q_500', 1.0)
    congestion_risk = df.get('Congestion_Risk_500', 0)
    
    # Flag: Very high informal content OR (High density AND Low connectivity AND Low quality)
    informal_flag = (informal_proxy_500 > 0.6) | ((congestion_risk > 12.0) & (quality_500 < 0.25) & (major_roads_500 < 0.1))
    
    # Downgrade RoadScore for flagged locations
    results.loc[informal_flag, 'RoadScore'] = results.loc[informal_flag, 'RoadScore'] * 0.75
    
    flagged_count = informal_flag.sum()
    if flagged_count > 0:
        print(f"   ! Quality gatekeeper flagged {flagged_count} locations as potentially informal")
    
    # Categorize
    def get_cat(score, is_flagged):
        """Categorize with quality check"""
        # If flagged as informal, cap at Urban
        if is_flagged and score >= 0.45:
            return 'Urban'
        
        for cat, (low, high) in CATEGORIES.items():
            if low <= score < high: return cat
        return 'Metro'
    
    results['Category'] = [get_cat(s, f) for s, f in zip(results['RoadScore'], informal_flag)]
    
    # Select only essential columns for output
    output_cols = ['address', 'lat', 'lon', 'RoadScore', 'Category', 'RS_Fusion', 'Sprawl_Factor']
    
    # Add components for transparency
    for r in [500, 1000, 2000]:
        for component in ['U', 'H', 'C', 'Q']:
            col = f'{component}_{r}'
            if col in results.columns:
                output_cols.append(col)
    
    # Save only selected columns
    results[output_cols].to_csv(args.output, index=False)
    
    print(f"DONE! Saved to {args.output}")
    print("\n   Sample Results:")
    # Display Quality scores if available
    display_cols = ['address', 'RoadScore', 'Category']
    if 'Q_500' in results.columns:
        display_cols.extend(['Q_500', 'Q_1000', 'Q_2000'])
    available_cols = [c for c in display_cols if c in results.columns]
    print(results[available_cols].head())

if __name__ == "__main__":
    main()
