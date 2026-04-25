import pandas as pd
import subprocess
import os
import sys
from pathlib import Path

# Configuration
TEST_DIR = Path("tests/data")
TEST_INPUT = TEST_DIR / "stress_input.csv"
TEST_OUTPUT = TEST_DIR / "stress_scores.csv"
PIPELINE_SCRIPT = "run_pipeline.py"

def setup_test_data():
    """Create a variety of edge-case inputs."""
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    
    data = [
        # 1. Normal location (Mumbai - for baseline)
        {"location": "Mumbai Normal", "lat": 19.0760, "lon": 72.8777},
        
        # 2. Ocean location (Should return 0s/Rural)
        {"location": "Arabian Sea", "lat": 18.5, "lon": 71.0},
        
        # 3. Invalid Coordinates (Extreme)
        {"location": "Invalid North", "lat": 95.0, "lon": 72.0},
        {"location": "Invalid East", "lat": 19.0, "lon": 200.0},
        
        # 4. Missing / NaN Data
        {"location": "Missing Lat", "lat": None, "lon": 72.0},
        {"location": "Missing Lon", "lat": 19.0, "lon": None},
        
        # 5. Weird Address Characters
        {"location": "Addr_!@#$%^&*()_+{}|:\"<>?~`-=[]\\;',./", "lat": 19.1, "lon": 72.9},
        
        # 6. Noisy / Extra Columns
        {"location": "Extra Junk", "lat": 19.2, "lon": 72.8, "junk_col": "rubbish", "more_junk": 12345},
        
        # 7. Extremely Close points (Check if cache or precision issues)
        {"location": "Close 1", "lat": 19.000001, "lon": 72.000001},
        {"location": "Close 2", "lat": 19.000002, "lon": 72.000002},
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(TEST_INPUT, index=False)
    print(f"DONE: Created stress test input with {len(df)} cases.")

def run_stress_test():
    """Execute the pipeline and check for crashes."""
    print("\nStarting Stress Test...")
    
    cmd = [
        sys.executable, PIPELINE_SCRIPT,
        "-i", str(TEST_INPUT),
        "-p", # Parallel
        "-w", "4" # 4 workers
    ]
    
    try:
        # Run and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("PASS: Pipeline finished without crashing.")
        return True
    except subprocess.CalledProcessError as e:
        print("FAIL: PIPELINE CRASHED!")
        print("--- STDERR ---")
        print(e.stderr)
        print("--- STDOUT ---")
        print(e.stdout)
        return False

def validate_outputs():
    """Check if the output CSV makes sense."""
    if not os.path.exists("data/output/road_scores.csv"):
        print("FAIL: Output file not found!")
        return False
        
    df = pd.read_csv("data/output/road_scores.csv")
    print(f"\nValidation Results ({len(df)} rows):")
    
    # Check if we have all rows
    print(df[['address', 'RoadScore', 'Category']])
    
    # Check for NaNs in final scores
    nan_count = df['RoadScore'].isna().sum()
    if nan_count > 0:
        print(f"WARN: Found {nan_count} NaNs in RoadScore")
    else:
        print("PASS: No NaNs found in RoadScore.")
        
    return True

if __name__ == "__main__":
    setup_test_data()
    if run_stress_test():
        validate_outputs()
    else:
        sys.exit(1)
