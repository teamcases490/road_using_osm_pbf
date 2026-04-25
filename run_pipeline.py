import subprocess
import argparse
import sys
import os
import json
import pandas as pd
from pathlib import Path

def run_pipeline(input_csv, parallel=False, workers=None, resume=False, limit=None):
    """Orchestrate the road network classification pipeline."""
    
    # Define paths
    output_dir = Path("data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    raw_output = output_dir / "raw_features.csv"
    preprocessed_output = output_dir / "preprocessed_features.csv"
    final_csv = output_dir / "road_scores.csv"
    final_jsonl = output_dir / "road_scores.jsonl"
    final_json = output_dir / "road_scores.json"

    # Step 1: Feature Extraction
    print("\n" + "="*50)
    print("STEP 1: FEATURE EXTRACTION & PREPROCESSING")
    print("="*50)
    
    extract_cmd = [
        sys.executable, "scripts/extract_features.py",
        "--input", input_csv,
        "--output", str(raw_output),
        "--preprocessed-output", str(preprocessed_output)
    ]
    
    if limit:
        extract_cmd.extend(["--limit", str(limit)])
    if parallel:
        extract_cmd.append("--parallel")
        if workers:
            extract_cmd.extend(["--workers", str(workers)])
    if resume:
        extract_cmd.append("--resume")
    
    try:
        subprocess.check_call(extract_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error during feature extraction: {e}")
        return

    # Step 2: Score Calculation
    print("\n" + "="*50)
    print("STEP 2: SCORE CALCULATION & CATEGORIZATION")
    print("="*50)
    
    score_cmd = [
        sys.executable, "scripts/calculate_scores.py",
        "--input", str(preprocessed_output),
        "--output", str(final_csv)
    ]
    
    try:
        subprocess.check_call(score_cmd)
        
        # Step 3: Multi-format Output Generation (JSON/JSONL)
        print("\nGenerating professional report formats...")
        df = pd.read_csv(final_csv)
        
        # Save JSONL
        df.to_json(final_jsonl, orient="records", lines=True)
        
        # Save Pretty JSON
        with open(final_json, "w", encoding="utf-8") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
            
    except subprocess.CalledProcessError as e:
        print(f"Error during score calculation: {e}")
        return

    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print(f"  CSV   -> {final_csv}")
    print(f"  JSONL -> {final_jsonl}")
    print(f"  JSON  -> {final_json}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Network Classification Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV with lat/lon")
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from previous CSV progress")
    parser.add_argument("--limit", "-l", type=int, help="Limit points")
    args = parser.parse_args()

    run_pipeline(args.input, args.parallel, args.workers, args.resume, args.limit)
