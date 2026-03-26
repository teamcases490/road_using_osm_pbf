import subprocess
import argparse
import sys
import os

def run_pipeline(input_csv, parallel=False, workers=None, resume=False, limit=None):
    """Orchestrate the road network classification pipeline."""
    
    # Define paths
    raw_output = "data/output/raw_features.csv"
    preprocessed_output = "data/output/preprocessed_features.csv"
    final_output = "data/output/road_scores.csv"

    # Step 1: Feature Extraction
    print("\n" + "="*50)
    print("STEP 1: FEATURE EXTRACTION & PREPROCESSING")
    print("="*50)
    
    extract_cmd = [
        sys.executable, "scripts/extract_features.py",
        "--input", input_csv,
        "--output", raw_output,
        "--preprocessed-output", preprocessed_output
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
        "--input", preprocessed_output,
        "--output", final_output
    ]
    
    try:
        subprocess.check_call(score_cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error during score calculation: {e}")
        return

    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print(f"Final results saved to: {final_output}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road Network Classification Pipeline")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input CSV with lat/lon")
    parser.add_argument("--parallel", "-p", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", "-w", type=int, help="Number of parallel workers (only used if --parallel is set)")
    parser.add_argument("--resume", "-r", action="store_true", help="Automatically resume from a previous checkpoint if it exists")
    parser.add_argument("--limit", "-l", type=int, help="Limit to process only N locations (useful for testing)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs("data/output", exist_ok=True)

    run_pipeline(args.input, args.parallel, args.workers, args.resume, args.limit)
