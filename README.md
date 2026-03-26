# Road Networks Classification Pipeline

This project provides a robust pipeline for extracting road network features from OpenStreetMap (OSM) and classifying locations into **Rural**, **Urban**, or **Metro** categories based on advanced connectivity and hierarchy metrics.

## Project Structure

```
Road_Networks/
├── scripts/
│   ├── extract_features.py   # Fetches OSM data and computes raw/preprocessed features
│   └── calculate_scores.py   # Calculates final road scores and categories
├── data/
│   ├── input/                # Place your input CSVs and rasters here
│   │   └── merged_gsw_compressed.tif  # Water raster for correction
│   ├── output/               # All generated results will be saved here
│   └── cache/                # OSMnx cache for faster re-runs
├── run_pipeline.py           # Orchestration script to run the full pipeline
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Then runs:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place your input CSV (containing `address`, `lat`, `lon`) in `data/input/`.
   - Ensure `data/input/merged_gsw_compressed.tif` exists for water correction.

## How to Run

### 1. Basic (Sequential) Run
The simplest way to run the pipeline, processing one location at a time:
```bash
python run_pipeline.py --input data/input/locations.csv
```

### 2. Fast (Parallel) Run
To process multiple locations simultaneously and speed up execution, use the `-p` (or `--parallel`) flag. You can optionally specify the number of workers with `-w`:
```bash
python run_pipeline.py --input data/input/locations.csv -p -w 4
```
*Note: If you use `-p` without specifying `-w`, it automatically defaults to `CPU_COUNT - 1`.*

### 3. Resuming an Interrupted Run
If your script gets stopped halfway through, a `processing_checkpoint.pkl` file keeps your progress safe. To pick up exactly where you left off, simply run your original command but add the `-r` (or `--resume`) flag at the end:
```bash
# Example: Resuming a basic run
python run_pipeline.py --input data/input/locations.csv -r

# Example: Resuming a parallel run
python run_pipeline.py --input data/input/locations.csv -p -w 4 -r
```

### 4. Step-by-Step (Manual)

If you only want to extract features *without* calculating the final scores:
```bash
python scripts/extract_features.py --input data/input/locations.csv --output data/output/raw_features.csv
```

If you only want to calculate scores from previously extracted features:
```bash
python scripts/calculate_scores.py --input data/output/preprocessed_features.csv --output data/output/road_scores.csv
```

## Key Components

- **Urbanicity**: Physical road density and grid complexity.
- **Hierarchy**: Distribution of major vs. local roads.
- **Capacity**: Network distribution ability and lane density.
- **Quality Gatekeeper**: Prevents informal settlements/slums from being misclassified as Metro.

## Outputs

- `raw_features.csv`: Original OSM attributes.
- `preprocessed_features.csv`: Cleaned data with engineered metrics.
- `road_scores.csv`: Final results with `RoadScore` and `Category`.
