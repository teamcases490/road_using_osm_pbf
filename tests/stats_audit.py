import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyze_results(csv_path):
    print("="*60)
    print("ROAD NETWORK SCORE: DESCRIPTIVE STATISTICS & AUDIT")
    print("="*60)
    
    df = pd.read_csv(csv_path)
    
    # 1. Basic Stats
    stats = df['RoadScore'].describe()
    print("\n[1] Overall RoadScore Statistics:")
    print(stats)
    
    # 2. Category Distribution
    print("\n[2] Category Distribution:")
    counts = df['Category'].value_counts()
    percents = df['Category'].value_counts(normalize=True) * 100
    for cat in ['Metro', 'Urban', 'Rural']:
        if cat in counts:
            print(f"  {cat:6}: {counts[cat]:4} ({percents[cat]:5.1f}%)")
            
    # 3. Component Contribution (Mean values)
    comp_cols = [c for c in df.columns if c.startswith(('U_', 'H_', 'C_', 'Q_'))]
    if comp_cols:
        print("\n[3] Component Means (500m / 1000m / 2000m):")
        comp_means = df[comp_cols].mean()
        for comp in ['U', 'H', 'C', 'Q']:
            vals = [f"{comp_means[f'{comp}_{r}']:.3f}" for r in [500, 1000, 2000]]
            print(f"  {comp:2}: {' / '.join(vals)}")

    # 4. Correlation Analysis (Sanity Check)
    # We expect RoadScore to be highly correlated with U (Urbanicity)
    print("\n[4] Correlation Matrix (Target: RoadScore):")
    core_metrics = ['U_500', 'H_500', 'C_500', 'Q_500', 'RoadScore']
    available_metrics = [m for m in core_metrics if m in df.columns]
    corr = df[available_metrics].corr()['RoadScore']
    print(corr)

    # 5. Outlier/Anomaly Detection
    print("\n[5] Anomaly Detection:")
    # High score but Rural category? (Logic check)
    anomalies = df[(df['RoadScore'] > 0.6) & (df['Category'] != 'Metro')]
    if not anomalies.empty:
        print(f"  ⚠️ ALERT: Found {len(anomalies)} inconsistent Category labels!")
    else:
        print("  ✅ CATEGORY LABELS: Consistent with thresholds.")
        
    # Check for 0.0 scores
    zeros = (df['RoadScore'] == 0).sum()
    print(f"  Zero Scores: {zeros} ({zeros/len(df)*100:.1f}%)")
    
    # Check for perfect 1.0 scores
    perfects = (df['RoadScore'] >= 0.99).sum()
    print(f"  Max Scores (>=0.99): {perfects}")

    print("\n" + "="*60)
    print("AUDIT COMPLETE")
    print("="*60)

if __name__ == "__main__":
    analyze_results("data/output/road_scores.csv")
