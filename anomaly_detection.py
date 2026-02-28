# anomaly_detection.py - Anomaly Detection Module
# Univariate (Z-Score, IQR), Multivariate (CBLOF, Isolation Forest)
# pip install pyod

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from pyod.models.cblof import CBLOF
from math import pi

def run_anomaly_detection(df, df_scaled_v2, kmeans_r2, kmeans_r2_labels, cluster_names, attrition_col):
    """Full anomaly detection pipeline: univariate + multivariate methods."""
    
    print("\n" + "="*60)
    print("ANOMALY DETECTION MODULE")
    print("="*60)
    
    # ============================================================
    # PART A — UNIVARIATE ANOMALY DETECTION
    # ============================================================
    
    target_cols = ['MonthlyIncome', 'DistanceFromHome', 'TotalWorkingYears',
                   'YearsAtCompany', 'YearsSinceLastPromotion']
    
    # --- A.1: Z-Score Method ---
    print("\n--- A.1: Z-Score Anomaly Detection (|z| > 3) ---")
    zscore_results = {}
    for col in target_cols:
        z_scores = np.abs(stats.zscore(df[col]))
        anomaly_mask = z_scores > 3
        anomaly_count = anomaly_mask.sum()
        anomaly_vals = df[col][anomaly_mask].values[:5]  # first 5 examples
        zscore_results[col] = {'count': anomaly_count, 'mask': anomaly_mask,
                               'examples': anomaly_vals}
        print(f"  {col:30s} | Anomalies: {anomaly_count:3d} | Examples: {anomaly_vals}")
    
    total_zscore = sum(v['count'] for v in zscore_results.values())
    print(f"\n  Total Z-Score anomalies across all columns: {total_zscore}")
    
    # --- A.2: IQR Method ---
    print("\n--- A.2: IQR Anomaly Detection (1.5*IQR rule) ---")
    iqr_results = {}
    for col in target_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        anomaly_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        anomaly_count = anomaly_mask.sum()
        iqr_results[col] = {'count': anomaly_count, 'mask': anomaly_mask,
                            'lower': lower_bound, 'upper': upper_bound}
        print(f"  {col:30s} | Anomalies: {anomaly_count:3d} | Bounds: [{lower_bound:.1f}, {upper_bound:.1f}]")
    
    # --- A.2b: Z-Score vs IQR Comparison ---
    print("\n--- Z-Score vs IQR Comparison ---")
    print(f"  {'Column':30s} | {'Z-Score':>8s} | {'IQR':>8s} | {'Difference':>10s}")
    print(f"  {'-'*30} | {'-'*8} | {'-'*8} | {'-'*10}")
    for col in target_cols:
        z_cnt = zscore_results[col]['count']
        i_cnt = iqr_results[col]['count']
        print(f"  {col:30s} | {z_cnt:8d} | {i_cnt:8d} | {abs(z_cnt-i_cnt):10d}")
    
    # --- A.3: Plot 1 — Univariate Anomaly Summary (2x3 boxplots) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    for i, col in enumerate(target_cols):
        anomaly_mask = iqr_results[col]['mask']
        normal_vals = df[col][~anomaly_mask]
        anomaly_vals = df[col][anomaly_mask]
        axes[i].boxplot(df[col].values, vert=False, widths=0.6,
                        boxprops=dict(color='#3498db'), medianprops=dict(color='red'))
        axes[i].scatter(normal_vals, np.random.normal(1, 0.04, len(normal_vals)),
                        alpha=0.3, s=10, color='#3498db', label='Normal')
        axes[i].scatter(anomaly_vals, np.random.normal(1, 0.04, len(anomaly_vals)),
                        alpha=0.8, s=40, color='#e74c3c', marker='o', edgecolors='black',
                        label=f'Anomaly (n={len(anomaly_vals)})')
        axes[i].set_title(f'{col} — {iqr_results[col]["count"]} IQR Anomalies', fontsize=11)
        axes[i].set_xlabel(col, fontsize=10)
        axes[i].legend(fontsize=8, loc='upper right')
    axes[5].axis('off')  # hide 6th subplot
    axes[5].text(0.5, 0.5, f'Total Anomalies\nZ-Score: {total_zscore}\nIQR: {sum(v["count"] for v in iqr_results.values())}',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=axes[5].transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange'))
    plt.suptitle('Univariate Anomaly Detection: IQR Method (Red = Anomalous)', fontsize=15)
    plt.tight_layout()
    plt.savefig('plot_anomaly_univariate.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot 1 saved: plot_anomaly_univariate.png")
    
    # ============================================================
    # PART B — MULTIVARIATE ANOMALY DETECTION (CBLOF)
    # ============================================================
    
    # --- B.4: CBLOF using existing KMeans model ---
    # CBLOF = Cluster-Based Local Outlier Factor
    # It uses cluster sizes and distances to identify anomalies
    # Employees far from their assigned cluster center are flagged
    print("\n--- B.4: CBLOF Multivariate Anomaly Detection ---")
    
    # Build CBLOF estimator — uses KMeans internally with same k=4
    # Note: we use alpha=0.75, beta=3 to ensure valid small/large cluster separation
    # CBLOF distinguishes between large and small clusters, then scores employees
    # based on distance to nearest large cluster center
    cblof = CBLOF(contamination=0.05, n_clusters=4, alpha=0.75, beta=3,
                  random_state=42)
    cblof.fit(df_scaled_v2.values)
    
    cblof_labels = cblof.labels_       # 0=normal, 1=anomaly
    cblof_scores = cblof.decision_scores_  # continuous score (higher=more anomalous)
    
    n_cblof_anomalies = cblof_labels.sum()
    print(f"  CBLOF anomalies detected: {n_cblof_anomalies} / {len(df)} ({n_cblof_anomalies/len(df)*100:.1f}%)")
    
    # --- B.5: Add CBLOF results to df_analysis ---
    # Build df_analysis fresh from df so we have all original columns
    df_analysis = df.copy()
    df_analysis['Cluster'] = kmeans_r2_labels
    df_analysis['Cluster_Name'] = df_analysis['Cluster'].map(cluster_names)
    df_analysis['Attrition_Binary'] = np.where(df_analysis['Attrition'] == 'Yes', 1, 0)
    df_analysis['CBLOF_Score'] = cblof_scores
    df_analysis['CBLOF_Anomaly'] = cblof_labels
    print("  Added CBLOF_Score and CBLOF_Anomaly columns to df_analysis")
    
    # --- B.6: Top 15 most anomalous employees ---
    print("\n--- B.6: Top 15 Most Anomalous Employees (CBLOF) ---")
    show_cols = ['Age', 'MonthlyIncome', 'JobRole', 'Department', 'OverTime',
                 'TotalWorkingYears', 'Attrition', 'Cluster_Name', 'CBLOF_Score']
    top15 = df_analysis.nlargest(15, 'CBLOF_Score')[show_cols]
    print(top15.to_string(index=False))
    
    # What % of CBLOF anomalies actually left?
    anom_attrition = df_analysis[df_analysis['CBLOF_Anomaly'] == 1]['Attrition_Binary'].mean() * 100
    norm_attrition = df_analysis[df_analysis['CBLOF_Anomaly'] == 0]['Attrition_Binary'].mean() * 100
    print(f"\n  Attrition rate among CBLOF anomalies: {anom_attrition:.1f}%")
    print(f"  Attrition rate among normal employees: {norm_attrition:.1f}%")
    print(f"  Anomalous employees are {anom_attrition/norm_attrition:.1f}x more likely to leave")
    
    print("\n[OK] Part B complete - CBLOF anomaly detection done")
    
    # --- B.7: Plot 2 — CBLOF Anomalies in PCA Space ---
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(df_scaled_v2)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    
    plt.figure(figsize=(14, 9))
    colors_map = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    # Plot normal employees by cluster
    for i in range(4):
        mask = (kmeans_r2_labels == i) & (cblof_labels == 0)
        plt.scatter(pca_data[mask, 0], pca_data[mask, 1], c=colors_map[i],
                    alpha=0.4, s=15, label=cluster_names[i])
    # Plot anomalies as large red stars
    anom_mask = cblof_labels == 1
    plt.scatter(pca_data[anom_mask, 0], pca_data[anom_mask, 1],
                c='red', marker='*', s=150, edgecolors='black', linewidths=0.5,
                alpha=0.9, label=f'CBLOF Anomaly (n={anom_mask.sum()})', zorder=5)
    plt.title('CBLOF Multivariate Anomaly Detection in PCA Space', fontsize=14)
    plt.xlabel(f'PC1 (Experience & Tenure) ({var1:.1f}% variance)', fontsize=11)
    plt.ylabel(f'PC2 (Engagement) ({var2:.1f}% variance)', fontsize=11)
    plt.legend(fontsize=9, loc='best')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot_anomaly_cblof_pca.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot 2 saved: plot_anomaly_cblof_pca.png")
    
    # ============================================================
    # PART C — ISOLATION FOREST
    # ============================================================
    
    # --- C.8: Isolation Forest ---
    print("\n--- C.8: Isolation Forest Anomaly Detection ---")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.05,
                                  random_state=42)
    iso_preds = iso_forest.fit_predict(df_scaled_v2)  # -1=anomaly, 1=normal
    iso_scores_raw = iso_forest.decision_function(df_scaled_v2)  # lower=more anomalous
    
    # Convert: -1 -> 1 (anomaly), 1 -> 0 (normal)
    iso_labels = np.where(iso_preds == -1, 1, 0)
    # Higher score = more anomalous (negate decision function)
    iso_scores = -iso_scores_raw
    
    df_analysis['IsoForest_Anomaly'] = iso_labels
    df_analysis['IsoForest_Score'] = iso_scores
    
    n_iso_anomalies = iso_labels.sum()
    print(f"  IsolationForest anomalies: {n_iso_anomalies} / {len(df)} ({n_iso_anomalies/len(df)*100:.1f}%)")
    
    # --- C.9: Plot 3 — Isolation Forest Score Distribution ---
    threshold = -iso_forest.offset_  # the threshold in negated score space
    
    fig, ax = plt.subplots(figsize=(12, 6))
    # Normal scores
    normal_scores = iso_scores[iso_labels == 0]
    anomaly_scores = iso_scores[iso_labels == 1]
    ax.hist(normal_scores, bins=50, alpha=0.7, color='#3498db', label='Normal', edgecolor='white')
    ax.hist(anomaly_scores, bins=20, alpha=0.7, color='#e74c3c', label='Anomaly', edgecolor='white')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2)
    ax.text(threshold + 0.005, ax.get_ylim()[1] * 0.9, f'Threshold: {threshold:.3f}',
            color='red', fontsize=10, fontweight='bold')
    ax.annotate(f'Anomalies: {n_iso_anomalies} employees (5%)',
                xy=(0.98, 0.85), xycoords='axes fraction', ha='right',
                fontsize=11, color='#e74c3c', fontweight='bold')
    ax.annotate(f'Normal: {len(df) - n_iso_anomalies} employees (95%)',
                xy=(0.98, 0.75), xycoords='axes fraction', ha='right',
                fontsize=11, color='#3498db', fontweight='bold')
    ax.set_title('Isolation Forest: Anomaly Score Distribution', fontsize=14)
    ax.set_xlabel('Anomaly Score (higher = more anomalous)', fontsize=11)
    ax.set_ylabel('Number of Employees', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_anomaly_isoforest_scores.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot 3 saved: plot_anomaly_isoforest_scores.png")
    
    # ============================================================
    # PART D — ANOMALY COMPARISON ANALYSIS
    # ============================================================
    
    # --- D.10: CBLOF vs IsolationForest Agreement ---
    print("\n--- D.10: CBLOF vs Isolation Forest Comparison ---")
    both = ((cblof_labels == 1) & (iso_labels == 1)).sum()
    cblof_only = ((cblof_labels == 1) & (iso_labels == 0)).sum()
    iso_only = ((cblof_labels == 0) & (iso_labels == 1)).sum()
    neither = ((cblof_labels == 0) & (iso_labels == 0)).sum()
    
    print(f"\n  {'':25s} | {'IsoForest=Anomaly':>18s} | {'IsoForest=Normal':>16s}")
    print(f"  {'-'*25} | {'-'*18} | {'-'*16}")
    print(f"  {'CBLOF=Anomaly':25s} | {both:18d} | {cblof_only:16d}")
    print(f"  {'CBLOF=Normal':25s} | {iso_only:18d} | {neither:16d}")
    print(f"\n  Agreement (both flag): {both} employees")
    print(f"  CBLOF only: {cblof_only}, IsoForest only: {iso_only}")
    print(f"  Agreement rate: {(both + neither) / len(df) * 100:.1f}%")
    
    # --- D.11: Plot 4 — Anomaly Distribution Across Clusters ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: CBLOF anomaly rate per cluster
    company_anom_rate = cblof_labels.mean() * 100
    cluster_anom_rates = df_analysis.groupby('Cluster_Name')['CBLOF_Anomaly'].mean() * 100
    cluster_order = list(cluster_names.values())
    rates_sorted = cluster_anom_rates.reindex(cluster_order)
    bar_colors = ['#e74c3c' if r > company_anom_rate else '#2ecc71' for r in rates_sorted]
    axes[0].bar(range(len(rates_sorted)), rates_sorted.values, color=bar_colors, edgecolor='white')
    axes[0].axhline(company_anom_rate, color='red', linestyle='--', linewidth=1.5)
    axes[0].text(len(rates_sorted)-0.5, company_anom_rate+0.3, f'Avg: {company_anom_rate:.1f}%',
                 color='red', fontsize=9)
    axes[0].set_xticks(range(len(cluster_order)))
    axes[0].set_xticklabels(cluster_order, fontsize=8, rotation=30, ha='right')
    for j, val in enumerate(rates_sorted.values):
        axes[0].text(j, val+0.2, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_title('CBLOF Anomaly Rate by Cluster (%)', fontsize=12)
    axes[0].set_ylabel('Anomaly Rate (%)', fontsize=10)
    
    # Right: Stacked bar — anomaly vs normal per cluster
    cluster_sizes = df_analysis['Cluster_Name'].value_counts().reindex(cluster_order)
    anom_counts = df_analysis[df_analysis['CBLOF_Anomaly']==1].groupby('Cluster_Name').size().reindex(cluster_order).fillna(0)
    norm_counts = cluster_sizes - anom_counts
    anom_pct = (anom_counts / cluster_sizes * 100).values
    norm_pct = (norm_counts / cluster_sizes * 100).values
    axes[1].bar(range(len(cluster_order)), norm_pct, color='#3498db', label='Normal', edgecolor='white')
    axes[1].bar(range(len(cluster_order)), anom_pct, bottom=norm_pct, color='#e74c3c', label='Anomaly', edgecolor='white')
    axes[1].set_xticks(range(len(cluster_order)))
    axes[1].set_xticklabels(cluster_order, fontsize=8, rotation=30, ha='right')
    for j, (ap, np_val) in enumerate(zip(anom_pct, norm_pct)):
        if ap > 2:
            axes[1].text(j, np_val + ap/2, f'{ap:.1f}%', ha='center', va='center', fontsize=9, color='white', fontweight='bold')
    axes[1].set_title('Anomaly vs Normal Composition by Cluster', fontsize=12)
    axes[1].set_ylabel('Percentage (%)', fontsize=10)
    axes[1].legend(fontsize=9)
    
    plt.suptitle('Anomalous Employee Distribution Across Behavioral Segments', fontsize=14)
    plt.tight_layout()
    plt.savefig('plot_anomaly_cluster_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot 4 saved: plot_anomaly_cluster_distribution.png")
    
    # --- D.12: Plot 5 — Anomalous vs Normal Employee Radar Chart ---
    radar_cols = ['Age', 'MonthlyIncome', 'JobSatisfaction',
                  'TotalWorkingYears', 'WorkLifeBalance']
    radar_labels = ['Age', 'Income', 'Job Satisfaction', 'Experience', 'Work-Life']
    
    # Add OverTime binary for radar
    df_analysis['OverTime_Binary'] = np.where(df_analysis['OverTime'] == 'Yes', 1, 0)
    radar_cols_full = radar_cols + ['OverTime_Binary']
    radar_labels_full = radar_labels + ['OverTime']
    
    anom_profile = df_analysis[df_analysis['CBLOF_Anomaly'] == 1][radar_cols_full].mean()
    norm_profile = df_analysis[df_analysis['CBLOF_Anomaly'] == 0][radar_cols_full].mean()
    
    # Normalize 0-1 using min-max across both profiles
    combined = pd.DataFrame([anom_profile, norm_profile])
    col_min = combined.min()
    col_max = combined.max()
    anom_norm = ((anom_profile - col_min) / (col_max - col_min + 1e-9)).values
    norm_norm = ((norm_profile - col_min) / (col_max - col_min + 1e-9)).values
    
    N = len(radar_labels_full)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    anom_vals = list(anom_norm) + [anom_norm[0]]
    norm_vals = list(norm_norm) + [norm_norm[0]]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, anom_vals, 'o-', linewidth=2, color='#e74c3c', label='Anomalous', markersize=5)
    ax.fill(angles, anom_vals, alpha=0.15, color='#e74c3c')
    ax.plot(angles, norm_vals, 'o-', linewidth=2, color='#3498db', label='Normal', markersize=5)
    ax.fill(angles, norm_vals, alpha=0.15, color='#3498db')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels_full, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title('Anomalous vs Normal Employee Behavioral Profile', fontsize=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig('plot_anomaly_profile_radar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot 5 saved: plot_anomaly_profile_radar.png")
    
    # --- D.13: 8 Key Observations ---
    print("\n" + "="*60)
    print("KEY OBSERVATIONS FROM ANOMALY DETECTION")
    print("="*60)
    
    highest_anom_cluster = cluster_anom_rates.idxmax()
    highest_anom_rate = cluster_anom_rates.max()
    lowest_anom_cluster = cluster_anom_rates.idxmin()
    
    print(f"""
1. CBLOF detected {n_cblof_anomalies} anomalous employees ({n_cblof_anomalies/len(df)*100:.1f}% of workforce)
2. Isolation Forest detected {n_iso_anomalies} anomalous employees ({n_iso_anomalies/len(df)*100:.1f}%)
3. Both methods agree on {both} employees — these are the most confidently anomalous
4. {highest_anom_cluster} has the highest anomaly concentration ({highest_anom_rate:.1f}%)
5. {lowest_anom_cluster} has the lowest anomaly rate — most uniform cluster
6. Anomalous employees have {anom_attrition:.1f}% attrition vs {norm_attrition:.1f}% for normal
   => Anomalies are {anom_attrition/norm_attrition:.1f}x more likely to leave the company
7. Z-Score found {total_zscore} univariate anomalies vs IQR found {sum(v['count'] for v in iqr_results.values())}
   => IQR is more sensitive (catches more edge cases in skewed distributions)
8. Multivariate methods (CBLOF, IsoForest) are superior to univariate because they
   detect employees who are unusual across MULTIPLE dimensions simultaneously
""")
    
    # --- D.14: HR Recommendations for Anomalous Employees ---
    print("="*60)
    print("=== ANOMALY EMPLOYEE HR RECOMMENDATIONS ===")
    print("="*60)
    
    print("""
Recommendation 1: Individual Case Review
  Anomalous employees do not fit neatly into any behavioral segment. They are
  statistical outliers — unusually high income for their age, extreme overtime,
  or atypical satisfaction scores. Each should receive an individual review from
  their direct manager and HR business partner to understand their unique situation.
  One-size-fits-all retention programs will NOT work for these employees.

Recommendation 2: Flight Risk Monitoring Dashboard
  Since anomalous employees leave at {:.1f}x the normal rate, they should be
  flagged in the HRIS system with a 'watch list' status. Monthly check-ins,
  skip-level meetings, and pulse surveys can detect disengagement early.
  The cost of losing one senior anomalous employee (avg replacement = 6-9 months
  salary) far exceeds the cost of proactive monitoring.

Recommendation 3: Anomaly-Aware Compensation Bands
  Many anomalies are employees whose compensation is misaligned with their
  profile (e.g., high experience but low pay, or low experience but high pay).
  HR should audit compensation bands for all flagged employees and correct any
  inequities. Pay equity issues are both a legal risk and a retention risk.
""".format(anom_attrition / norm_attrition))
    
    print("[OK] Anomaly detection module complete")
    
    return df_analysis
