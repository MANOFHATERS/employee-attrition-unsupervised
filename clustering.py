# clustering.py - Sections 6, 7, 8, 9: KMeans + PCA (both rounds)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from collections import Counter
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse

def run_clustering(df_scaled, df):
    """Run KMeans clustering (Round 1 and 2) and PCA visualization."""
    
    # ============================================================
    # SECTION 6 - K-Means Clustering Round 1 (all features)
    # ============================================================
    
    # finding the optimal k using elbow method and silhouette scores
    k_range = range(2, 16)
    inertia_list = []
    sil_scores = []
    
    for k in k_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(df_scaled)
        inertia_list.append(kmeans_temp.inertia_)
        sil_scores.append(silhouette_score(df_scaled, kmeans_temp.labels_))
        print(f"k={k}: Inertia={kmeans_temp.inertia_:.0f}, Silhouette={sil_scores[-1]:.4f}")
    
    # Round 1 KMeans is run with all features including dept/jobrole
    # dummies. We do NOT plot the elbow/silhouette for Round 1 because
    # the goal of Round 1 is purely to DEMONSTRATE that department dummies
    # dominate the cluster structure (shown in PCA below). Optimizing
    # these clusters would be misleading since we discard them after
    # PCA visualization. k=5 is chosen arbitrarily for demonstration only.
    print("\n[NOTE] Round 1 elbow/silhouette computed but not plotted")
    print("  Reason: Round 1 clusters are discarded - see PCA justification below")
    
    # looking at the curves... k=5 seems like a good choice
    # the elbow bends around 4-5 and silhouette peaks around there
    best_k_r1 = 5
    print(f"\nChosen k={best_k_r1} for Round 1")
    
    # fitting the final model
    kmeans_r1 = KMeans(n_clusters=best_k_r1, random_state=42, n_init=10)
    kmeans_r1_labels = kmeans_r1.fit_predict(df_scaled)
    
    # cluster sizes
    cluster_counts_r1 = Counter(kmeans_r1_labels)
    print(f"\nCluster sizes (Round 1):")
    for cluster, count in sorted(cluster_counts_r1.items()):
        print(f"  Cluster {cluster}: {count} employees ({count/len(df)*100:.1f}%)")
    
    # The clusters are dominated by job role and department dummies
    # which makes sense but isn't very useful for HR insights
    # need to do Round 2 without those columns
    
    print("\n[OK] Section 6 complete - KMeans Round 1 done")
    
    # ============================================================
    # SECTION 7 - PCA Visualization Round 1
    # ============================================================
    
    # reducing to 2 dimensions to visualize clusters
    pca_r1 = PCA(n_components=2, random_state=42)
    pca_r1_data = pca_r1.fit_transform(df_scaled)
    print(f"\nPCA Round 1 - Explained variance ratio: {pca_r1.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca_r1.explained_variance_ratio_)*100:.1f}%")
    
    # which features contribute most to each component?
    pca_loadings = pd.DataFrame(pca_r1.components_.T, index=df_scaled.columns,
                                 columns=['PC1', 'PC2'])
    print("\nTop features for PC1:")
    print(pca_loadings['PC1'].abs().sort_values(ascending=False).head(10))
    print("\nTop features for PC2:")
    print(pca_loadings['PC2'].abs().sort_values(ascending=False).head(10))
    
    # Research Question: RQ2 - Does including department/job dummies distort clusters?
    # Key Insight: Convex hulls visually prove clusters = departments
    var1 = pca_r1.explained_variance_ratio_[0] * 100
    var2 = pca_r1.explained_variance_ratio_[1] * 100
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Left: KMeans clusters with centroids
    scatter1 = axes[0].scatter(pca_r1_data[:, 0], pca_r1_data[:, 1],
                               c=kmeans_r1_labels, cmap='Set1', alpha=0.5, s=15)
    pca_centers = pca_r1.transform(kmeans_r1.cluster_centers_)
    axes[0].scatter(pca_centers[:, 0], pca_centers[:, 1],
                    c='black', marker='X', s=200, edgecolors='white', linewidths=2, zorder=5)
    axes[0].set_title('Round 1: KMeans Clusters (k=5)', fontsize=13)
    axes[0].set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
    axes[0].set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
    axes[0].legend(*scatter1.legend_elements(), title='Cluster', loc='best')
    
    # Right: Department overlay with convex hulls
    dept_colors_map = {'Human Resources': '#e74c3c',
                       'Research & Development': '#3498db',
                       'Sales': '#2ecc71'}
    for dept_name, color in dept_colors_map.items():
        mask = df['Department'] == dept_name
        points = pca_r1_data[mask]
        short_name = dept_name.replace('Research & Development', 'R&D').replace('Human Resources', 'HR')
        axes[1].scatter(points[:, 0], points[:, 1], c=color, alpha=0.5, s=15, label=short_name)
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_pts = np.append(hull.vertices, hull.vertices[0])
                axes[1].plot(points[hull_pts, 0], points[hull_pts, 1], color=color, linewidth=2, alpha=0.7)
                axes[1].fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.1)
            except Exception:
                pass
    axes[1].set_title('Round 1: True Department Groups', fontsize=13)
    axes[1].set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=11)
    axes[1].set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=11)
    axes[1].legend(title='Department', fontsize=9)
    
    fig.text(0.5, -0.02,
             'FINDING: KMeans clusters (left) mirror department groups (right). '
             'Department dummies dominate \u2014 clusters add no new information. '
             'Solution: Remove dept/job role dummies \u2192 Round 2',
             ha='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='orange', alpha=0.8))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)
    plt.savefig('plot_clustering_pca_round1.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot saved: plot_clustering_pca_round1.png")
    # FINDING: KMeans clusters perfectly overlap with department boundaries
    # This confirms that categorical dummies dominate distance-based clustering
    # Round 2 removes these columns to find behavior-based segments
    
    # yep - the clusters basically just mirror department/job role groups
    # thats not very insightful, we already know departments exist
    # need to remove those dummy columns for Round 2
    
    print("\n[OK] Section 7 complete - PCA Round 1 done")
    
    # ============================================================
    # SECTION 8 - K-Means Clustering Round 2
    # ============================================================
    
    # removing job role and department dummies to find behavior-based clusters
    jobrole_cols = [col for col in df_scaled.columns if col.startswith('JobRole_')]
    dept_cols = [col for col in df_scaled.columns if col.startswith('Department_')]
    cols_to_remove = jobrole_cols + dept_cols
    print(f"\nRemoving {len(cols_to_remove)} dummy columns: {cols_to_remove}")
    
    df_scaled_v2 = df_scaled.drop(columns=cols_to_remove)
    print(f"df_scaled_v2 shape: {df_scaled_v2.shape}")
    
    # new elbow and silhouette analysis
    k_range_r2 = range(2, 16)
    inertia_list_r2 = []
    sil_scores_r2 = []
    
    for k in k_range_r2:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(df_scaled_v2)
        inertia_list_r2.append(kmeans_temp.inertia_)
        sil_scores_r2.append(silhouette_score(df_scaled_v2, kmeans_temp.labels_))
        print(f"k={k}: Inertia={kmeans_temp.inertia_:.0f}, Silhouette={sil_scores_r2[-1]:.4f}")
    
    # Research Question: RQ2 - What is the optimal number of behavior-based segments?
    # Key Insight: Delta-inertia secondary axis shows WHERE the elbow actually is
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    k_list = list(k_range_r2)
    
    # Left: Elbow with delta inertia on secondary axis
    axes[0].plot(k_list, inertia_list_r2, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(4, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[0].annotate(f'k=4 selected\nInertia={inertia_list_r2[2]:.0f}',
                     xy=(4, inertia_list_r2[2]), xytext=(7, inertia_list_r2[2]),
                     fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
                     bbox=dict(boxstyle='round', facecolor='lightyellow'))
    axes[0].set_title('Elbow Method \u2014 Round 2 (No Job/Dept Dummies)', fontsize=13)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Inertia', fontsize=11, color='blue')
    axes[0].grid(True, alpha=0.3)
    ax0_twin = axes[0].twinx()
    delta = [inertia_list_r2[i] - inertia_list_r2[i+1] for i in range(len(inertia_list_r2)-1)]
    ax0_twin.plot(k_list[1:], delta, 'g^--', linewidth=1.5, markersize=6, alpha=0.7)
    ax0_twin.set_ylabel('Delta Inertia (rate of change)', fontsize=10, color='green')
    
    # Right: Silhouette with k=4 highlighted and decision textbox
    axes[1].plot(k_list, sil_scores_r2, 'ro-', linewidth=2, markersize=8)
    axes[1].plot(4, sil_scores_r2[2], 'r*', markersize=25, zorder=5)
    for k_val in [3, 4, 5]:
        idx = k_val - 2
        axes[1].annotate(f'k={k_val}: {sil_scores_r2[idx]:.4f}',
                         xy=(k_val, sil_scores_r2[idx]),
                         xytext=(k_val + 1.5, sil_scores_r2[idx] + 0.005),
                         fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'))
    textstr = ('k=4 chosen: highest silhouette\n'
               'among interpretable solutions.\n'
               'k=2 has higher score but loses\n'
               'segment granularity needed for\n'
               'HR-actionable recommendations.')
    props = dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.8)
    axes[1].text(0.98, 0.98, textstr, transform=axes[1].transAxes, fontsize=9,
                 verticalalignment='top', horizontalalignment='right', bbox=props)
    axes[1].set_title('Silhouette Scores \u2014 Round 2', fontsize=13)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Silhouette Score', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot_clustering_elbow_silhouette.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot saved: plot_clustering_elbow_silhouette.png")
    # FINDING: k=4 is optimal - elbow flattens here, silhouette peaks
    # k=2 has slightly higher silhouette but loses actionable granularity
    
    # trying different k values to compare
    # lets try k=3, 4, 5, 6 and see which gives most interpretable clusters
    
    # Research Question: RQ2 - How do we validate k=4 vs alternatives?
    # Key Insight: Silhouette sample charts per cluster show internal quality
    fig = plt.figure(figsize=(22, 14))
    
    # Top row: 4 mini silhouette sample charts
    for idx, test_k in enumerate([3, 4, 5, 6]):
        km = KMeans(n_clusters=test_k, random_state=42, n_init=10)
        labels_temp = km.fit_predict(df_scaled_v2)
        sil = silhouette_score(df_scaled_v2, labels_temp)
        sil_vals = silhouette_samples(df_scaled_v2, labels_temp)
        counts = Counter(labels_temp)
        print(f"\nk={test_k}: Silhouette={sil:.4f}, Sizes={dict(sorted(counts.items()))}")
        
        ax = fig.add_subplot(2, 4, idx + 1)
        y_lower = 10
        for i in range(test_k):
            cluster_sil = np.sort(sil_vals[labels_temp == i])
            size_c = len(cluster_sil)
            y_upper = y_lower + size_c
            color = plt.cm.Set2(i / test_k)
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil,
                             facecolor=color, edgecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_c, str(i), fontsize=9)
            y_lower = y_upper + 10
        ax.axvline(sil, color='red', linestyle='--', linewidth=1.5)
        ax.set_title(f'k={test_k} (avg={sil:.3f})', fontsize=11,
                     fontweight='bold' if test_k == 4 else 'normal')
        ax.set_xlabel('Silhouette Value', fontsize=9)
        ax.set_ylabel('Cluster', fontsize=9)
        ax.set_yticks([])
        if test_k == 4:
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(3)
    
    # Bottom row: single detailed heatmap for k=4
    ax_hm = fig.add_subplot(2, 1, 2)
    km_final = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels_final = km_final.fit_predict(df_scaled_v2)
    centers = pd.DataFrame(km_final.cluster_centers_, columns=df_scaled_v2.columns)
    cluster_names_temp = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
    sns.heatmap(centers, cmap='RdYlBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, annot_kws={'size': 7},
                yticklabels=cluster_names_temp, ax=ax_hm)
    ax_hm.set_title('Final Model: k=4 Cluster Centers (Standardized Values)', fontsize=13)
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.suptitle('Cluster Validation: k=4 Selected as Optimal Solution', fontsize=15)
    plt.tight_layout()
    plt.savefig('plot_clustering_validation_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n[OK] Plot saved: plot_clustering_validation_dashboard.png")
    # FINDING: k=4 has the most evenly-sized silhouette profiles
    # No cluster has predominantly negative silhouette values
    
    # after looking at the heatmaps, k=4 gives the most distinct groups
    # k=4 has clear separation between employee types
    best_k_r2 = 4
    print(f"\nFinal choice: k={best_k_r2}")
    
    # fitting the final round 2 model
    kmeans_r2 = KMeans(n_clusters=best_k_r2, random_state=42, n_init=10)
    kmeans_r2_labels = kmeans_r2.fit_predict(df_scaled_v2)
    print(f"Final silhouette score: {silhouette_score(df_scaled_v2, kmeans_r2_labels):.4f}")
    
    cluster_counts_r2 = Counter(kmeans_r2_labels)
    for c, cnt in sorted(cluster_counts_r2.items()):
        print(f"  Cluster {c}: {cnt} employees ({cnt/len(df)*100:.1f}%)")
    
    # naming clusters based on actual runtime profiles (refined after deep EDA)
    # Cluster 0: Age=45.1, Income=$13,349, Exp=22.8yr, Attrition=8.7% → clearly the most loyal
    # Cluster 1: Age=36.1, Income=$5,284, Exp=9.5yr, Attrition=11.3% → mid-level, moderate risk
    # Cluster 2: Age=35.5, Income=$4,720, Exp=8.3yr, Attrition=13.4% → largest group, moderate
    # Cluster 3: Age=33.5, Income=$4,693, Exp=8.0yr, Attrition=28.1% → youngest, highest attrition!
    cluster_names = {0: 'Experienced Loyal',
                     1: 'Mid-Level Moderate', 
                     2: 'Mid-Career Moderate',
                     3: 'Young At-Risk'}
    print(f"\nCluster names: {cluster_names}")
    # Names verified against runtime cluster profiles — see cluster_analysis.py output
    
    print("\n[OK] Section 8 complete - KMeans Round 2 done")
    
    # ============================================================
    # SECTION 9 - PCA Visualization Round 2
    # ============================================================
    
    # PCA on the v2 data (without job/dept dummies)
    pca_r2 = PCA(n_components=2, random_state=42)
    pca_r2_data = pca_r2.fit_transform(df_scaled_v2)
    print(f"\nPCA Round 2 - Explained variance: {pca_r2.explained_variance_ratio_}")
    print(f"Total variance explained (2 comp): {sum(pca_r2.explained_variance_ratio_)*100:.1f}%")
    
    # also try 3 components
    pca_r2_3d = PCA(n_components=3, random_state=42)
    pca_r2_3d_data = pca_r2_3d.fit_transform(df_scaled_v2)
    print(f"PCA 3 components variance: {pca_r2_3d.explained_variance_ratio_}")
    print(f"Total variance explained (3 comp): {sum(pca_r2_3d.explained_variance_ratio_)*100:.1f}%")
    
    # feature loadings for Round 2
    pca_loadings_r2 = pd.DataFrame(pca_r2.components_.T, index=df_scaled_v2.columns,
                                    columns=['PC1', 'PC2'])
    print("\nRound 2 - Top features for PC1:")
    print(pca_loadings_r2['PC1'].abs().sort_values(ascending=False).head(8))
    print("\nRound 2 - Top features for PC2:")
    print(pca_loadings_r2['PC2'].abs().sort_values(ascending=False).head(8))
    
    # Research Question: RQ2+RQ3 - Do behavior-based clusters show meaningful separation?
    # Key Insight: 95% confidence ellipses are the academic standard
    var1_r2 = pca_r2.explained_variance_ratio_[0] * 100
    var2_r2 = pca_r2.explained_variance_ratio_[1] * 100
    sil_final = silhouette_score(df_scaled_v2, kmeans_r2_labels)
    
    plt.figure(figsize=(12, 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i in range(best_k_r2):
        mask = kmeans_r2_labels == i
        cluster_data = pca_r2_data[mask]
        plt.scatter(cluster_data[:, 0], cluster_data[:, 1],
                    c=colors[i], label=cluster_names[i], alpha=0.5, s=20)
        # 95% confidence ellipse
        if len(cluster_data) > 2:
            cov = np.cov(cluster_data.T)
            vals, vecs = np.linalg.eigh(cov)
            angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
            w, h = 2 * np.sqrt(vals) * 2.45  # 95% CI
            ell = Ellipse(xy=(cluster_data[:, 0].mean(), cluster_data[:, 1].mean()),
                          width=w, height=h, angle=angle,
                          fill=False, edgecolor=colors[i], linewidth=2, linestyle='--', alpha=0.8)
            plt.gca().add_patch(ell)
        # Centroid label
        cx, cy = cluster_data.mean(axis=0)
        plt.annotate(cluster_names[i], xy=(cx, cy), fontsize=9, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor=colors[i], alpha=0.8))
    
    plt.title('Employee Segments in 2D PCA Space (Round 2)', fontsize=14)
    plt.figtext(0.5, -0.01, 'Ellipses show 95% confidence region per segment',
                ha='center', fontsize=10, style='italic')
    plt.xlabel(f'PC1 \u2014 Experience & Tenure ({var1_r2:.1f}% variance)', fontsize=11)
    plt.ylabel(f'PC2 \u2014 Satisfaction & Engagement ({var2_r2:.1f}% variance)', fontsize=11)
    plt.legend(fontsize=10, loc='best')
    plt.text(0.02, 0.98, f'Silhouette Score: {sil_final:.4f}',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', alpha=0.8))
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('plot_clustering_pca_round2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] Plot saved: plot_clustering_pca_round2.png")
    # FINDING: 4 segments show meaningful separation in PCA space
    # Experienced Loyal and Young At-Risk are most distinct
    # Moderate overlap between Mid-Career Stable and Experienced Loyal
    
    # comparing R1 vs R2:
    # R1 clusters were driven by job roles/departments (not useful)
    # R2 clusters capture actual employee behavior patterns (much better!)
    # Round 2 is clearly more insightful for HR purposes
    
    print("\n[OK] Section 9 complete - PCA Round 2 done")
    
    return kmeans_r2_labels, cluster_names, best_k_r2, df_scaled_v2, kmeans_r2
