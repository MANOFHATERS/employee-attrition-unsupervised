# main.py - Entry point for Employee Segmentation Project
# This script calls all modules in order to run the full analysis

import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# setting up the plots to look nice
plt.style.use('ggplot')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)

from load_data import load_and_explore
from eda import run_eda
from preprocessing import clean_and_preprocess
from clustering import run_clustering
from cluster_analysis import deep_cluster_eda
from recommendations import print_recommendations
from anomaly_detection import run_anomaly_detection
from hierarchical_clustering import run_hierarchical_clustering
from statistical_validation import run_statistical_validation

if __name__ == '__main__':
    
    # Section 1 - Load and explore the data
    df = load_and_explore()
    
    # Section 2 - Exploratory Data Analysis
    run_eda(df)
    
    # Sections 3, 4, 5 - Clean, encode, and scale
    df_clean, df_encoded, df_model, df_scaled, attrition_labels, attrition_col = clean_and_preprocess(df)
    
    # Sections 6, 7, 8, 9 - KMeans clustering and PCA (both rounds)
    kmeans_r2_labels, cluster_names, best_k_r2, df_scaled_v2, kmeans_r2 = run_clustering(df_scaled, df)
    
    # Section 10 - Deep EDA on final clusters
    df_analysis = deep_cluster_eda(df, kmeans_r2_labels, cluster_names)
    
    # Sections 11, 12 - HR Recommendations and Limitations
    print_recommendations()
    
    # Section 13 - Anomaly Detection (Univariate + Multivariate)
    df_analysis = run_anomaly_detection(df, df_scaled_v2, kmeans_r2,
                                         kmeans_r2_labels, cluster_names, attrition_col)
    
    # Section 14 - Hierarchical Clustering Comparison
    agglo_labels, sil_kmeans, sil_agglo, ari = run_hierarchical_clustering(
        df_scaled_v2, kmeans_r2_labels, cluster_names)
    
    # Section 15 - Statistical Validation
    kw_df, effect_df, bootstrap_sils = run_statistical_validation(
        df_analysis, cluster_names, df_scaled_v2, kmeans_r2_labels)
    
    print("\n" + "="*60)
    print("PROJECT COMPLETE! All 15 sections finished.")
    print("  - 13 original visualizations")
    print("  - 5 anomaly detection plots")
    print("  - 2 hierarchical clustering plots")
    print("  - 2 statistical validation plots")
    print("  - Total: 22 visualizations")
    print("="*60)

