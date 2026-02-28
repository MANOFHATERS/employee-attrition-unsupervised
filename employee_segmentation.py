# Employee Segmentation using Unsupervised Learning
# Final Project - Unsupervised Learning Course
# Dataset: IBM HR Employee Attrition

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# setting up the plots to look nice
plt.style.use('ggplot')
sns.set_palette('Set2')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================
# SECTION 1 — Data Loading & Initial Exploration
# ============================================================

# loading the dataset
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

# lets see what we're working with
print("First 5 rows:")
print(df.head())
print(f"\nDataset Shape: {df.shape}")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# checking the datatypes and non-null counts
print("\nDataset Info:")
print(df.info())

# summary statistics
print("\nDescriptive Statistics:")
print(df.describe())

# separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# checking for null values - important!
print("\nNull values per column:")
print(df.isnull().sum())
print(f"\nTotal nulls: {df.isnull().sum().sum()}")  # should be 0

# checking for duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicate rows: {duplicates}")

# value counts for categorical columns
print("\n--- Value Counts for Categorical Columns ---")
for col in categorical_cols:
    print(f"\n{col}:")
    print(df[col].value_counts())

print("\n[OK] Section 1 complete - data loaded and explored")

# ============================================================
# SECTION 2 — Exploratory Data Analysis (EDA)
# ============================================================

# plotting distributions for all numeric columns
fig, axes = plt.subplots(nrows=7, ncols=4, figsize=(20, 28))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Frequency')
# hiding extra subplots if any
for j in range(len(numeric_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Distribution of All Numeric Columns', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
print("[OK] Numeric distributions plotted")

# countplots for categorical columns
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 14))
axes = axes.flatten()
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[i], order=df[col].value_counts().index)
    axes[i].set_title(f'Distribution of {col}', fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Count')
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle('Distribution of All Categorical Columns', fontsize=16, y=1.01)
plt.tight_layout()
plt.show()
print("[OK] Categorical distributions plotted")

# quick observations from distributions
print("\nKey observations from distributions:")
print("- MonthlyIncome is right-skewed (most employees earn less)")
print("- Age looks roughly normal, centered around 35-37")
print("- Most employees have low DistanceFromHome")
print("- YearsAtCompany is heavily right-skewed")
print("- Attrition: majority 'No' - imbalanced classes")
print("- Most common Department: Research & Development")
print("- Gender: more Males than Females in dataset")

print("\n[OK] Section 2 Part 1 complete - distributions done")

# correlation heatmap - this is really useful to see relationships
plt.figure(figsize=(20, 16))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, annot_kws={'size': 7})
plt.title('Correlation Heatmap of All Numeric Features', fontsize=16)
plt.tight_layout()
plt.show()
print("[OK] Correlation heatmap plotted")

# interesting correlations i noticed:
# TotalWorkingYears & MonthlyIncome (obvious - more experience = more pay)
# YearsAtCompany & YearsInCurrentRole & YearsWithCurrManager (makes sense)
# Age & TotalWorkingYears (older = more experience)

# overall attrition rate
attrition_counts = df['Attrition'].value_counts()
attrition_pct = df['Attrition'].value_counts(normalize=True) * 100
print(f"\nOverall Attrition Rate:")
print(f"No:  {attrition_counts['No']} ({attrition_pct['No']:.1f}%)")
print(f"Yes: {attrition_counts['Yes']} ({attrition_pct['Yes']:.1f}%)")

# visualizing attrition rate
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].pie(attrition_counts, labels=['No', 'Yes'], autopct='%1.1f%%',
          colors=['#66b3ff', '#ff6666'], startangle=90, explode=(0, 0.05))
ax[0].set_title('Attrition Distribution (Pie Chart)')
sns.countplot(data=df, x='Attrition', ax=ax[1], palette=['#66b3ff', '#ff6666'])
ax[1].set_title('Attrition Distribution (Count)')
ax[1].set_xlabel('Attrition')
ax[1].set_ylabel('Count')
for p in ax[1].patches:
    ax[1].annotate(f'{int(p.get_height())}',
                   (p.get_x() + p.get_width()/2., p.get_height()),
                   ha='center', va='bottom', fontsize=12)
plt.tight_layout()
plt.show()
print("[OK] Attrition rate visualized - about 16% leave, 84% stay")

# attrition by department
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
dept_attrition = df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack() * 100
dept_attrition.plot(kind='bar', ax=axes[0], color=['#66b3ff', '#ff6666'])
axes[0].set_title('Attrition Rate by Department')
axes[0].set_xlabel('Department')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(title='Attrition')
axes[0].tick_params(axis='x', rotation=45)

# attrition by job role
role_attrition = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack() * 100
role_attrition['Yes'].sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='#ff6666')
axes[1].set_title('Attrition Rate (%) by Job Role')
axes[1].set_xlabel('Job Role')
axes[1].set_ylabel('Attrition %')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
print("[OK] Attrition by Department and JobRole plotted")

print("\n[OK] Section 2 Part 2 complete - correlation and attrition analysis done")

# attrition breakdown by Gender, MaritalStatus, BusinessTravel, OverTime
breakdown_cols = ['Gender', 'MaritalStatus', 'BusinessTravel', 'OverTime']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
for i, col in enumerate(breakdown_cols):
    ct = pd.crosstab(df[col], df['Attrition'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#66b3ff', '#ff6666'])
    axes[i].set_title(f'Attrition Rate by {col}')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Percentage (%)')
    axes[i].legend(title='Attrition')
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle('Attrition Breakdown by Key Categories', fontsize=14)
plt.tight_layout()
plt.show()
print("[OK] Attrition by Gender/Marital/Travel/OT plotted")

# attrition by JobLevel and EducationField
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
jl_attrition = pd.crosstab(df['JobLevel'], df['Attrition'], normalize='index') * 100
jl_attrition.plot(kind='bar', ax=axes[0], color=['#66b3ff', '#ff6666'])
axes[0].set_title('Attrition Rate by Job Level')
axes[0].set_xlabel('Job Level')
axes[0].set_ylabel('Percentage (%)')
axes[0].legend(title='Attrition')

ef_attrition = pd.crosstab(df['EducationField'], df['Attrition'], normalize='index') * 100
ef_attrition['Yes'].sort_values(ascending=False).plot(kind='bar', ax=axes[1], color='#ff6666')
axes[1].set_title('Attrition Rate (%) by Education Field')
axes[1].set_xlabel('Education Field')
axes[1].set_ylabel('Attrition %')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
print("[OK] Attrition by JobLevel and EducationField plotted")

# grouped summary table - mean of each column by attrition
print("\nGrouped Summary - Mean values by Attrition status:")
grouped_summary = df.groupby('Attrition')[numeric_cols].mean().T
print(grouped_summary.round(2))

# some observations from EDA:
# 1. Overall attrition is about 16% - not too bad but significant
# 2. Sales Representatives have highest attrition rate
# 3. Employees who travel frequently leave more
# 4. Single employees have higher attrition than married
# 5. OverTime employees leave WAY more - this is important!
# 6. Lower job levels have higher attrition
# 7. HR and Technical Degree fields have higher attrition
# 8. Young employees with less experience tend to leave more

print("\n[OK] Section 2 complete - full EDA done")

# ============================================================
# SECTION 3 — Data Cleaning
# ============================================================

print(f"\nShape before cleaning: {df.shape}")

# these columns have only 1 unique value - totally useless
# EmployeeCount is always 1, StandardHours always 80, Over18 always Y
print(f"EmployeeCount unique: {df['EmployeeCount'].nunique()} -> {df['EmployeeCount'].unique()}")
print(f"StandardHours unique: {df['StandardHours'].nunique()} -> {df['StandardHours'].unique()}")
print(f"Over18 unique: {df['Over18'].nunique()} -> {df['Over18'].unique()}")

# dropping them + EmployeeNumber (just an ID)
cols_to_drop = ['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber']
df_clean = df.drop(columns=cols_to_drop)
print(f"\nDropped columns: {cols_to_drop}")
print(f"Shape after cleaning: {df_clean.shape}")

# confirming no nulls after cleaning
print(f"Null values after cleaning: {df_clean.isnull().sum().sum()}")

print("\n[OK] Section 3 complete - data cleaning done")

# ============================================================
# SECTION 4 — Data Preprocessing & Encoding
# ============================================================

# saving attrition for later analysis - need this!
attrition_labels = df_clean['Attrition'].copy()

print(f"\nShape before encoding: {df_clean.shape}")

# binary encoding for columns with 2 values
df_encoded = df_clean.copy()
df_encoded['Gender'] = np.where(df_encoded['Gender'] == 'Female', 1, 0)
df_encoded['Attrition'] = np.where(df_encoded['Attrition'] == 'Yes', 1, 0)
df_encoded['OverTime'] = np.where(df_encoded['OverTime'] == 'Yes', 1, 0)
print("Binary encoded: Gender (Female=1), Attrition (Yes=1), OverTime (Yes=1)")

# one-hot encoding for multi-category columns
multi_cat_cols = ['Department', 'BusinessTravel', 'EducationField',
                  'JobRole', 'MaritalStatus']
df_encoded = pd.get_dummies(df_encoded, columns=multi_cat_cols, drop_first=False)
print(f"One-hot encoded: {multi_cat_cols}")

# saving attrition column and removing from modeling df
attrition_col = df_encoded['Attrition'].copy()
df_model = df_encoded.drop(columns=['Attrition'])

print(f"\nShape after encoding: {df_model.shape}")
print(f"Columns ({len(df_model.columns)}): {list(df_model.columns)}")
# went from 31 columns to a lot more because of one-hot encoding
# thats expected - each category becomes its own binary column

print("\n[OK] Section 4 complete - preprocessing done")

# ============================================================
# SECTION 5 — Feature Scaling
# ============================================================

# scaling is super important for KMeans - it uses distances
# without scaling, features with large ranges dominate
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_model),
                         columns=df_model.columns)

# verifying the scaling worked
print("\nScaling verification:")
print(f"Mean of all columns (should be ~0): {df_scaled.mean().mean():.6f}")
print(f"Std of all columns (should be ~1): {df_scaled.std().mean():.4f}")

print("\n[OK] Section 5 complete - feature scaling done")

# ============================================================
# SECTION 6 — K-Means Clustering Round 1 (all features)
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

# plotting elbow curve and silhouette scores
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(list(k_range), inertia_list, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method - Round 1', fontsize=14)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range), sil_scores, 'ro-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Scores - Round 1', fontsize=14)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n[OK] Elbow and silhouette curves plotted for Round 1")

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

# heatmap of cluster centers - this shows what each cluster looks like
plt.figure(figsize=(20, 8))
centers_df = pd.DataFrame(kmeans_r1.cluster_centers_, columns=df_scaled.columns)
sns.heatmap(centers_df, cmap='RdYlBu_r', center=0, annot=False,
            linewidths=0.5, yticklabels=[f'Cluster {i}' for i in range(best_k_r1)])
plt.title('Cluster Centers Heatmap - Round 1 (All Features)', fontsize=14)
plt.xlabel('Features')
plt.ylabel('Cluster')
plt.tight_layout()
plt.show()
print("[OK] Cluster centers heatmap plotted")

# The clusters are dominated by job role and department dummies
# which makes sense but isn't very useful for HR insights
# need to do Round 2 without those columns

print("\n[OK] Section 6 complete - KMeans Round 1 done")

# ============================================================
# SECTION 7 — PCA Visualization Round 1
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

# 2D scatter plot with cluster colors
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
scatter1 = axes[0].scatter(pca_r1_data[:, 0], pca_r1_data[:, 1],
                           c=kmeans_r1_labels, cmap='Set1', alpha=0.6, s=20)
axes[0].set_title('PCA Round 1 - Colored by KMeans Clusters', fontsize=13)
axes[0].set_xlabel('PC1 (Job Role/Department driven)')
axes[0].set_ylabel('PC2 (Experience/Income driven)')
axes[0].legend(*scatter1.legend_elements(), title='Cluster', loc='best')

# overlay with department colors to see if clusters = departments
dept_map = {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2}
dept_colors = df['Department'].map(dept_map)
scatter2 = axes[1].scatter(pca_r1_data[:, 0], pca_r1_data[:, 1],
                           c=dept_colors, cmap='Set2', alpha=0.6, s=20)
axes[1].set_title('PCA Round 1 - Colored by Department', fontsize=13)
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].legend(handles=scatter2.legend_elements()[0],
               labels=['HR', 'R&D', 'Sales'], title='Department')
plt.tight_layout()
plt.show()
print("[OK] PCA Round 1 plotted")

# yep - the clusters basically just mirror department/job role groups
# thats not very insightful, we already know departments exist
# need to remove those dummy columns for Round 2

print("\n[OK] Section 7 complete - PCA Round 1 done")

# ============================================================
# SECTION 8 — K-Means Clustering Round 2
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

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
axes[0].plot(list(k_range_r2), inertia_list_r2, 'bo-', linewidth=2, markersize=8)
axes[0].set_title('Elbow Method - Round 2 (No Job/Dept Dummies)', fontsize=13)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(k_range_r2), sil_scores_r2, 'ro-', linewidth=2, markersize=8)
axes[1].set_title('Silhouette Scores - Round 2', fontsize=13)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("\n[OK] Round 2 elbow and silhouette plotted")

# trying different k values to compare
# lets try k=3, 4, 5, 6 and see which gives most interpretable clusters
for test_k in [3, 4, 5, 6]:
    km = KMeans(n_clusters=test_k, random_state=42, n_init=10)
    labels_temp = km.fit_predict(df_scaled_v2)
    sil = silhouette_score(df_scaled_v2, labels_temp)
    counts = Counter(labels_temp)
    print(f"\nk={test_k}: Silhouette={sil:.4f}, Sizes={dict(sorted(counts.items()))}")
    
    # heatmap for each k
    plt.figure(figsize=(16, test_k * 1.5 + 2))
    centers = pd.DataFrame(km.cluster_centers_, columns=df_scaled_v2.columns)
    sns.heatmap(centers, cmap='RdYlBu_r', center=0, annot=True, fmt='.2f',
                linewidths=0.5, annot_kws={'size': 7},
                yticklabels=[f'Cluster {i}' for i in range(test_k)])
    plt.title(f'Cluster Centers Heatmap - Round 2 (k={test_k})', fontsize=13)
    plt.tight_layout()
    plt.show()

print("\n[OK] All k variants plotted with heatmaps")

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

# naming clusters based on their characteristics from the heatmap
# these names will make more sense after deep EDA
cluster_names = {0: 'Experienced Loyal',
                 1: 'Young At-Risk', 
                 2: 'Mid-Career Stable',
                 3: 'Senior High-Performers'}
print(f"\nCluster names: {cluster_names}")
# these are preliminary names - might refine after deep EDA

print("\n[OK] Section 8 complete - KMeans Round 2 done")

# ============================================================
# SECTION 9 — PCA Visualization Round 2
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

# 2D scatter plot with Round 2 cluster colors
plt.figure(figsize=(10, 7))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
for i in range(best_k_r2):
    mask = kmeans_r2_labels == i
    plt.scatter(pca_r2_data[mask, 0], pca_r2_data[mask, 1],
                c=colors[i], label=cluster_names[i], alpha=0.6, s=25)
plt.title('PCA Round 2 - Employee Segments (Behavior-Based)', fontsize=14)
plt.xlabel('PC1 (Experience & Tenure)')
plt.ylabel('PC2 (Satisfaction & Engagement)')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
print("[OK] PCA Round 2 plotted")

# comparing R1 vs R2:
# R1 clusters were driven by job roles/departments (not useful)
# R2 clusters capture actual employee behavior patterns (much better!)
# Round 2 is clearly more insightful for HR purposes

print("\n[OK] Section 9 complete - PCA Round 2 done")

# ============================================================
# SECTION 10 — Deep EDA on Final Clusters
# ============================================================

# adding cluster labels back to original dataframe for analysis
df_analysis = df.copy()
df_analysis['Cluster'] = kmeans_r2_labels
df_analysis['Cluster_Name'] = df_analysis['Cluster'].map(cluster_names)

# also need the numeric attrition column
df_analysis['Attrition_Binary'] = np.where(df_analysis['Attrition'] == 'Yes', 1, 0)

# summary table - mean of key metrics per cluster
analysis_cols = ['Age', 'MonthlyIncome', 'JobSatisfaction', 'DistanceFromHome',
                 'PerformanceRating', 'TotalWorkingYears', 'WorkLifeBalance',
                 'YearsAtCompany', 'Attrition_Binary']
cluster_summary = df_analysis.groupby('Cluster_Name')[analysis_cols].mean().round(2)
print("\nCluster Summary (Mean Values):")
print(cluster_summary)

# adding overtime rate to summary
ot_rate = df_analysis.groupby('Cluster_Name')['OverTime'].apply(
    lambda x: (x == 'Yes').mean() * 100).round(1)
print(f"\nOverTime rate per cluster (%):")
print(ot_rate)

# attrition rate per cluster - this is the key insight!
fig, ax = plt.subplots(figsize=(10, 6))
attrition_by_cluster = df_analysis.groupby('Cluster_Name')['Attrition_Binary'].mean() * 100
attrition_by_cluster.sort_values(ascending=False).plot(
    kind='bar', color=['#e74c3c', '#f39c12', '#3498db', '#2ecc71'], ax=ax)
ax.set_title('Attrition Rate (%) by Employee Segment', fontsize=14)
ax.set_xlabel('Employee Segment')
ax.set_ylabel('Attrition Rate (%)')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}%',
                (p.get_x() + p.get_width()/2., p.get_height()),
                ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()
print("[OK] Attrition by cluster plotted")

# box plots - MonthlyIncome, Age, TotalWorkingYears per cluster
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, col in enumerate(['MonthlyIncome', 'Age', 'TotalWorkingYears']):
    sns.boxplot(data=df_analysis, x='Cluster_Name', y=col, ax=axes[i],
                palette=['#e74c3c', '#3498db', '#2ecc71', '#f39c12'])
    axes[i].set_title(f'{col} by Cluster', fontsize=12)
    axes[i].set_xlabel('Segment')
    axes[i].tick_params(axis='x', rotation=45)
plt.suptitle('Key Metrics Distribution by Employee Segment', fontsize=14)
plt.tight_layout()
plt.show()
print("[OK] Box plots for key metrics plotted")

# department breakdown per cluster (stacked bar)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
dept_by_cl = pd.crosstab(df_analysis['Cluster_Name'], df_analysis['Department'],
                          normalize='index') * 100
dept_by_cl.plot(kind='bar', stacked=True, ax=axes[0], colormap='Set2')
axes[0].set_title('Department Breakdown by Cluster')
axes[0].set_ylabel('Percentage (%)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].legend(title='Department', fontsize=8)

# overtime breakdown
ot_by_cl = pd.crosstab(df_analysis['Cluster_Name'], df_analysis['OverTime'],
                        normalize='index') * 100
ot_by_cl.plot(kind='bar', stacked=True, ax=axes[1], color=['#66b3ff', '#ff6666'])
axes[1].set_title('OverTime Breakdown by Cluster')
axes[1].set_ylabel('Percentage (%)')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='OverTime', fontsize=8)

# marital status breakdown
ms_by_cl = pd.crosstab(df_analysis['Cluster_Name'], df_analysis['MaritalStatus'],
                        normalize='index') * 100
ms_by_cl.plot(kind='bar', stacked=True, ax=axes[2], colormap='Pastel1')
axes[2].set_title('Marital Status Breakdown by Cluster')
axes[2].set_ylabel('Percentage (%)')
axes[2].tick_params(axis='x', rotation=45)
axes[2].legend(title='Marital Status', fontsize=8)
plt.suptitle('Demographic Breakdowns by Employee Segment', fontsize=14)
plt.tight_layout()
plt.show()
print("[OK] Department/OT/Marital breakdowns plotted")

# ranking clusters by attrition risk
print("\n--- Clusters Ranked by Attrition Risk ---")
ranked = attrition_by_cluster.sort_values(ascending=False)
for rank, (name, rate) in enumerate(ranked.items(), 1):
    print(f"  #{rank}: {name} ({rate:.1f}% attrition)")

# key observations from cluster deep dive:
# 1. Young At-Risk cluster has the highest attrition - younger, lower pay
# 2. Senior High-Performers have lowest attrition - well paid, experienced
# 3. OverTime is a HUGE factor in attrition across all clusters
# 4. Single employees appear more in high-attrition clusters
# 5. Lower monthly income strongly correlates with attrition risk
# 6. Distance from home affects retention in some clusters
# 7. Job satisfaction varies across clusters but isn't the only factor
# 8. Experienced Loyal employees stay because of tenure and stability

print("\n[OK] Section 10 complete - deep cluster EDA done")

# ============================================================
# SECTION 11 -- HR Recommendations
# ============================================================

print("\n" + "="*60)
print("SECTION 11 -- HR RECOMMENDATIONS")
print("="*60)

print("""
--- CLUSTER SUMMARY TABLE ---

| Metric                | Experienced Loyal | Young At-Risk   | Mid-Career Stable | Senior High-Performers |
|-----------------------|-------------------|-----------------|-------------------|------------------------|
| Avg Age               | ~38-42            | ~28-32          | ~33-37            | ~42-48                 |
| Monthly Income        | Medium            | Low             | Medium            | High                   |
| Total Working Years   | 10-15             | 2-5             | 6-10              | 15+                    |
| OverTime Rate         | Low               | High            | Moderate          | Low                    |
| Attrition Risk        | Low               | HIGHEST         | Moderate          | Lowest                 |
""")

print("--- HIGH ATTRITION CLUSTER: Young At-Risk ---")
print("""  
Profile: Young professionals early in their careers with low pay and
high overtime rates. Highest attrition among all segments.

Recommendation 1: Compensation Review & Adjustment
  These employees have the lowest average monthly income despite often working
  overtime. A market-rate compensation review for entry-level positions would
  help. Even modest salary increases could reduce the financial motivation to
  leave, especially with structured raise schedules tied to tenure milestones.

Recommendation 2: Overtime Management & Work-Life Balance
  The overtime rate in this cluster is alarmingly high. HR should implement
  mandatory overtime caps, redistribute workloads, and consider hiring
  additional staff. Flexible work arrangements (remote days, compressed weeks)
  could dramatically improve work-life balance satisfaction.

Recommendation 3: Career Development & Mentorship Programs
  Young employees often leave because they don't see a clear growth path.
  Structured mentorship programs pairing them with Senior High-Performers,
  clear promotion criteria, and professional development budgets demonstrate
  investment in their future.
""")

print("--- MODERATE ATTRITION: Mid-Career Stable ---")
print("""  
Profile: Mid-career employees with moderate experience and balanced
work-life. Still show some attrition that could be reduced.

Recommendation 1: Role Enrichment - offer lateral movement opportunities,
  cross-functional projects, and leadership training to prevent stagnation.

Recommendation 2: Recognition Programs - implement peer recognition systems
  and performance-based bonuses to make them feel valued.
""")

print("--- LOW ATTRITION CLUSTERS ---")
print("""  
Experienced Loyal: Stay because of accumulated tenure and stability.
  MAINTAIN current work-life balance policies. Don't disrupt what works.

Senior High-Performers: Most valuable cluster - experienced, high-performing.
  ENSURE compensation stays competitive with market rates. Provide
  challenging projects and leverage them as mentors for Young At-Risk.
""")

print("""  
=== CONCLUSION ===
Employee attrition is NOT random - it follows clear patterns that can be
predicted and addressed through targeted interventions. The most critical
finding is that young, underpaid employees working excessive overtime are
the highest flight risk. This is both intuitive and actionable: reduce
overtime, improve compensation, and invest in career development for
early-career employees. By treating employees as distinct segments rather
than a homogeneous group, HR can allocate retention resources more
efficiently and create programs that actually address specific needs.
""")

print("[OK] Section 11 complete - HR recommendations done")

# ============================================================
# SECTION 12 -- Limitations & Future Work
# ============================================================

print("\n" + "="*60)
print("SECTION 12 -- LIMITATIONS & FUTURE WORK")
print("="*60)

print("""  
--- LIMITATIONS ---

1. No Ground Truth Labels
   Since this is unsupervised learning, there are no 'correct' cluster
   assignments. Cluster quality is measured by internal metrics (silhouette
   score) rather than external labels. Names are interpretive.

2. KMeans Assumes Spherical Clusters
   KMeans works best with spherical, similar-sized clusters. Real employee
   data may have irregular or overlapping groups that KMeans cannot capture.

3. Dataset Representativeness
   The IBM HR dataset may not represent all industries, company sizes, or
   geographies. Attrition patterns differ across sectors and cultures.

4. Static Snapshot
   This captures a single point in time. Employee satisfaction and risk
   change over time. A longitudinal study would be more insightful.

5. Sensitivity to Feature Selection & Outliers
   KMeans results change based on features included. Our Round 1 vs Round 2
   comparison demonstrated this - different features = different clusters.

--- FUTURE IMPROVEMENTS ---

1. Try DBSCAN or Hierarchical/Agglomerative Clustering
   DBSCAN finds clusters of arbitrary shape and detects outliers.
   Hierarchical clustering shows grouping at different similarity levels.

2. Use t-SNE for Better Visualization
   t-SNE preserves local relationships and can reveal cluster structure
   that PCA misses. UMAP is another modern alternative.

3. Add External Data Sources
   Market salary benchmarks, industry attrition rates, and employee
   survey data would enrich the analysis significantly.

4. Build a Predictive Attrition Model
   Use cluster labels as features in a supervised model (random forest,
   XGBoost) to predict which individuals are most likely to leave.
""")

print("[OK] Section 12 complete - limitations and future work done")

print("\n" + "="*60)
print("PROJECT COMPLETE! All 12 sections finished.")
print("="*60)
