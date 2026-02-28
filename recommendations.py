# recommendations.py - Sections 11, 12: HR Recommendations & Limitations

def print_recommendations():
    """Print HR recommendations and project limitations."""
    
    # ============================================================
    # SECTION 11 -- HR Recommendations
    # ============================================================
    
    print("\n" + "="*60)
    print("SECTION 11 -- HR RECOMMENDATIONS")
    print("="*60)
    
    print("""
--- CLUSTER SUMMARY TABLE ---

| Metric                | Experienced Loyal | Mid-Level Moderate | Mid-Career Moderate | Young At-Risk          |
|-----------------------|-------------------|--------------------|---------------------|------------------------|
| Avg Age               | ~42-48            | ~34-38             | ~33-37              | ~28-35                 |
| Monthly Income        | High              | Medium             | Medium              | Low                    |
| Total Working Years   | 15+               | 8-12               | 6-10                | 2-10                   |
| OverTime Rate         | Moderate          | High               | Moderate            | High                   |
| Attrition Risk        | Lowest (8.7%)     | Low (11.3%)        | Moderate (13.4%)    | HIGHEST (28.1%)        |
""")
    
    print("--- HIGH ATTRITION CLUSTER: Young At-Risk (Cluster 3) ---")
    print("""  
Profile: Youngest professionals (avg age 33.5) with lowest pay ($4,693) and
high overtime rates. HIGHEST attrition at 28.1% among all segments.

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
  Structured mentorship programs pairing them with Experienced Loyal employees,
  clear promotion criteria, and professional development budgets demonstrate
  investment in their future.
""")
    
    print("--- MODERATE ATTRITION: Mid-Career Moderate (Cluster 2) ---")
    print("""  
Profile: Largest group (506 employees), mid-career with moderate experience
and balanced work-life. 13.4% attrition — reducible.

Recommendation 1: Role Enrichment - offer lateral movement opportunities,
  cross-functional projects, and leadership training to prevent stagnation.

Recommendation 2: Recognition Programs - implement peer recognition systems
  and performance-based bonuses to make them feel valued.
""")
    
    print("--- LOW ATTRITION CLUSTERS ---")
    print("""  
Experienced Loyal (Cluster 0): Oldest (45.1yr), highest paid ($13,349),
  most experienced (22.8yr). Attrition only 8.7%.
  MAINTAIN current work-life balance policies. Don't disrupt what works.
  LEVERAGE them as mentors for Young At-Risk employees.

Mid-Level Moderate (Cluster 1): Age ~36, income $5,284, 9.5yr experience.
  Only 11.3% attrition. ENSURE compensation stays competitive with market
  rates. Provide challenging projects and structured raise schedules.
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
