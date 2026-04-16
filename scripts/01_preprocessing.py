# ============================================================
# SCRIPT 01: DATA PREPROCESSING
# Project: Disulfidptosis-Related Prognostic Signature for HCC
# Date: 2026
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------
# SET YOUR PATH HERE
# --------------------------------------------------
data_dir = "data/raw"
output_dir = "data/processed"
fig_dir = "results/figures/main"
table_dir = "results/tables/main"

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(table_dir, exist_ok=True)

print("=" * 65)
print("   STEP 1: LOADING ALL DATA")
print("=" * 65)

# ============================================
# 1.1 Load Expression Data
# ============================================
print("\n[Data] Loading Expression Data...")
expr = pd.read_csv(
    os.path.join(data_dir, "HiSeqV2"),
    sep='\t',
    index_col=0
)
print(f"   Shape: {expr.shape}")
print(f"   Genes: {expr.shape[0]}")
print(f"   Samples: {expr.shape[1]}")
print(f"   Value range: {expr.min().min():.2f} to {expr.max().max():.2f}")

# ============================================
# 1.2 Load Survival Data
# ============================================
print("\n[Data] Loading Survival Data...")
surv = pd.read_csv(
    os.path.join(data_dir, "survival_data.txt"),
    sep='\t'
)
print(f"   Shape: {surv.shape}")
print(f"   Columns: {list(surv.columns)}")

# ============================================
# 1.3 Load Clinical Data
# ============================================
print("\n[Data] Loading Clinical Data...")
clin = pd.read_csv(
    os.path.join(data_dir, "TCGA.LIHC.sampleMap_LIHC_clinicalMatrix"),
    sep='\t'
)
print(f"   Shape: {clin.shape}")

# ============================================
# 1.4 Load Disulfidptosis Gene List
# ============================================
print("\n[Data] Loading Disulfidptosis Gene List...")
ds_genes = pd.read_csv(
    os.path.join(data_dir, "disulfidptosis_genes.csv")
)
print(f"   Total disulfidptosis genes: {len(ds_genes)}")
print(f"   Gene list: {list(ds_genes['Gene'])}")

print("\n[OK] All data loaded successfully!")


# ============================================
# STEP 2: UNDERSTAND TCGA BARCODES
# ============================================
print("\n" + "=" * 65)
print("   STEP 2: UNDERSTANDING TCGA SAMPLE BARCODES")
print("=" * 65)

# TCGA barcode structure:
# TCGA-XX-XXXX-01A = Tumor sample (01 = Primary Solid Tumor)
# TCGA-XX-XXXX-11A = Normal sample (11 = Solid Tissue Normal)
# TCGA-XX-XXXX-02A = Recurrent Tumor (rare)

print("\n   TCGA Barcode Examples:")
print("   " + "-" * 50)
for sample in expr.columns[:5]:
    # Extract sample type code
    parts = sample.split('-')
    if len(parts) >= 4:
        sample_type = parts[3][:2]
        if sample_type == '01':
            stype = 'Tumor'
        elif sample_type == '11':
            stype = 'Normal'
        else:
            stype = f'Other ({sample_type})'
        print(f"   {sample} -> {stype}")
    else:
        print(f"   {sample} -> Unknown format")

# Separate tumor and normal samples
print("\n   Separating Tumor and Normal samples...")

tumor_samples = []
normal_samples = []

for sample in expr.columns:
    parts = sample.split('-')
    if len(parts) >= 4:
        sample_type = parts[3][:2]
        if sample_type == '01':
            tumor_samples.append(sample)
        elif sample_type == '11':
            normal_samples.append(sample)

print(f"   [OK] Tumor samples:  {len(tumor_samples)}")
print(f"   [OK] Normal samples: {len(normal_samples)}")
print(f"   [OK] Total:          {len(tumor_samples) + len(normal_samples)}")

# Create separate dataframes
expr_tumor = expr[tumor_samples]
expr_normal = expr[normal_samples]

print(f"\n   Tumor expression matrix:  {expr_tumor.shape}")
print(f"   Normal expression matrix: {expr_normal.shape}")




# ============================================
# STEP 3: FILTER LOW EXPRESSION GENES
# ============================================
print("\n" + "=" * 65)
print("   STEP 3: FILTERING LOW EXPRESSION GENES")
print("=" * 65)

print(f"\n   Before filtering: {expr.shape[0]} genes")

# Remove genes with 0 expression in more than 50% of samples
min_samples = len(expr.columns) * 0.5
genes_expressed = expr[(expr > 0).sum(axis=1) >= min_samples].index
expr_filtered = expr.loc[genes_expressed]

print(f"   After filtering (expressed in >50% samples): {expr_filtered.shape[0]} genes")
print(f"   Removed: {expr.shape[0] - expr_filtered.shape[0]} genes")

# Update tumor and normal matrices
expr_tumor_filtered = expr_filtered[tumor_samples]
expr_normal_filtered = expr_filtered[normal_samples]

print(f"\n   Filtered Tumor matrix:  {expr_tumor_filtered.shape}")
print(f"   Filtered Normal matrix: {expr_normal_filtered.shape}")


# ============================================
# STEP 4: PROCESS CLINICAL AND SURVIVAL DATA
# ============================================
print("\n" + "=" * 65)
print("   STEP 4: PROCESSING CLINICAL & SURVIVAL DATA")
print("=" * 65)

# --------------------------------------------------
# 4.1 Explore Clinical Columns
# --------------------------------------------------
print("\n   All clinical columns:")
for i, col in enumerate(clin.columns):
    print(f"   {i+1:3d}. {col}")

# --------------------------------------------------
# 4.2 Identify Key Clinical Variables
# --------------------------------------------------
print("\n   Searching for key clinical variables...")

# Find the sample ID column
id_cols = [col for col in clin.columns if 'sample' in col.lower() 
           or 'barcode' in col.lower() or col == 'sampleID']
print(f"\n   Possible ID columns: {id_cols}")

# Print first few rows to understand structure
print(f"\n   First 3 rows of clinical data:")
print(clin.head(3).to_string())


# ============================================
# STEP 4.3: EXTRACT KEY CLINICAL VARIABLES
# ============================================

# --------------------------------------------------
# ⚠️ IMPORTANT: You may need to change column names
# based on your actual clinical data columns
# Run the exploration above first, then update these
# --------------------------------------------------

# First, let's see what columns might contain our variables
print("\n   Searching for clinical variable columns...\n")

search_terms = {
    'sample_id': ['sampleID', 'sample', 'bcr_patient_barcode', 
                  '_PATIENT', 'submitter_id'],
    'age': ['age_at_initial_pathologic_diagnosis', 'age', 
            'age_at_diagnosis', 'days_to_birth',
            'age_at_initial_pathologic_diagnosis'],
    'gender': ['gender', 'sex'],
    'stage': ['clinical_stage', 'pathologic_stage', 
              'ajcc_pathologic_tumor_stage', 
              'pathological_stage_grouping',
              'tumor_stage'],
    'grade': ['histological_grade', 'neoplasm_histologic_grade',
              'tumor_grade', 'grade'],
    'T': ['pathologic_T', 'ajcc_tumor_pathologic_pt', 'T'],
    'N': ['pathologic_N', 'ajcc_nodes_pathologic_pn', 'N'],
    'M': ['pathologic_M', 'ajcc_metastasis_pathologic_pm', 'M'],
    'vital_status': ['vital_status', 'OS', 'status']
}

found_columns = {}

for var_name, possible_names in search_terms.items():
    found = False
    for col_name in possible_names:
        if col_name in clin.columns:
            found_columns[var_name] = col_name
            unique_vals = clin[col_name].dropna().unique()
            n_unique = len(unique_vals)
            print(f"   [OK] {var_name:15s} -> column: '{col_name}'")
            if n_unique <= 15:
                print(f"      Values: {sorted([str(v) for v in unique_vals])}")
            else:
                print(f"      Unique values: {n_unique}")
            found = True
            break
    if not found:
        print(f"   [!] {var_name:15s} -> NOT FOUND")
        # Try partial matching
        partial = [c for c in clin.columns 
                  if var_name.lower() in c.lower()]
        if partial:
            print(f"      Possible matches: {partial}")

print(f"\n   Found columns mapping: {found_columns}")



# ============================================
# STEP 5: MERGE EXPRESSION WITH SURVIVAL DATA
# ============================================
print("\n" + "=" * 65)
print("   STEP 5: MERGING EXPRESSION WITH SURVIVAL DATA")
print("=" * 65)

# --------------------------------------------------
# 5.1 Prepare survival data
# --------------------------------------------------
print("\n   Preparing survival data...")

# Identify ID column and rename to 'sample'
if 'sampleID' in surv.columns and 'sample' not in surv.columns:
    surv = surv.rename(columns={'sampleID': 'sample'})

# Check if OS and OS.time exist, if not calculate them
if 'OS' not in surv.columns or 'OS.time' not in surv.columns:
    print("   [!] OS or OS.time not found. Attempting to calculate from clinical fields...")
    
    # Calculate OS (1 for Dead/Deceased, 0 for Alive/Living)
    if 'vital_status' in surv.columns:
        # Map values to 0 and 1
        surv['OS'] = surv['vital_status'].apply(
            lambda x: 1 if str(x).upper() in ['DECEASED', 'DEAD'] else (0 if str(x).upper() in ['LIVING', 'ALIVE'] else np.nan)
        )
    
    # Calculate OS.time (days to death or days to last followup)
    if 'days_to_death' in surv.columns and 'days_to_last_followup' in surv.columns:
        # Convert to numeric, errors='coerce' turns strings to NaN
        surv['days_to_death'] = pd.to_numeric(surv['days_to_death'], errors='coerce')
        surv['days_to_last_followup'] = pd.to_numeric(surv['days_to_last_followup'], errors='coerce')
        
        # OS.time is days_to_death if deceased, else days_to_last_followup
        surv['OS.time'] = surv.apply(
            lambda row: row['days_to_death'] if pd.notnull(row['days_to_death']) else row['days_to_last_followup'],
            axis=1
        )

# Select and clean required columns
required_cols = ['sample', 'OS', 'OS.time']
available_cols = [c for c in required_cols if c in surv.columns]

if len(available_cols) < 3:
    print(f"   [X] Critical error: Required columns {required_cols} cannot be found or calculated.")
    print(f"   Available columns: {list(surv.columns[:15])}...")
    # Attempt to proceed with what we have, but this will likely cause issues later
    surv_clean = surv[available_cols].copy()
else:
    surv_clean = surv[required_cols].copy()

# Remove rows with missing values in survival status or time
surv_clean = surv_clean.dropna(subset=['OS', 'OS.time'])

# Ensure OS.time is numeric and positive
surv_clean['OS.time'] = pd.to_numeric(surv_clean['OS.time'], errors='coerce')
surv_clean = surv_clean[surv_clean['OS.time'] > 0]

print(f"   Survival data after cleaning:")
print(f"   Total samples: {len(surv_clean)}")
if 'OS' in surv_clean.columns:
    print(f"   Deaths (OS=1): {(surv_clean['OS']==1).sum()}")
    print(f"   Alive (OS=0):  {(surv_clean['OS']==0).sum()}")

# --------------------------------------------------
# 5.2 Match expression samples with survival
# --------------------------------------------------
print("\n   Matching expression tumor samples with survival data...")

# Get tumor samples from expression data
tumor_expr = expr_filtered[tumor_samples].T  # Transpose: samples as rows
tumor_expr.index.name = 'sample'
tumor_expr = tumor_expr.reset_index()

print(f"   Tumor expression samples: {len(tumor_expr)}")
print(f"   Survival samples: {len(surv_clean)}")

# Find common samples
common_samples = set(tumor_expr['sample']) & set(surv_clean['sample'])
print(f"   Common samples: {len(common_samples)}")

# If no common samples, try matching with shorter barcode
if len(common_samples) == 0:
    print("\n   [!] No direct match found. Trying shorter barcode...")
    
    # Try truncating to 15 characters (TCGA-XX-XXXX-01)
    tumor_expr['sample_short'] = tumor_expr['sample'].str[:15]
    surv_clean['sample_short'] = surv_clean['sample'].str[:15]
    
    common_samples_short = set(tumor_expr['sample_short']) & set(surv_clean['sample_short'])
    print(f"   Common samples (short barcode): {len(common_samples_short)}")
    
    if len(common_samples_short) > 0:
        # Merge using short barcode
        merged = pd.merge(
            tumor_expr,
            surv_clean[['sample_short', 'OS', 'OS.time']],
            on='sample_short',
            how='inner'
        )
        print(f"   [OK] Merged successfully using short barcode!")
    else:
        # Try even shorter (TCGA-XX-XXXX)
        tumor_expr['sample_short2'] = tumor_expr['sample'].str[:12]
        surv_clean['sample_short2'] = surv_clean['sample'].str[:12]
        
        common_samples_short2 = set(tumor_expr['sample_short2']) & set(surv_clean['sample_short2'])
        print(f"   Common samples (patient ID): {len(common_samples_short2)}")
        
        merged = pd.merge(
            tumor_expr,
            surv_clean[['sample_short2', 'OS', 'OS.time']],
            on='sample_short2',
            how='inner'
        )
        print(f"   [OK] Merged successfully using patient ID!")
else:
    # Direct merge
    merged = pd.merge(
        tumor_expr,
        surv_clean[['sample', 'OS', 'OS.time']],
        on='sample',
        how='inner'
    )
    print(f"   [OK] Merged successfully with direct match!")

print(f"\n   Final merged dataset:")
print(f"   Samples: {len(merged)}")
print(f"   Genes: {len(merged.columns) - 4}")  # Subtract ID + OS + OS.time columns
print(f"   Deaths: {(merged['OS']==1).sum()}")
print(f"   Alive: {(merged['OS']==0).sum()}")



# ============================================
# STEP 6: ADD CLINICAL VARIABLES
# ============================================
print("\n" + "=" * 65)
print("   STEP 6: ADDING CLINICAL VARIABLES")
print("=" * 65)

# --------------------------------------------------
# 6.1 Prepare clinical data
# --------------------------------------------------

# First identify the sample ID column in clinical data
print("\n   First column of clinical data:", clin.columns[0])
print("   First few values:", list(clin.iloc[:3, 0]))

# The first column is likely the sample ID
# Rename it for merging
clin_id_col = clin.columns[0]

# Create a short version for matching
clin['sample_match'] = clin[clin_id_col].astype(str)

# Try to match with merged data
# First check what ID format clinical data uses
print(f"\n   Clinical ID format: {clin['sample_match'].iloc[0]}")
print(f"   Expression ID format: {merged['sample'].iloc[0]}")

# --------------------------------------------------
# 6.2 Select key clinical columns
# --------------------------------------------------
# We'll select the columns we found earlier
# Adjust these column names based on YOUR data

clinical_cols_to_keep = [clin_id_col]

# Add found clinical variables
for var_name, col_name in found_columns.items():
    if col_name in clin.columns and col_name != clin_id_col:
        clinical_cols_to_keep.append(col_name)

clin_subset = clin[clinical_cols_to_keep].copy()
clin_subset = clin_subset.rename(columns={clin_id_col: 'patient_id'})

print(f"\n   Clinical variables selected:")
for col in clin_subset.columns:
    print(f"   - {col}")

print(f"\n   Clinical data shape: {clin_subset.shape}")


# ============================================
# STEP 7: SAVE ALL PROCESSED DATA
# ============================================
print("\n" + "=" * 65)
print("   STEP 7: SAVING PROCESSED DATA")
print("=" * 65)

# --------------------------------------------------
# 7.1 Save tumor expression matrix
# --------------------------------------------------
expr_tumor_filtered.to_csv(
    os.path.join(output_dir, "expression_tumor.csv")
)
print(f"   [OK] Saved: expression_tumor.csv")
print(f"      Shape: {expr_tumor_filtered.shape}")

# --------------------------------------------------
# 7.2 Save normal expression matrix
# --------------------------------------------------
expr_normal_filtered.to_csv(
    os.path.join(output_dir, "expression_normal.csv")
)
print(f"   [OK] Saved: expression_normal.csv")
print(f"      Shape: {expr_normal_filtered.shape}")

# --------------------------------------------------
# 7.3 Save merged expression + survival data
# --------------------------------------------------
merged.to_csv(
    os.path.join(output_dir, "expression_with_survival.csv"),
    index=False
)
print(f"   [OK] Saved: expression_with_survival.csv")
print(f"      Shape: {merged.shape}")

# --------------------------------------------------
# 7.4 Save full expression matrix (filtered)
# --------------------------------------------------
expr_filtered.to_csv(
    os.path.join(output_dir, "expression_all_filtered.csv")
)
print(f"   [OK] Saved: expression_all_filtered.csv")
print(f"      Shape: {expr_filtered.shape}")

# --------------------------------------------------
# 7.5 Save sample lists
# --------------------------------------------------
pd.DataFrame({'sample': tumor_samples}).to_csv(
    os.path.join(output_dir, "tumor_samples.csv"), index=False
)
pd.DataFrame({'sample': normal_samples}).to_csv(
    os.path.join(output_dir, "normal_samples.csv"), index=False
)
print(f"   [OK] Saved: tumor_samples.csv ({len(tumor_samples)} samples)")
print(f"   [OK] Saved: normal_samples.csv ({len(normal_samples)} samples)")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 65)
print("   DAY 2 PREPROCESSING SUMMARY")
print("=" * 65)
print(f"""
   EXPRESSION DATA:
   * Total genes (after filtering): {expr_filtered.shape[0]}
   * Tumor samples:                 {len(tumor_samples)}
   * Normal samples:                {len(normal_samples)}

   SURVIVAL DATA:
   * Patients with OS data:         {len(merged)}
   * Deaths (events):               {(merged['OS']==1).sum()}
   * Alive (censored):              {(merged['OS']==0).sum()}

   FILES SAVED:
   * expression_tumor.csv
   * expression_normal.csv
   * expression_with_survival.csv
   * expression_all_filtered.csv
   * tumor_samples.csv
   * normal_samples.csv

   [OK] PREPROCESSING COMPLETE!
   [OK] Ready for Day 3: DEG Analysis
""")
