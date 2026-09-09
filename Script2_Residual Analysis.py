import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Memory

from statsforecast import StatsForecast
from statsforecast.models import MSTL, SeasonalNaive
from hierarchicalforecast.utils import aggregate

import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# ---------------------------------------------------------
# 1. Configuration & Caching Setup
# ---------------------------------------------------------
# Set up caching directory to prevent redundant model fitting
cache_dir = os.path.join(os.getcwd(), 'sf_cache')
if not os.path.exists(cache_dir): 
    os.makedirs(cache_dir)
memory = Memory(cache_dir, verbose=0)

CONFIG = {
    'FILENAME': 'data_9features.xlsx', 
    'HORIZON': 12,
    'FOLDS': 4,
    'STEP_SIZE': 12,
    'GROUPS': ['GFA', 'Consents', 'Value'] 
}

# ---------------------------------------------------------
# 2. Data Loading & Hierarchy Setup
# ---------------------------------------------------------
def load_and_filter_hierarchy(filename, group_keyword):
    """
    Loads dataset and filters columns based on the specified hierarchy group.
    Transforms data into a long format and generates hierarchical aggregation tags.
    """
    df = pd.read_excel(filename)
    date_col = df.columns[0]
    
    group_cols = [date_col] + [c for c in df.columns if group_keyword in c]
    df_group = df[group_cols].copy()
    
    df_group = df_group.rename(columns={date_col: 'ds'})
    df_group['ds'] = pd.to_datetime(df_group['ds'])
    
    df_melted = df_group.melt(id_vars='ds', var_name='full_id', value_name='y')
    df_melted[['Sub', 'Type']] = df_melted['full_id'].str.rsplit(' ', n=1, expand=True)
    
    hier_df, S_df, tags = aggregate(df_melted, [['Type'], ['Type', 'Sub']])
    return hier_df.reset_index(), tags

# ---------------------------------------------------------
# 3. Cross-Validation Engine
# ---------------------------------------------------------
@memory.cache
def run_cached_cv(df, horizon, n_windows, step_size):
    """
    Executes cross-validation specifically for the MSTL baseline model.
    Results are cached to accelerate experimental runs.
    """
    models = [MSTL(season_length=[12]), SeasonalNaive(season_length=12)]
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)
    
    return sf.cross_validation(df=df, h=horizon, step_size=step_size, n_windows=n_windows)

# ---------------------------------------------------------
# 4. Statistical Residual Diagnostics
# ---------------------------------------------------------
def calculate_residual_diagnostics(df, tags):
    """
    Calculates residual diagnostics (Bias and Ljung-Box Autocorrelation test)
    for the MSTL base model.
    """
    diag_list = []
    model = 'MSTL'
    
    for uid in df['unique_id'].unique():
        feat_df = df[df['unique_id'] == uid].copy()
        
        # Collapse overlapping CV folds into a single chronological path
        feat_df = feat_df.groupby('ds').mean(numeric_only=True).reset_index()
        feat_df = feat_df.sort_values('ds')
        
        y_true = feat_df['y'].values
        preds = feat_df[model].values
        residuals = y_true - preds
        
        level = 'Total' if uid in tags['Type'] else 'Bottom'
        
        # 1. Bias (Mean Error)
        bias = np.mean(residuals)
        
        # 2. Autocorrelation (Ljung-Box Test at lag 12 for monthly data)
        try:
            lb_res = acorr_ljungbox(residuals, lags=[12], return_df=True)
            lb_pvalue = lb_res['lb_pvalue'].iloc[0]
        except:
            lb_pvalue = np.nan
            
        diag_list.append({
            'Feature': uid, 'Level': level, 'Model': model,
            'Bias': bias, 'Ljung-Box(p)': lb_pvalue
        })
        
    return pd.DataFrame(diag_list)

# ---------------------------------------------------------
# 5. Diagnostic Plotting Engine
# ---------------------------------------------------------
def plot_residual_diagnostics(df, feature, model='MSTL'):
    """
    Generates a comprehensive 4-panel diagnostic plot for a given feature's residuals.
    Includes Time Plot, Distribution (KDE), Normal Q-Q Plot, and Autocorrelation (ACF).
    """
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'Cambria'})
    
    feat_df = df[df['unique_id'] == feature].copy()
    
    # Collapse overlapping CV folds
    feat_df = feat_df.groupby('ds').mean(numeric_only=True).reset_index()
    feat_df = feat_df.sort_values('ds')
    
    residuals = feat_df['y'] - feat_df[model]
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'MSTL Residual Diagnostics: {feature}', fontsize=18, fontweight='bold', y=0.98)
    
    # 1. Time Plot (Bias Check)
    ax1 = plt.subplot(221)
    ax1.plot(feat_df['ds'], residuals, marker='o', linestyle='-', color='#1f77b4', markersize=4)
    ax1.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax1.set_title('Residuals over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Residual Value')
    
    # 2. Histogram + KDE (Normality Check)
    ax2 = plt.subplot(222)
    sns.histplot(residuals, kde=True, ax=ax2, color='#ff7f0e', edgecolor='black')
    ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Frequency')
    
    # 3. Q-Q Plot
    ax3 = plt.subplot(223)
    sm.qqplot(residuals, line='s', ax=ax3, markerfacecolor='#2ca02c', markeredgecolor='black', alpha=0.7)
    ax3.set_title('Normal Q-Q Plot', fontsize=14, fontweight='bold')
    
    # 4. ACF Plot (Information Leakage Check)
    ax4 = plt.subplot(224)
    lags = min(24, len(residuals) // 2 - 1) 
    sm.graphics.tsa.plot_acf(residuals, lags=lags, ax=ax4, color='#d62728', vlines_kwargs={"colors": '#d62728'})
    ax4.set_title(f'Autocorrelation Function (ACF) - Lags: {lags}', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()

# ---------------------------------------------------------
# 6. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    all_diag_dfs = []
    cv_results_list = [] 

    for group in CONFIG['GROUPS']:
        print(f"\nProcessing MSTL Base Diagnostics: {group}")
        try:
            hier_df, tags = load_and_filter_hierarchy(CONFIG['FILENAME'], group)
            cv_results = run_cached_cv(hier_df, CONFIG['HORIZON'], CONFIG['FOLDS'], CONFIG['STEP_SIZE'])
            
            group_diags = calculate_residual_diagnostics(cv_results, tags)
            group_diags['Group'] = group
            all_diag_dfs.append(group_diags)
            cv_results_list.append(cv_results)
            
        except Exception as e:
            print(f"Error processing group {group}: {e}")

    diag_df = pd.concat(all_diag_dfs, ignore_index=True)
    full_cv_df = pd.concat(cv_results_list, ignore_index=True)

    # --- PRINT: STATISTICAL DIAGNOSTICS SUMMARY ---
    print("\n" + "="*80)
    print("MSTL RESIDUAL DIAGNOSTICS SUMMARY")
    print("="*80)
    print(f"{'GROUP':<10} | {'FEATURE':<25} | {'BIAS':>10} | {'LJUNG-BOX(p)':>15}")
    print("-" * 80)
    
    sorted_diag = diag_df.sort_values(['Group', 'Level', 'Feature'])
    
    for _, row in sorted_diag.iterrows():
        lb_flag = "*" if row['Ljung-Box(p)'] < 0.05 else " "
        print(f"{row['Group']:<10} | {row['Feature']:<25} | {row['Bias']:>10.2f} | {row['Ljung-Box(p)']:>13.4f}{lb_flag}")

    print("\n* Denotes statistically significant autocorrelation (p < 0.05).")
    
    # --- RENDER PLOTS ---
    print("\nGenerating Visual Diagnostics...")
    for feature in diag_df['Feature'].unique():
        plot_residual_diagnostics(full_cv_df, feature=feature, model='MSTL')
        
    print("\nProcessing complete.")