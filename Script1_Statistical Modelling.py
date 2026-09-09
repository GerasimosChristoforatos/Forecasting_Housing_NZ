import os
import pandas as pd
import numpy as np
from joblib import Memory

from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA, AutoETS, AutoTheta, AutoCES,
    DynamicOptimizedTheta, MSTL,
    SeasonalExponentialSmoothingOptimized,
    CrostonOptimized, SeasonalNaive
)
from hierarchicalforecast.utils import aggregate

# ---------------------------------------------------------
# 1. Configuration & Caching Setup
# ---------------------------------------------------------
# Set up caching directory to prevent redundant model fitting
cache_dir = os.path.join(os.getcwd(), 'statsforecast_cache')
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
    Executes cross-validation across a suite of classical statistical models.
    Results are cached to accelerate repeated experimental runs.
    """
    models = [
        AutoARIMA(season_length=12),
        AutoETS(season_length=12),
        AutoTheta(season_length=12),
        AutoCES(season_length=12),
        DynamicOptimizedTheta(season_length=12),
        MSTL(season_length=12),
        SeasonalExponentialSmoothingOptimized(season_length=12),
        CrostonOptimized(),
        SeasonalNaive(season_length=12) # Benchmark baseline
    ]
    
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)
    
    cv_df = sf.cross_validation(
        df=df, h=horizon, step_size=step_size, n_windows=n_windows
    )
    return cv_df

# ---------------------------------------------------------
# 4. Metrics Calculation 
# ---------------------------------------------------------
def calculate_metrics(cv_df, tags):
    """
    Calculates scale-dependent (MAE, RMSE, MSE) and scale-independent (sMAPE, rMAE) 
    error metrics for each feature and model.
    """
    metrics_list = []
    total_ids = tags['Type']
    model_cols = [c for c in cv_df.columns if c not in ['unique_id', 'ds', 'cutoff', 'y']]
    
    for uid in cv_df['unique_id'].unique():
        feat_df = cv_df[cv_df['unique_id'] == uid]
        level = 'Total' if uid in total_ids else 'Bottom'
        
        y_true = feat_df['y'].values
        y_naive = feat_df['SeasonalNaive'].values
        mae_naive = np.mean(np.abs(y_true - y_naive)) + 1e-9
        
        # Designate top-level aggregated features
        feature_name = f"Total {uid}" if level == 'Total' else uid
        
        for model in model_cols:
            y_pred = feat_df[model].values
            
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred)**2) 
            rmse = np.sqrt(mse)
            rmae = mae / mae_naive
            smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-9))
            
            metrics_list.append({
                'Feature': feature_name, 'Level': level, 'Model': model,
                'MAE': mae, 'RMSE': rmse, 'MSE': mse, 'sMAPE': smape, 'rMAE': rmae
            })
            
    return pd.DataFrame(metrics_list)

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    all_metrics_dfs = []

    for group in CONFIG['GROUPS']:
        print(f"\nProcessing Hierarchy Group: {group}")
        try:
            hier_df, tags = load_and_filter_hierarchy(CONFIG['FILENAME'], group)
            cv_results = run_cached_cv(hier_df, CONFIG['HORIZON'], CONFIG['FOLDS'], CONFIG['STEP_SIZE'])
            
            group_metrics = calculate_metrics(cv_results, tags)
            group_metrics['Group'] = group
            all_metrics_dfs.append(group_metrics)
        except Exception as e:
            print(f"Error processing group {group}: {e}")

    metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)

    # --- PRINT 1: FULL DETAIL PER FEATURE ---
    print("\n" + "="*145)
    print(f"{'GROUP':<10} | {'LEVEL':<8} | {'FEATURE':<25} | {'MODEL':<35} | {'MAE':>10} | {'RMSE':>10} | {'MSE':>12} | {'sMAPE':>10} | {'rMAE':>10}")
    print("-" * 145)
    
    sorted_df = metrics_df.sort_values(['Group', 'Level', 'Feature', 'MAE'], ascending=[True, False, True, True])
    for _, row in sorted_df.iterrows():
        print(f"{row['Group']:<10} | {row['Level']:<8} | {row['Feature']:<25} | {row['Model']:<35} | {row['MAE']:>10.2f} | {row['RMSE']:>10.2f} | {row['MSE']:>12.2f} | {row['sMAPE']:>9.2f}% | {row['rMAE']:>10.4f}")

    # --- PRINT 2: HIERARCHY-SPECIFIC AVERAGES ---
    print("\n" + "="*60)
    print("HIERARCHY-SPECIFIC AVERAGES")
    print("="*60)
    
    for group in CONFIG['GROUPS']:
        print(f"\nSUMMARY FOR: {group}")
        for lvl in ['Total', 'Bottom']:
            group_lvl_df = metrics_df[(metrics_df['Group'] == group) & (metrics_df['Level'] == lvl)]
            if not group_lvl_df.empty:
                print(f"--- {lvl.upper()} LEVEL ---")
                
                summary_sum = group_lvl_df.groupby('Model')[['MAE', 'MSE']].sum()
                summary_sum['RMSE'] = np.sqrt(summary_sum['MSE']) 
                summary_mean = group_lvl_df.groupby('Model')[['sMAPE', 'rMAE']].mean()
                
                summary = pd.concat([summary_sum[['MAE', 'RMSE', 'MSE']], summary_mean], axis=1)
                print(summary.sort_values('MAE').round(4))

    # --- PRINT 3: GLOBAL AVERAGES ---
    print("\n" + "="*72)
    print("GLOBAL AGGREGATED PERFORMANCE (DIMENSIONLESS METRICS)")
    print("="*72)
    
    for lvl in ['Total', 'Bottom']:
        print(f"\nGLOBAL {lvl.upper()} LEVEL AVERAGE:")
        lvl_df = metrics_df[metrics_df['Level'] == lvl]
        
        g_sum = lvl_df.groupby('Model')[['MAE', 'MSE']].sum()
        g_sum['RMSE'] = np.sqrt(g_sum['MSE'])
        g_mean = lvl_df.groupby('Model')[['sMAPE', 'rMAE']].mean()
        
        global_summary = pd.concat([g_sum[['MAE', 'RMSE', 'MSE']], g_mean], axis=1)
        print(global_summary[['rMAE', 'sMAPE']].sort_values('rMAE').round(4))
        
    print("\nProcessing complete.")
    
# --- ADD THIS TO SCRIPT 1 TO TEST THE FRANKENSTEIN MODEL ---
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace

print("\n" + "="*60)
print("RUNNING FRANKENSTEIN (BEST-STAT) RECONCILIATION TEST")
print("="*60)

# 1. Define the map based on your feature-wise best models
best_model_map = {
    'Consents Apartments': 'MSTL',
    'Consents Detached': 'MSTL',
    'GFA': 'MSTL', # Assuming 'Total GFA' unique_id is just 'GFA'
    'GFA Detached': 'MSTL',
    'Value': 'MSTL', # Assuming 'Total Value' unique_id is just 'Value'
    'Value Detached': 'MSTL',
    'Consents': 'DynamicOptimizedTheta', # Assuming 'Total Consents'
    'GFA Apartments': 'DynamicOptimizedTheta',
    'Value Apartments': 'DynamicOptimizedTheta',
    'Consents Townhouses': 'AutoTheta',
    'GFA Townhouses': 'AutoTheta',
    'Value Townhouses': 'AutoTheta'
}

all_reconciled_frankensteins = []

for group in CONFIG['GROUPS']:
    # Load data for the group
    hier_df, tags = load_and_filter_hierarchy(CONFIG['FILENAME'], group)
    cv_results = run_cached_cv(hier_df, CONFIG['HORIZON'], CONFIG['FOLDS'], CONFIG['STEP_SIZE'])
    
    # 2. Create the Frankenstein column
    cv_results['Frankenstein'] = np.nan
    for uid, model_name in best_model_map.items():
        # Make sure the uid exists in this group's cv_results
        if uid in cv_results['unique_id'].unique():
            mask = cv_results['unique_id'] == uid
            cv_results.loc[mask, 'Frankenstein'] = cv_results.loc[mask, model_name]
    
    # Fill any missing mappings with MSTL just in case the ID names slightly differ
    cv_results['Frankenstein'] = cv_results['Frankenstein'].fillna(cv_results['MSTL'])
    
    # 3. Reconcile the Frankenstein column alongside MSTL
    cv_numeric = cv_results[['unique_id', 'ds', 'y', 'MSTL', 'Frankenstein']].copy()
    
    # Need S_df for reconciliation (load_and_filter_hierarchy in script 1 doesn't return S_df, 
    # so we quickly recreate it here or copy the function from script 4)
    df_melted = hier_df.melt(id_vars='ds', var_name='full_id', value_name='y_dummy') if 'ds' in hier_df.columns else hier_df # simplify for extraction
    # Use the hierarchical forecast setup
    reconcilers = [MinTrace(method='emint', nonnegative=False)]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)
    
    # Note: To run this seamlessly, ensure load_and_filter_hierarchy returns S_df like in Script 4.
    _, S_df, _ = load_and_filter_hierarchy(CONFIG['FILENAME'], group) # Assuming you adapt Script 1 to return S_df
    
    reconciled_df = hrec.reconcile(Y_hat_df=cv_numeric, Y_df=cv_numeric, S_df=S_df, tags=tags).reset_index()
    all_reconciled_frankensteins.append(reconciled_df)

final_frank_df = pd.concat(all_reconciled_frankensteins, ignore_index=True)

# 4. Quick MAE calculation to compare
frank_metrics = []
for uid in final_frank_df['unique_id'].unique():
    feat_df = final_frank_df[final_frank_df['unique_id'] == uid]
    y_true = feat_df['y'].values
    
    mae_mstl_mint = np.mean(np.abs(y_true - feat_df['MSTL/MinTrace_method-emint'].values))
    mae_frank_mint = np.mean(np.abs(y_true - feat_df['Frankenstein/MinTrace_method-emint'].values))
    
    frank_metrics.append({
        'Feature': uid,
        'MSTL_MinT MAE': mae_mstl_mint,
        'Frankenstein_MinT MAE': mae_frank_mint,
        'Winner': 'MSTL_MinT' if mae_mstl_mint < mae_frank_mint else 'Frankenstein_MinT'
    })

print(pd.DataFrame(frank_metrics))