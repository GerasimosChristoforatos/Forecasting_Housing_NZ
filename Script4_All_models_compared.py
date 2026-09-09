import os
import sys
import copy
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import seaborn as sns
import optuna
from joblib import Memory

from statsmodels.graphics.tsaplots import plot_acf
from pytorch_lightning.callbacks import Callback

# Suppress warnings and restrict threads to prevent deadlocks during multiprocessing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

optuna.logging.set_verbosity(optuna.logging.WARNING)
loggers_to_mute = [
    "pytorch_lightning", "pytorch_lightning.utilities.rank_zero",
    "lightning_fabric.utilities.seed", "lightning_fabric.utilities.warnings", "neuralforecast"
]
for logger_name in loggers_to_mute:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from statsforecast import StatsForecast
from statsforecast.models import MSTL, SeasonalNaive
from hierarchicalforecast.utils import aggregate
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from neuralforecast import NeuralForecast
from neuralforecast.models import LSTM

# ---------------------------------------------------------
# 1. Configuration & Caching Setup
# ---------------------------------------------------------
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

TARGET_FEATURES = {
    'GFA': ['Detached', 'Townhouses'],
    'Consents': ['Detached', 'Townhouses'],
    'Value': ['Detached']
}

# Standardized color palette for model visualizations
GLOBAL_COLORS = {
    'LSTM': '#e6194b',
    'MSTL': '#4363d8',
    'MSTL_LSTM': '#ab0086',
    'LSTM_MinT': '#ffe400',
    'MSTL_MinT': '#00ffb9',
    'MSTL_LSTM_MinT': '#ffa034'
}

# ---------------------------------------------------------
# 2. Custom Validation Callback
# ---------------------------------------------------------
class InMemoryRestoreBestWeights(Callback):
    """
    Monitors validation loss during training and retains the model weights 
    from the best epoch in memory to prevent overfitting.
    """
    def __init__(self):
        super().__init__()
        self.best_weights = None
        self.best_score = float('inf')

    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        current_score = metrics.get('ptl/val_loss', metrics.get('val_loss'))

        if current_score is None:
            return

        current_score = current_score.item() if hasattr(current_score, 'item') else current_score

        if current_score < self.best_score:
            self.best_score = current_score
            self.best_weights = copy.deepcopy(pl_module.state_dict())

    def on_train_end(self, trainer, pl_module):
        if self.best_weights is not None:
            pl_module.load_state_dict(self.best_weights)

# ---------------------------------------------------------
# 3. Data Processing & Hierarchy Setup
# ---------------------------------------------------------
def load_and_filter_hierarchy(filename, group_keyword):
    """
    Loads data and applies hierarchical tags for structural evaluation.
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
    return hier_df.reset_index(), S_df, tags

# ---------------------------------------------------------
# 4. Cached Cross-Validation Engine
# ---------------------------------------------------------
@memory.cache
def run_cached_cv(df, horizon, n_windows, step_size):
    """
    Computes expanding-window cross-validation for baseline models (MSTL, SeasonalNaive).
    """
    models = [MSTL(season_length=12), SeasonalNaive(season_length=12)]
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)

    cv_df = sf.cross_validation(
        df=df, h=horizon, step_size=step_size, n_windows=n_windows, fitted=True
    )
    fitted_df = sf.cross_validation_fitted_values()
    return cv_df, fitted_df

# ---------------------------------------------------------
# 5. Hybrid Engine: MSTL + LSTM Residual Correction
# ---------------------------------------------------------
def apply_mstl_lstm_residual_learning(cv_df, fitted_df, horizon, tags, group_name):
    """
    Trains an LSTM to forecast the stationary residuals generated by the MSTL baseline.
    """
    target_subs = TARGET_FEATURES.get(group_name, [])
    print(f"Training hybrid MSTL-LSTM on targeted residuals for {group_name}...")

    res_cv = cv_df.copy()
    res_cv['MSTL_LSTM'] = res_cv['MSTL']

    total_ids = tags['Type']
    bottom_ids = [uid for uid in cv_df['unique_id'].unique() if uid not in total_ids]

    target_bottom_ids = [uid for uid in bottom_ids if any(sub in uid for sub in target_subs)]
    target_all_ids = target_bottom_ids + list(total_ids)

    cutoffs = sorted(cv_df['cutoff'].unique())
    fold_col = 'cutoff' if 'cutoff' in fitted_df.columns else 'fold'
    folds = sorted(fitted_df[fold_col].unique())

    for i, cutoff in enumerate(cutoffs):
        fold_id = folds[i]

        train_fold = fitted_df[fitted_df[fold_col] == fold_id].copy()
        target_train = train_fold[train_fold['unique_id'].isin(target_all_ids)].copy()

        if target_train.empty:
            continue

        target_train['y_res'] = target_train['y'] - target_train['MSTL']
        core_train = target_train[['unique_id', 'ds', 'y_res']].rename(columns={'y_res': 'y'})

        input_size = 3
        windows_batch_size = 2048
        patience_epochs = 10

        total_datapoints = len(core_train)
        num_features = core_train['unique_id'].nunique()
        total_windows = max(1, total_datapoints - (num_features * (input_size + horizon - 1)))
        steps_per_epoch = int(np.ceil(total_windows / windows_batch_size))

        dl_models = [
            LSTM(
                h=horizon,
                input_size=input_size,
                encoder_hidden_size=40,
                decoder_hidden_size=48,
                learning_rate=0.003827587073536582,
                encoder_dropout=0.006037174144305116,
                scaler_type='minmax',
                windows_batch_size=windows_batch_size,
                max_steps=150 * steps_per_epoch,
                val_check_steps=steps_per_epoch,
                early_stop_patience_steps=patience_epochs,
                callbacks=[InMemoryRestoreBestWeights()],
                enable_model_summary=False,
                enable_progress_bar=False,
                random_seed=42,
                logger=False
            )
        ]

        nf = NeuralForecast(models=dl_models, freq='MS')
        nf.fit(df=core_train, val_size=horizon)
        pred_dl = nf.predict()

        mask = res_cv['cutoff'] == cutoff
        fold_cv = res_cv[mask].copy()
        fold_cv = fold_cv.merge(pred_dl, on=['unique_id', 'ds'], how='left')

        fold_cv['MSTL_LSTM'] = fold_cv['MSTL'] + fold_cv['LSTM'].fillna(0)
        res_cv.loc[mask, 'MSTL_LSTM'] = fold_cv['MSTL_LSTM'].values

    return res_cv

# ---------------------------------------------------------
# 6. Baseline DL Engine: Pure LSTM 
# ---------------------------------------------------------
def apply_vanilla_lstm_learning(cv_df, fitted_df, horizon, tags, group_name):
    """
    Trains a standalone LSTM architecture directly on the target series for baseline comparison.
    """
    print(f"Training standalone LSTM directly on original features for {group_name}...")

    res_cv = cv_df.copy()
    res_cv['LSTM'] = np.nan

    total_ids = list(tags['Type'])
    bottom_ids = [uid for uid in cv_df['unique_id'].unique() if uid not in total_ids]
    target_all_ids = bottom_ids + total_ids

    cutoffs = sorted(cv_df['cutoff'].unique())
    fold_col = 'cutoff' if 'cutoff' in fitted_df.columns else 'fold'
    folds = sorted(fitted_df[fold_col].unique())

    for i, cutoff in enumerate(cutoffs):
        fold_id = folds[i]

        train_fold = fitted_df[fitted_df[fold_col] == fold_id].copy()
        target_train = train_fold[train_fold['unique_id'].isin(target_all_ids)].copy()

        if target_train.empty:
            continue

        core_train = target_train[['unique_id', 'ds', 'y']].copy()

        input_size = 3
        windows_batch_size = 128
        patience_epochs = 10

        total_datapoints = len(core_train)
        num_features = core_train['unique_id'].nunique()
        total_windows = max(1, total_datapoints - (num_features * (input_size + horizon - 1)))
        steps_per_epoch = int(np.ceil(total_windows / windows_batch_size))

        dl_models = [
            LSTM(
                h=horizon,
                input_size=input_size,
                encoder_hidden_size=42,
                decoder_hidden_size=113,
                learning_rate=0.000118322919390128,
                encoder_dropout=0.131804763512595,
                scaler_type='minmax',
                windows_batch_size=windows_batch_size,
                max_steps=150 * steps_per_epoch,
                val_check_steps=steps_per_epoch,
                early_stop_patience_steps=patience_epochs,
                callbacks=[InMemoryRestoreBestWeights()],
                enable_model_summary=False,
                enable_progress_bar=False,
                random_seed=42,
                logger=False,
            )
        ]

        nf = NeuralForecast(models=dl_models, freq='MS')
        nf.fit(df=core_train, val_size=horizon)
        pred_dl = nf.predict()

        mask = res_cv['cutoff'] == cutoff
        fold_cv = res_cv[mask].copy()

        if 'LSTM' in fold_cv.columns:
            fold_cv = fold_cv.drop(columns=['LSTM'])

        fold_cv = fold_cv.merge(pred_dl, on=['unique_id', 'ds'], how='left')
        res_cv.loc[mask, 'LSTM'] = fold_cv['LSTM'].fillna(0).values

    return res_cv

# ---------------------------------------------------------
# 7. Hierarchical Reconciliation 
# ---------------------------------------------------------
def apply_grand_reconciliation(cv_df, S_df, tags):
    """
    Applies MinTrace (Empirical) reconciliation to align base forecasts across hierarchical levels.
    """
    cv_df_numeric = cv_df.drop(columns=['cutoff'])
    reconcilers = [MinTrace(method='emint', nonnegative=False)]
    hrec = HierarchicalReconciliation(reconcilers=reconcilers)

    reconciled_df = hrec.reconcile(Y_hat_df=cv_df_numeric, Y_df=cv_df_numeric, S_df=S_df, tags=tags)
    reconciled_df = reconciled_df.reset_index()

    rename_dict = {
        'MSTL/MinTrace_method-emint': 'MSTL_MinT',
        'MSTL_LSTM/MinTrace_method-emint': 'MSTL_LSTM_MinT',
        'LSTM/MinTrace_method-emint': 'LSTM_MinT'
    }
    return reconciled_df.rename(columns=rename_dict)

# ---------------------------------------------------------
# 8. Evaluation Metrics Calculation
# ---------------------------------------------------------
def calculate_metrics(df, tags):
    """
    Computes comparative forecast accuracy metrics per structural feature.
    """
    metrics_list = []
    total_ids = tags['Type']
    eval_models = ['SeasonalNaive', 'MSTL', 'LSTM', 'MSTL_LSTM', 'MSTL_MinT', 'LSTM_MinT', 'MSTL_LSTM_MinT']

    for uid in df['unique_id'].unique():
        feat_df = df[df['unique_id'] == uid]
        level = 'Total' if uid in total_ids else 'Bottom'

        y_true = feat_df['y'].values
        y_naive = feat_df['SeasonalNaive'].values
        mae_naive = np.mean(np.abs(y_true - y_naive)) + 1e-9

        feature_name = f"Total {uid}" if level == 'Total' else uid

        for model in eval_models:
            if model not in df.columns or feat_df[model].isna().all(): 
                continue

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
# 9. Performance Visualization
# ---------------------------------------------------------
def generate_plots(metrics_df, chosen_cmap="copper"):
    print(f"\nGenerating performance heatmaps...")
    sns.set_theme(style="white")
    plt.rcParams.update({'font.family': 'Cambria'})
    plt.rcParams.update({'hatch.linewidth': 0.5})

    def logical_row_sort(name):
        name_str = str(name)
        group_order = 1 if 'Consents' in name_str else (2 if 'GFA' in name_str else (3 if 'Value' in name_str else 4))
        level_order = 0 if 'Total' in name_str else 1
        return (group_order, level_order, name_str)

    global_avg_for_sorting = metrics_df.groupby('Model')[['rMAE', 'sMAPE']].mean().T
    if 'rMAE' in global_avg_for_sorting.index:
        global_model_order = global_avg_for_sorting.loc['rMAE'].sort_values(ascending=False).index
    else:
        global_model_order = global_avg_for_sorting.columns
        
    metrics_list = ['MAE', 'RMSE', 'MSE', 'sMAPE', 'rMAE']

    # --- Feature Level Heatmaps ---
    for metric in metrics_list:
        feature_pivot = metrics_df.pivot(index='Feature', columns='Model', values=metric)
        feature_pivot = feature_pivot.reindex(columns=global_model_order)
        sorted_feature_rows = sorted(feature_pivot.index, key=logical_row_sort)
        feature_pivot = feature_pivot.loc[sorted_feature_rows]

        annot_array_feat = np.empty_like(feature_pivot.values, dtype=object)
        for i, row_name in enumerate(feature_pivot.index):
            for j in range(feature_pivot.shape[1]):
                val = feature_pivot.values[i, j]
                if pd.isna(val):
                    annot_array_feat[i, j] = ""
                    continue
                
                if metric == 'sMAPE':
                    annot_array_feat[i, j] = f"{val:.1f}%"
                elif metric == 'rMAE':
                    annot_array_feat[i, j] = f"{val:.3f}"
                elif metric == 'MSE':
                    annot_array_feat[i, j] = f"{val:.2e}" 
                elif 'Value' in str(row_name):
                    annot_array_feat[i, j] = f"{val/1e6:.2f}e6"
                else:
                    annot_array_feat[i, j] = f"{val:,.0f}"

        feature_pivot.index = [str(idx).replace('/', '/\n').replace(' ', '\n') for idx in feature_pivot.index]
        feature_norm = feature_pivot.div(feature_pivot.max(axis=1), axis=0)

        plt.figure(figsize=(16, 11))
        ax1 = sns.heatmap(feature_norm, annot=annot_array_feat, fmt="", cmap=chosen_cmap,
                          cbar=False, annot_kws={"size": 18, "fontweight": "bold", "color": "white"})

        plt.title(f'Feature Level {metric} Comparison\n(Models sorted Worst to Best by Global rMAE -> Right is Best)', fontsize=19, fontweight='bold')

        clean_labels1 = [col.replace('Optimized', 'Opt.') for col in feature_pivot.columns]
        ax1.set_xticklabels(clean_labels1, rotation=25, ha='right', fontsize=16)
        plt.yticks(rotation=0, fontsize=16)

        for i in range(feature_pivot.shape[0]):
            row_vals = feature_pivot.iloc[i].values
            if pd.isna(row_vals).all(): continue
            best_col_idx = np.nanargmin(row_vals)
            ax1.add_patch(patches.Rectangle(
                (best_col_idx, i), 1, 1, fill=False, hatch='xx', edgecolor='white', lw=1, alpha=0.6
            ))

        plt.tight_layout()
        plt.show()

    # --- Hierarchy Aggregations ---
    hier_sum = metrics_df.groupby(['Model', 'Group', 'Level'])[['MAE', 'MSE']].sum()
    hier_sum['RMSE'] = np.sqrt(hier_sum['MSE']) 
    hier_mean = metrics_df.groupby(['Model', 'Group', 'Level'])[['sMAPE', 'rMAE']].mean()
    heatmap_df = pd.concat([hier_sum, hier_mean], axis=1).reset_index()
    heatmap_df['Hierarchy_Category'] = heatmap_df['Group'] + ' ' + heatmap_df['Level']

    # --- Hierarchy Level Heatmaps ---
    for metric in metrics_list:
        hier_pivot = heatmap_df.pivot(index='Hierarchy_Category', columns='Model', values=metric)
        hier_pivot = hier_pivot.reindex(columns=global_model_order)
        sorted_hier_rows = sorted(hier_pivot.index, key=logical_row_sort)
        hier_pivot = hier_pivot.loc[sorted_hier_rows]

        annot_array_hier = np.empty_like(hier_pivot.values, dtype=object)
        for i, row_name in enumerate(hier_pivot.index):
            for j in range(hier_pivot.shape[1]):
                val = hier_pivot.values[i, j]
                if pd.isna(val):
                    annot_array_hier[i, j] = ""
                    continue
                
                if metric == 'sMAPE':
                    annot_array_hier[i, j] = f"{val:.1f}%"
                elif metric == 'rMAE':
                    annot_array_hier[i, j] = f"{val:.3f}"
                elif metric == 'MSE':
                    annot_array_hier[i, j] = f"{val:.2e}"
                elif 'Value' in str(row_name):
                    annot_array_hier[i, j] = f"{val/1e6:.2f}e6"
                else:
                    annot_array_hier[i, j] = f"{val:,.0f}"

        hier_pivot.index = [str(idx).replace('/', '/\n').replace(' ', '\n') for idx in hier_pivot.index]
        hier_norm = hier_pivot.div(hier_pivot.max(axis=1), axis=0)

        plt.figure(figsize=(16, 7))
        ax2 = sns.heatmap(hier_norm, annot=annot_array_hier, fmt="", cmap=chosen_cmap,
                          cbar=False, annot_kws={"size": 18, "fontweight": "bold", "color": "white"})

        agg_type = "Summed" if metric in ['MAE', 'MSE'] else ("Root Summed" if metric == 'RMSE' else "Averaged")
        plt.title(f'Hierarchy Level {agg_type} {metric} Comparison\n(Models sorted Worst to Best by Global rMAE -> Right is Best)', fontsize=19, fontweight='bold')

        clean_labels2 = [col.replace('Optimized', 'Opt.') for col in hier_pivot.columns]
        ax2.set_xticklabels(clean_labels2, rotation=25, ha='right', fontsize=16)
        plt.yticks(rotation=0, fontsize=16)

        for i in range(hier_pivot.shape[0]):
            row_vals = hier_pivot.iloc[i].values
            if pd.isna(row_vals).all(): continue
            best_col_idx = np.nanargmin(row_vals)
            ax2.add_patch(patches.Rectangle(
                (best_col_idx, i), 1, 1, fill=False, hatch='xx', edgecolor='white', lw=1, alpha=0.6
            ))
        plt.tight_layout()
        plt.show()

    # --- Global Level Heatmap ---
    global_sum = metrics_df.groupby('Model')[['MAE', 'MSE']].sum()
    global_sum['RMSE'] = np.sqrt(global_sum['MSE'])
    global_mean = metrics_df.groupby('Model')[['sMAPE', 'rMAE']].mean()
    global_avg = pd.concat([global_sum, global_mean], axis=1).T
    
    global_avg = global_avg.reindex(columns=global_model_order)
    global_avg = global_avg.loc[metrics_list] 
    global_norm = global_avg.div(global_avg.max(axis=1), axis=0)

    annot_array = np.empty_like(global_avg.values, dtype=object)
    for i in range(global_avg.shape[0]):
        for j in range(global_avg.shape[1]):
            val = global_avg.values[i, j]
            metric = global_avg.index[i]
            if pd.isna(val):
                annot_array[i, j] = ""
                continue
            if metric == 'sMAPE':
                annot_array[i, j] = f"{val:.2f}%"
            elif metric == 'rMAE':
                annot_array[i, j] = f"{val:.3f}"
            else:
                annot_array[i, j] = f"{val:.2e}" 

    plt.figure(figsize=(16, 6))
    ax3 = sns.heatmap(global_norm, annot=annot_array, fmt="", cmap=chosen_cmap,
                      cbar=False, annot_kws={"size": 18, "fontweight": "bold", "color": "white"})

    plt.title('Global Performance Overview (All Metrics)\n(Models sorted Worst to Best by Global rMAE -> Right is Best)', fontsize=19, fontweight='bold')

    clean_labels3 = [col.replace('Optimized', 'Opt.') for col in global_avg.columns]
    ax3.set_xticklabels(clean_labels3, rotation=25, ha='right', fontsize=16)
    plt.yticks(rotation=0, fontsize=16)

    for i in range(global_avg.shape[0]):
        row_vals = global_avg.iloc[i].values
        if pd.isna(row_vals).all(): continue
        best_col_idx = np.nanargmin(row_vals)
        ax3.add_patch(patches.Rectangle(
            (best_col_idx, i), 1, 1, fill=False, hatch='xx', edgecolor='white', lw=1, alpha=0.6
        ))
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------
# 10A. Residual Diagnostics (3-Panel)
# ---------------------------------------------------------
def generate_residual_diagnostics(full_cv_df, metrics_df):
    print("\nGenerating residual diagnostic plots (Residuals, ACF, Bias/Variance)...")
    sns.set_theme(style="whitegrid")

    models = ['LSTM', 'MSTL', 'LSTM_MinT', 'MSTL_LSTM', 'MSTL_MinT', 'MSTL_LSTM_MinT']

    for group in full_cv_df['Group'].unique():
        group_df = full_cv_df[full_cv_df['Group'] == group]

        for uid in group_df['unique_id'].unique():
            feat_df = group_df[group_df['unique_id'] == uid].copy()
            feat_df = feat_df.groupby('ds').mean(numeric_only=True).reset_index().sort_values('ds')

            feature_name_search = f"Total {uid}" if uid in CONFIG['GROUPS'] else uid
            uid_metrics = metrics_df[metrics_df['Feature'] == feature_name_search]
            if not uid_metrics.empty:
                sorted_models = uid_metrics.groupby('Model')['MAE'].mean().sort_values(ascending=False).index.tolist()
                sorted_models = [m for m in sorted_models if m in models]
                for m in models:
                    if m not in sorted_models: sorted_models.append(m)
            else:
                sorted_models = models

            valid_models = [m for m in sorted_models if m in feat_df.columns and not feat_df[m].isna().all()]
            n_cols = max(len(valid_models), 1)

            fig = plt.figure(figsize=(24, 16))
            gs = gridspec.GridSpec(3, n_cols, height_ratios=[2.5, 1.5, 1.0], hspace=0.35)
            fig.suptitle(f"Feature: {uid}", fontsize=28, fontweight='bold', y=0.96)

            # --- 1. Stitched Residuals Over Time ---
            ax1 = fig.add_subplot(gs[0, :])

            for mod in sorted_models:
                if mod in feat_df.columns and not feat_df[mod].isna().all():
                    error = feat_df['y'] - feat_df[mod]
                    lw = 3 if mod == 'MSTL_LSTM_MinT' else 2
                    ls ='--' if mod == 'MSTL_LSTM_MinT' else '-'
                    z_order = 10 if mod == 'MSTL_LSTM_MinT' else 1
                    ax1.plot(feat_df['ds'], error, label=mod, color=GLOBAL_COLORS[mod], linestyle=ls,  linewidth=lw, alpha=0.8, zorder=z_order)

            ax1.axhline(0, color='black', linestyle='-', linewidth=2)
            ax1.set_title("Stitched Out-of-Sample Residuals", fontsize=20, fontweight='bold')
            ax1.set_ylabel("Error", fontweight='bold', fontsize=18)
            ax1.tick_params(axis='both', labelsize=17)
            ax1.legend(loc='upper right', ncol=6, fontsize=20)

            # --- 2. Violin Bias/Variance Check ---
            ax2 = fig.add_subplot(gs[1, :])
            error_data = []
            valid_colors = []

            for mod in sorted_models:
                if mod in feat_df.columns and not feat_df[mod].isna().all():
                    err = feat_df['y'] - feat_df[mod]
                    error_data.append(pd.DataFrame({'Error': err, 'Model': mod}))
                    valid_colors.append(GLOBAL_COLORS[mod])

            if error_data:
                error_df = pd.concat(error_data)
                sns.violinplot(data=error_df, x='Model', y='Error', ax=ax2, palette=valid_colors, inner="quartile")

            ax2.axhline(0, color='black', linestyle='--', linewidth=2)
            ax2.set_title("Residual Distributions", fontsize=20, fontweight='bold')
            ax2.set_ylabel("Error", fontweight='bold', fontsize=18)
            ax2.set_xlabel("") 
            ax2.tick_params(axis='both', labelsize=18)

            # --- 3. ACF Plots ---
            for idx, mod in enumerate(valid_models):
                ax_acf = fig.add_subplot(gs[2, idx])
                error = feat_df['y'] - feat_df[mod]
                plot_acf(error.dropna(), ax=ax_acf, title="", color=GLOBAL_COLORS[mod], vlines_kwargs={"colors": GLOBAL_COLORS[mod]})
                ax_acf.set_title(f"ACF: {mod}", fontsize=20, fontweight='bold')
                ax_acf.set_xlabel("Lags", fontsize=18)
                
                if idx == 0:
                    ax_acf.set_ylabel("Autocorrelation", fontsize=18)
                else:
                    ax_acf.set_ylabel("")
                    
                ax_acf.tick_params(axis='both', labelsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()

# ---------------------------------------------------------
# 10B. Comprehensive Grid Plots (Total Features)
# ---------------------------------------------------------
from matplotlib.patches import Patch

def generate_paper_plots(full_cv_df, metrics_df):
    print("\nGenerating comprehensive dashboard plots for aggregate features...")
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({'font.family': 'Cambria'})

    models = ['LSTM', 'MSTL', 'LSTM_MinT', 'MSTL_LSTM', 'MSTL_MinT', 'MSTL_LSTM_MinT']
    highlight_models = ['MSTL_LSTM', 'MSTL_MinT', 'MSTL_LSTM_MinT']

    TITLE_FS = 28
    LABEL_FS = 24
    TICK_FS = 22
    LEGEND_FS = 22
    ANNOT_FS = 22

    total_ids = CONFIG['GROUPS']

    # --- Pre-process data for improvement percentage visualization ---
    global_plot_df = metrics_df.pivot_table(index=['Group', 'Level', 'Feature'], columns='Model', values='MAE').reset_index()

    global_plot_df['Pct_Decrease_MSTL'] = ((global_plot_df['MSTL'] - global_plot_df['MSTL_LSTM_MinT']) / global_plot_df['MSTL']) * 100
    global_plot_df['Pct_Decrease_LSTM'] = ((global_plot_df['MSTL_LSTM'] - global_plot_df['MSTL_LSTM_MinT']) / global_plot_df['MSTL_LSTM']) * 100
    global_plot_df['Original_Feature'] = global_plot_df['Feature']

    global_plot_df['Feature'] = global_plot_df.apply(
        lambda row: f"Total\n{row['Group']}" if row['Level'] == 'Total' else row['Feature'].replace('/', ' ').replace('_', ' ').replace(' ', '/\n'), axis=1
    )

    group_order = {'Consents': 0, 'GFA': 1, 'Value': 2}
    level_order = {'Total': 0, 'Bottom': 1}
    global_plot_df['Group_Rank'] = global_plot_df['Group'].map(group_order).fillna(99)
    global_plot_df['Level_Rank'] = global_plot_df['Level'].map(level_order).fillna(99)
    global_plot_df = global_plot_df.sort_values(['Group_Rank', 'Level_Rank', 'Feature'], ascending=[True, True, True])
    global_plot_df = global_plot_df.iloc[::-1].reset_index(drop=True)

    for group in full_cv_df['Group'].unique():
        group_df = full_cv_df[full_cv_df['Group'] == group]

        for uid in group_df['unique_id'].unique():
            if uid not in total_ids:
                continue

            feat_df = group_df[group_df['unique_id'] == uid].copy()
            feat_df = feat_df.groupby('ds').mean(numeric_only=True).reset_index().sort_values('ds')

            feature_name = f"Total {uid}"

            uid_metrics = metrics_df[metrics_df['Feature'] == feature_name]
            if not uid_metrics.empty:
                sorted_models = uid_metrics.groupby('Model')['MAE'].mean().sort_values(ascending=False).index.tolist()
                sorted_models = [m for m in sorted_models if m in models]
                for m in models:
                    if m not in sorted_models: sorted_models.append(m)
            else:
                sorted_models = models

            fig = plt.figure(figsize=(28, 13))
            gs = gridspec.GridSpec(2, 2, width_ratios=[1, 3.5], height_ratios=[1, 1.2], wspace=0.15, hspace=0.35)

            if 'GFA' in uid:
                unit = '(m²)'
            elif 'Value' in uid:
                unit = '($)'
            else:
                unit = '(count)'

            # --- Panel A: Error Reduction Comparison ---
            ax_left = fig.add_subplot(gs[:, 0])

            colors_mstl = ['#f2b468'] * len(global_plot_df)  
            colors_lstm = ['#faddb9'] * len(global_plot_df)  

            y_pos = np.arange(len(global_plot_df))
            bar_height = 0.38 

            bars1 = ax_left.barh(y_pos + bar_height/2, global_plot_df['Pct_Decrease_MSTL'], color=colors_mstl, edgecolor='dimgrey', height=bar_height)
            bars2 = ax_left.barh(y_pos - bar_height/2, global_plot_df['Pct_Decrease_LSTM'], color=colors_lstm, edgecolor='dimgrey', height=bar_height)

            ax_left.set_yticks(y_pos)
            ax_left.set_yticklabels(global_plot_df['Feature'])

            ax_left.set_title('a) MSTL_LSTM_MinT Improvements', fontsize=TITLE_FS, fontweight='bold', pad=15, loc='left')
            ax_left.set_xlabel('MAE Decrease (%)', fontsize=28, fontweight='bold')
            ax_left.axvline(0, color='black', linewidth=1.5)
            ax_left.grid(axis='x', linestyle='--', alpha=0.5)
            ax_left.tick_params(axis='x', labelsize=TICK_FS, bottom=True)

            max_abs_val = max(
                abs(global_plot_df['Pct_Decrease_MSTL']).max(),
                abs(global_plot_df['Pct_Decrease_LSTM']).max()
            )
            max_abs_val = max_abs_val if max_abs_val > 0 else 1.0

            def annotate_bars(bars, orig_features, levels):
                for bar, orig_feat, lvl in zip(bars, orig_features, levels):
                    width = bar.get_width()
                    y_c = bar.get_y() + bar.get_height() / 2

                    if orig_feat == feature_name:
                        bar.set_edgecolor('black')
                        bar.set_linewidth(2.5)
                    elif lvl == 'Total':
                        bar.set_linewidth(1.5)

                    ha = 'right' if width >= 0 else 'left'
                    offset = -max_abs_val * 0.02 if width >= 0 else max_abs_val * 0.02
                    weight = 'heavy' if lvl == 'Total' else 'bold'

                    ax_left.text(width + offset, y_c, f'{width:.1f}%',
                            va='center', ha=ha, color='black', fontsize=ANNOT_FS, fontweight=weight)

            annotate_bars(bars1, global_plot_df['Original_Feature'], global_plot_df['Level'])
            annotate_bars(bars2, global_plot_df['Original_Feature'], global_plot_df['Level'])

            for tick_label, level in zip(ax_left.get_yticklabels(), global_plot_df['Level']):
                tick_label.set_fontsize(24)
                if level == 'Total':
                    tick_label.set_fontweight('bold')

            leg1 = Patch(facecolor='#f2b468', edgecolor='dimgrey', label='vs MSTL')
            leg2 = Patch(facecolor='#faddb9', edgecolor='dimgrey', label='vs MSTL_LSTM')
            ax_left.legend(handles=[leg1, leg2], loc='lower right', fontsize=LEGEND_FS, frameon=True, framealpha=0.9)

            ax_left.spines['top'].set_visible(False)
            ax_left.spines['right'].set_visible(False)
            ax_left.spines['left'].set_visible(False)
            ax_left.tick_params(axis='y', length=0)

            # --- Panel B: Prediction Trajectories ---
            ax1 = fig.add_subplot(gs[0, 1])

            # Plot observed values
            ax1.plot(feat_df['ds'], feat_df['y'], label='Observed', color='black', linestyle='--', linewidth=4.0, alpha=0.6, zorder=1)

            other_models_added = False
            for mod in sorted_models:
                if mod in feat_df.columns and not feat_df[mod].isna().all() and mod not in highlight_models:
                    lbl = 'Other Baselines' if not other_models_added else "_nolegend_"
                    ax1.plot(feat_df['ds'], feat_df[mod], label=lbl, color='grey', linewidth=1.2, alpha=0.6, zorder=3)
                    other_models_added = True

            for mod in sorted_models:
                if mod in feat_df.columns and not feat_df[mod].isna().all() and mod in highlight_models:
                    lw = 3.5 if mod == 'MSTL_LSTM_MinT' else 2
                    z_order = 10 if mod == 'MSTL_LSTM_MinT' else 7 
                    ax1.plot(feat_df['ds'], feat_df[mod], label=mod, color=GLOBAL_COLORS[mod], linewidth=lw, alpha=0.9, zorder=z_order)

            ax1.set_title(f"b) Observed and Predicted values: Total {uid}", fontsize=TITLE_FS, fontweight='bold', pad=15, loc='left')
            ax1.set_ylabel(f"Value {unit}", fontweight='bold', fontsize=LABEL_FS)
            ax1.tick_params(axis='both', which='major', labelsize=TICK_FS)
            
            # Apply scientific notation to y-axis
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
            ax1.yaxis.get_offset_text().set_fontsize(TICK_FS)

            handles, labels = ax1.get_legend_handles_labels()
            line_leg = ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False, prop={'weight': 'bold', 'size': LEGEND_FS})
            
            # Adjust legend line widths for visibility
            for line in line_leg.get_lines():
                line.set_linewidth(5.0)

            # --- Panel C: Residual Distributions ---
            ax2 = fig.add_subplot(gs[1, 1])
            error_data = []
            valid_colors = []

            for mod in sorted_models:
                if mod in feat_df.columns and not feat_df[mod].isna().all():
                    err = feat_df['y'] - feat_df[mod]
                    error_data.append(pd.DataFrame({'Error': err, 'Model': mod}))
                    valid_colors.append(GLOBAL_COLORS[mod])

            if error_data:
                error_df = pd.concat(error_data)
                sns.violinplot(data=error_df, x='Model', y='Error', ax=ax2, palette=valid_colors, inner="quartile")

            ax2.axhline(0, color='black', linestyle='--', linewidth=2.5)
            ax2.set_title(f"c) Residual Distributions: Total {uid}", fontsize=TITLE_FS, fontweight='bold', pad=15, loc='left')
            ax2.set_ylabel(f"Residuals {unit}", fontweight='bold', fontsize=LABEL_FS)

            ax2.set_xlabel("")
            ax2.tick_params(axis='x', rotation=0, labelsize=TICK_FS)
            ax2.tick_params(axis='y', labelsize=TICK_FS)

            fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.08)
            plt.show()

# ---------------------------------------------------------
# 11. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    all_metrics_dfs = []
    all_reconciled_cvs = []

    for group in CONFIG['GROUPS']:
        print(f"\nProcessing structural group: {group}")
        try:
            hier_df, S_df, tags = load_and_filter_hierarchy(CONFIG['FILENAME'], group)

            cv_results, fitted_results = run_cached_cv(hier_df, CONFIG['HORIZON'], CONFIG['FOLDS'], CONFIG['STEP_SIZE'])
            cv_with_mstl_lstm = apply_mstl_lstm_residual_learning(cv_results, fitted_results, CONFIG['HORIZON'], tags, group)
            cv_with_all_models = apply_vanilla_lstm_learning(cv_with_mstl_lstm, fitted_results, CONFIG['HORIZON'], tags, group)

            print(f"Applying EMinT Hierarchical Reconciliation for {group}...")
            grand_reconciled_cv = apply_grand_reconciliation(cv_with_all_models, S_df, tags)

            grand_reconciled_cv['Group'] = group
            all_reconciled_cvs.append(grand_reconciled_cv)

            group_metrics = calculate_metrics(grand_reconciled_cv, tags)
            group_metrics['Group'] = group
            all_metrics_dfs.append(group_metrics)

        except Exception as e:
            print(f"Error processing group {group}: {e}")

    metrics_df = pd.concat(all_metrics_dfs, ignore_index=True)
    full_cv_df = pd.concat(all_reconciled_cvs, ignore_index=True)

    print("\n" + "="*145)
    print("Global Evaluation Metrics (Sorted by MAE)")
    print("="*145)

    print(f"{'GROUP':<10} | {'LEVEL':<8} | {'FEATURE':<25} | {'MODEL':<20} | {'MAE':>10} | {'RMSE':>10} | {'MSE':>12} | {'sMAPE':>10} | {'rMAE':>10}")
    print("-" * 145)

    sorted_df = metrics_df.sort_values(['Group', 'Level', 'Feature', 'MAE'], ascending=[True, False, True, True])
    for _, row in sorted_df.iterrows():
        print(f"{row['Group']:<10} | {row['Level']:<8} | {row['Feature']:<25} | {row['Model']:<20} | {row['MAE']:>10.2f} | {row['RMSE']:>10.2f} | {row['MSE']:>12.2f} | {row['sMAPE']:>9.2f}% | {row['rMAE']:>10.4f}")
# Add this line around line 430 in Script4:
    full_cv_df.to_csv('full_cv_reconciled.csv', index=False)
    excel_export_path = 'Evaluation_Merged_LSTM_EMinT_Final.xlsx'
    print(f"\nSaving metrics to file: '{excel_export_path}'...")

    try:
        with pd.ExcelWriter(excel_export_path, engine='openpyxl') as writer:
            sorted_df.to_excel(writer, sheet_name='1_Feature_Level', index=False)

            hierarchy_rows = []
            for group in CONFIG['GROUPS']:
                for lvl in ['Total', 'Bottom']:
                    sub = metrics_df[(metrics_df['Group'] == group) & (metrics_df['Level'] == lvl)]
                    if not sub.empty:
                        summary_mean = sub.groupby('Model')[['sMAPE', 'rMAE']].mean()
                        summary_sum = sub.groupby('Model')[['MAE', 'MSE']].sum()
                        summary_sum['RMSE'] = np.sqrt(summary_sum['MSE'])
                        summary = pd.concat([summary_sum[['MAE', 'RMSE']], summary_mean], axis=1).reset_index()
                        summary.insert(0, 'Level', lvl)
                        summary.insert(0, 'Group', group)
                        hierarchy_rows.append(summary)
            if hierarchy_rows:
                hierarchy_df = pd.concat(hierarchy_rows, ignore_index=True)
                hierarchy_df = hierarchy_df.sort_values(['Group', 'Level', 'rMAE'], ascending=[True, False, True])
                hierarchy_df.to_excel(writer, sheet_name='2_Hierarchy_Averages', index=False)

            global_rows = []
            for lvl in ['Total', 'Bottom']:
                global_summary = metrics_df[metrics_df['Level'] == lvl].groupby('Model')[['rMAE', 'sMAPE']].mean().reset_index()
                global_summary.insert(0, 'Level', lvl)
                global_rows.append(global_summary)
            if global_rows:
                global_df = pd.concat(global_rows, ignore_index=True)
                global_df = global_df.sort_values(['Level', 'rMAE'], ascending=[False, True])
                global_df.to_excel(writer, sheet_name='3_Global_Averages', index=False)

        print("Export completed successfully.")
    except Exception as e:
        print(f"Failed to save Excel file. Encountered error: {e}")

    generate_plots(metrics_df, chosen_cmap="copper")
    generate_residual_diagnostics(full_cv_df, metrics_df)
    generate_paper_plots(full_cv_df, metrics_df)
