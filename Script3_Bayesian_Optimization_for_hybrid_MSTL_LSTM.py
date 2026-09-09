import os
import copy
import warnings
import logging
import numpy as np
import pandas as pd
import optuna
from joblib import Memory
from optuna.samplers import TPESampler
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
from statsforecast.models import MSTL
from hierarchicalforecast.utils import aggregate
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

# ---------------------------------------------------------
# 2. Custom Validation Callback
# ---------------------------------------------------------
class InMemoryRestoreBestWeights(Callback):
    """
    Monitors validation loss during training and retains the model weights 
    from the best epoch in memory to prevent overfitting during hyperparameter sweeps.
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
# 3. Data Processing & Base Model Cross-Validation
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
    
    return hier_df.reset_index(), tags

@memory.cache
def run_cached_cv(df, horizon, n_windows, step_size):
    """
    Computes expanding-window cross-validation for the MSTL baseline model.
    """
    models = [MSTL(season_length=12)]
    sf = StatsForecast(models=models, freq='MS', n_jobs=-1)
    cv_df = sf.cross_validation(df=df, h=horizon, step_size=step_size, n_windows=n_windows, fitted=True)
    fitted_df = sf.cross_validation_fitted_values()
    return cv_df, fitted_df

# ---------------------------------------------------------
# 4. Bayesian Optimization Objective Function
# ---------------------------------------------------------
def objective(trial, cv_data_dict, horizon):
    """
    Defines the hyperparameter search space and evaluates the geometric mean 
    of the improvement ratio (Hybrid MAE / Baseline MAE) across targeted features.
    """
    input_size = trial.suggest_int('input_size', 1, 36)
    encoder_hidden_size = trial.suggest_int('encoder_hidden_size', 4, 128)
    decoder_hidden_size = trial.suggest_int('decoder_hidden_size', 4, 128)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.01, log=True)
    encoder_dropout = trial.suggest_float('encoder_dropout', 0.0, 0.3)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'robust', 'minmax'])
    windows_batch_size = trial.suggest_categorical('windows_batch_size', [128, 2048])
    
    patience_epochs = 10 
    all_ratios = []

    try:
        for group in CONFIG['GROUPS']:
            target_subs = TARGET_FEATURES.get(group, [])
            if not target_subs: 
                continue

            cv_df = cv_data_dict[group]['cv']
            fitted_df = cv_data_dict[group]['fitted']
            bottom_ids = cv_data_dict[group]['bottom_ids']
            total_ids = cv_data_dict[group]['total_ids']

            target_bottom_ids = [uid for uid in bottom_ids if any(sub in uid for sub in target_subs)]
            target_all_ids = target_bottom_ids + total_ids

            cutoffs = sorted(cv_df['cutoff'].unique())
            fold_col = 'cutoff' if 'cutoff' in fitted_df.columns else 'fold'
            folds = sorted(fitted_df[fold_col].unique())

            for i, cutoff in enumerate(cutoffs):
                fold_id = folds[i]

                train_fold = fitted_df[fitted_df[fold_col] == fold_id].copy()
                target_train = train_fold[train_fold['unique_id'].isin(target_all_ids)].copy()
                if target_train.empty: 
                    continue

                # Isolate the stationary residuals of the baseline model
                target_train['y_res'] = target_train['y'] - target_train['MSTL']
                core_train = target_train[['unique_id', 'ds', 'y_res']].rename(columns={'y_res': 'y'})

                # Dynamically calculate steps per epoch based on batch size and valid temporal windows
                total_datapoints = len(core_train)
                num_features = core_train['unique_id'].nunique()
                total_windows = total_datapoints - (num_features * (input_size + horizon - 1))
                total_windows = max(1, total_windows) 
                
                steps_per_epoch = int(np.ceil(total_windows / windows_batch_size))

                dl_models = [
                    LSTM(
                        h=horizon,
                        input_size=input_size,
                        encoder_hidden_size=encoder_hidden_size,
                        decoder_hidden_size=decoder_hidden_size,
                        learning_rate=learning_rate,
                        encoder_dropout=encoder_dropout,
                        scaler_type=scaler_type,
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

                mask = cv_df['cutoff'] == cutoff
                fold_test = cv_df[mask].copy()
                fold_test = fold_test[fold_test['unique_id'].isin(target_all_ids)]

                merged = fold_test.merge(pred_dl, on=['unique_id', 'ds'], how='left')
                merged['LSTM'] = merged['LSTM'].fillna(0)
                
                # Combine base predictions with deep learning residual corrections
                merged['MSTL_lstm'] = merged['MSTL'] + merged['LSTM']

                # Evaluate ratio improvement per feature
                for uid in target_all_ids:
                    uid_df = merged[merged['unique_id'] == uid]
                    if uid_df.empty: 
                        continue

                    y_true = uid_df['y'].values
                    y_mstl = uid_df['MSTL'].values
                    y_mstl_lstm = uid_df['MSTL_lstm'].values

                    mae_mstl = np.mean(np.abs(y_true - y_mstl)) + 1e-9
                    mae_lstm = np.mean(np.abs(y_true - y_mstl_lstm)) + 1e-9

                    ratio = mae_lstm / mae_mstl
                    all_ratios.append(ratio)

    except Exception as e:
        print(f"Trial failed internally: {e}")
        return float('inf')

    # Return the geometric mean of the scale-independent ratios
    return np.exp(np.mean(np.log(np.array(all_ratios))))

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Pre-computing Baseline MSTL Cross-Validation...")
    cv_data_dict = {}

    for group in CONFIG['GROUPS']:
        print(f"Processing structural group: {group}")
        hier_df, tags = load_and_filter_hierarchy(CONFIG['FILENAME'], group)
        total_ids = list(tags['Type'])
        bottom_ids = [uid for uid in hier_df['unique_id'].unique() if uid not in total_ids]

        cv_results, fitted_results = run_cached_cv(hier_df, CONFIG['HORIZON'], CONFIG['FOLDS'], CONFIG['STEP_SIZE'])

        cv_data_dict[group] = {
            'cv': cv_results,
            'fitted': fitted_results,
            'bottom_ids': bottom_ids,
            'total_ids': total_ids
        }

    print("-" * 60)
    print("Executing Optuna Bayesian Optimization (1000 Trials)")
    print("Objective: Minimize Geometric Mean Ratio (Hybrid vs Baseline)")
    print("-" * 60)

    def print_best_callback(study, trial):
        """Outputs updated hyperparameter configurations when a superior model is found."""
        try:
            if study.best_trial.number == trial.number:
                print(f"\nNew optimal parameters found at Trial {trial.number} (Geometric Mean Ratio: {trial.value:.4f})")
                for key, val in trial.params.items():
                    print(f"  {key}: {val}")
                print("-" * 50)
        except ValueError:
            pass

    study = optuna.create_study(
        direction='minimize',
        sampler=TPESampler(n_startup_trials=100, seed=42)
    )

    study.optimize(
        lambda trial: objective(trial, cv_data_dict, CONFIG['HORIZON']),
        n_trials=1000,
        show_progress_bar=True,
        callbacks=[print_best_callback]
    )

    print("\n" + "="*50)
    print("Optimization Complete.")
    best_score = study.best_value
    print(f"Final Global Geometric Improvement Ratio: {best_score:.4f}")
    print("Optimal Hyperparameters:")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")
    print("="*50)
