import os
import warnings
import numpy as np
import optuna
import time

from optuna.storages import RetryFailedTrialCallback
from optuna.pruners import ThresholdPruner
from optuna.samplers import TPESampler

import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"/")
from modules.models import ModelSNN
from modules.other.utils import read_data

BASE_SEED = 42

warnings.filterwarnings("ignore")
np.random.seed(BASE_SEED)
PATH = os.path.dirname(os.path.realpath(__file__))
print("PATH",PATH)

def optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters, layers):
    betas = tuple(
        trial.suggest_float(f'beta{i+1}', 0.1, 0.95, log=True) for i in range(layers)
    ) if hyperparameters['beta'] is None else hyperparameters['beta']
    slope = trial.suggest_int('slope', 10, 50, step=1) if hyperparameters['slope'] is None else hyperparameters['slope']
    thresholds=tuple(
        trial.suggest_float(f'threshold{i+1}', 0.1, 1, log=True) for i in range(layers)
    ) if hyperparameters['threshold'] is None else hyperparameters['threshold']
    weight_minority_class = trial.suggest_float('weight', 0.95, 1, log=True) if hyperparameters['weight'] is None else hyperparameters['weight']
    class_weights = (1-weight_minority_class, weight_minority_class) 
    adam_betas = tuple( 
        trial.suggest_float(f'adam_beta{i+1}', 0.97, 0.99, step=0.001) for i in range(2)
    ) if hyperparameters['adam_beta'] is None else hyperparameters['adam_beta']
    learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True) if hyperparameters['learning_rate'] is None else hyperparameters['learning_rate']
    train_df = train_dfs[dataset_name].iloc[:, :32]
    test_df = test_dfs[dataset_name].iloc[:, :32]
    x_train = train_df.drop(columns=["fraud_bool"])
    y_train = train_df["fraud_bool"]
    x_test = test_df.drop(columns=["fraud_bool"])
    y_test = test_df["fraud_bool"]
    num_classes = len(np.unique(y_train))
    num_features = len(x_train.columns)
    model = ModelSNN(
        num_features=num_features,
        num_classes=num_classes,
        architecture=hyperparameters['architecture'],
        class_weights=class_weights,
        betas=betas,
        slope=slope,
        thresholds=thresholds,
        batch_size=hyperparameters['batch'],
        num_epochs=hyperparameters['epoch'],
        num_steps=hyperparameters['step'],
        adam_betas=adam_betas,
        learning_rate=learning_rate,
        gpu_number=int(trial.number)%3,
        verbose=0
    )
    fit_time = time.time()
    model.fit(x_train, y_train)
    trial.set_user_attr("@time train", time.time()-fit_time)
    inference_time = time.time()
    predictions, targets = model.predict(x_test, y_test)
    trial.set_user_attr("@time inference", time.time()-inference_time)
    eta = (TRIALS_OPTUNA*(time.time()-fit_time)-trial.number*(time.time()-fit_time))/3600
    print(f"ETA: {eta/24:.0f}d {eta%24:.0f}h {eta%1*60:.0f}m")
    metrics = model.evaluate(targets, predictions)
    aequitas_results = model.evaluate_aequitas(x_test, targets, predictions)
    metrics.update(aequitas_results)
    print(f'Trial {trial.number}: Recall–{metrics["recall"]*100:.1f}% FPR–{metrics["fpr"]*100:.1f}% FPRRatio–{metrics["fpr_ratio"]*100:.1f}%') if (metrics["recall"]>0.4 and metrics["fpr"]<0.1) else None
    trial.set_user_attr("@global accuracy", metrics["accuracy"])
    trial.set_user_attr("@global precision", metrics["precision"])
    trial.set_user_attr("@global recall", metrics["recall"])
    trial.set_user_attr("@global fpr", metrics["fpr"])
    trial.set_user_attr("@global f1_score", metrics["f1_score"])
    trial.set_user_attr("@global auc", metrics["auc"])
    try:
        trial.set_user_attr("@5FPR fpr", metrics["fpr@5FPR"])
        trial.set_user_attr("@5FPR recall", metrics["recall@5FPR"])
        trial.set_user_attr("@5FPR accuracy", metrics["accuracy@5FPR"])
        trial.set_user_attr("@5FPR precision", metrics["precision@5FPR"])
        trial.set_user_attr("@5FPR fpr_ratio", metrics["fpr_ratio"])
        trial.set_user_attr("@5FPR threshold", metrics["threshold"])
    except Exception:
        pass
    objectives = [metrics[y] for (_,y) in OBJECTIVE]
    return objectives

def main(datasets_list, study_name, trials_optuna, sampler, objective, hyperparameters):
    base_path = f"{PATH}/../data/"
    _, datasets, train_dfs, test_dfs = read_data(base_path, datasets_list, seed=BASE_SEED)
    for dataset_name in datasets.keys(): 
        storage = optuna.storages.RDBStorage(
            url="sqlite:///epia2024.db",
            heartbeat_interval=60,
            grace_period=120,
            failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
        ) 
        study = optuna.create_study(
            directions=[x for (x,_) in objective],
            storage=storage,
            load_if_exists=True,
            study_name=f"{study_name}",
            sampler=sampler,
            pruner=ThresholdPruner(lower=0.01, upper=0.99)
        )
        layers = 3 if hyperparameters["architecture"] == "Net1_CSNN" else 4
        study.optimize(lambda trial, dataset_name=dataset_name, layers=layers: optimize_parameters(trial, dataset_name, train_dfs, test_dfs, hyperparameters, layers), n_trials=trials_optuna)
        try:
            print(study.best_params)
            print(study.best_value)
            print(study.best_trial)
        except Exception:
            pass



if __name__ == "__main__":
    HYPERPARAMETERS = {
        "datasets": ["Base"],
        "architecture": "Net2_CSNN",
        "batch": 1024,
        "epoch": 10,
        "step": 10,
        "beta": None,
        "slope": None,
        "threshold": None,
        "weight": None,
        "adam_beta": None,
        "learning_rate": None
    }
    DATASETS = ["Base"]
    STUDY_NAME = "test"
    TRIALS_OPTUNA = 1000
    SAMPLER = TPESampler()
    OBJECTIVE = [("minimize","fpr"), ("maximize","recall")]
main(DATASETS, STUDY_NAME, TRIALS_OPTUNA, SAMPLER, OBJECTIVE, HYPERPARAMETERS)
