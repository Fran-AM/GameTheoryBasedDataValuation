from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.utils import Utility

from sklearn.linear_model import SGDClassifier

from utils.utils import setup_logger, compute_values
from dataset import get_openML_data

from sklearn.metrics import f1_score

logger = setup_logger()

def run():
    logger.info("Starting weighted accurary experiment")
    # params = params_show() Este para cuando esté en el dvc
    params = {
        'mislabel_detection': {
            'datasets': [ 'click','phoneme', 'wind', 'cpu', '2dplanes'],
            'hidden_neurons': 100,
            'activation_function': 'relu',
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'batch_size': 32,
            'data_points': 200,
            'flip_ratio': 0.1,
            'max_iter': 100,
            'methods': ["LOO", "Banzhaf", "Shapley", "Beta-1-16", "Beta-1-4", "Beta-16-1", "Beta-4-1"],
            'n_repeat': 1
        },
        'weighted_acc': {
            'datasets': ['click','phoneme', 'wind', 'cpu', '2dplanes'],
            'loss': 'log_loss',
            'data_points': 200,
            'methods': ["LOO", "Banzhaf", "Shapley", "Beta-1-16", "Beta-1-4", "Beta-16-1", "Beta-4-1"],
            'n_repeat': 1
        }
    }

    # Params used in the experiment
    wacc_params =  params["weighted_acc"]
    n_repeat = wacc_params["n_repeat"]
    
    experiment_output_dir = (
        Path(Repo.find_root())
        / "output"
        / "weighted_acc"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    model = SGDClassifier(
        loss = wacc_params["loss"]
    )

    all_results = {}

    for dataset_name in wacc_params["datasets"]:
        logger.info(f"{dataset_name=}")

        dataset = get_openML_data(
            dataset=dataset_name,
            n_data=wacc_params["data_points"],
        )
        x_train, y_train = dataset.get_training_data()
        x_test, y_test = dataset.get_test_data()

        utility = Utility(model, dataset, 'accuracy')

        dataset_results = {}

        for method_name in wacc_params["methods"]:
            logger.info(f"{method_name=}")
            method_results = {}
            # Compute the values and f1 as many times
            # as specified in n_repeat
            for repetition in range(n_repeat):
                logger.info(f"{repetition=}")
                logger.info("Computing values")
                values = compute_values(
                    method_name,
                    utility=utility
                )
                # Train the weighted model
                val = values.values
                if np.min(val)==np.max(val):
                    norm_values = np.ones(len(val))
                else:
                    norm_values = (val - np.min(val))/(np.max(val)-np.min(val))

                model.fit(x_train, y_train, sample_weight=norm_values)
                y_weight = model.predict(x_test)
                w_acc = f1_score(y_test, y_weight)
                method_results[repetition] = float(w_acc)

            dataset_results[method_name] = method_results
        # Train the non weighted model
        model.fit(x_train, y_train)
        y_unif = model.predict(x_test)
        u_acc = f1_score(y_test, y_unif)
        dataset_results["uniform"] = float(u_acc)

        all_results[dataset_name] = dataset_results

    # Save results to yaml file
    output_file = experiment_output_dir / "results.yaml"
    with open(output_file, 'w') as file:
        yaml.dump(all_results, file, default_flow_style=False)
        logger.info(f"Saved results to {output_file}")

    logger.info("Finished weighted accuracy experiment")

if __name__ == "__main__":
    run()