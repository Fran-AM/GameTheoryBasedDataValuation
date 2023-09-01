from pathlib import Path

import pandas as pd
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.utils import Utility

from sklearn.neural_network import MLPClassifier

from utils.utils import setup_logger, convert_values_to_dataframe, compute_values, f1_misslabel
from dataset import get_openML_data
logger = setup_logger()

def run():
    logger.info("Starting mislabel detection experiment")
    # params = params_show() Este para cuando esté en el dvc
    params = {
        'mislabel_detection': {
            'datasets': ['phoneme','wind'],
            'hidden_neurons': 10,
            'activation_function': 'relu',
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'batch_size': 32,
            'data_points': 64,
            'test_points': 64,
            'flip_ratio': 0.1,
            'max_iter': 100,
            'methods': ["LOO"],
            'n_repeat': 2
        },
        'weighted_acc': {
            'model': 'LinearRegression'
        }
    }

    # Params used in the experiment
    md_params =  params["mislabel_detection"]
    n_repeat = md_params["n_repeat"]
    

    for dataset_name in md_params["datasets"]:
        # Create the output directory
        experiment_output_dir = (
            Path(Repo.find_root())
            / "output"
            / "mislabel_detection"
            / f"dataset={dataset_name}"
        )
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = get_openML_data(
            dataset=dataset_name,
            n_data=md_params["data_points"],
            n_test=md_params["data_points"],
            flip_ratio=0.1
        )

        for repetition in range(n_repeat):
            logger.info(f"Iteracion {repetition}")

            # Score dictionary creation
            scores = {method: [] for method in params["mislabel_detection"]["methods"]}

            repetition_output_dir = experiment_output_dir / f"{repetition}"
            repetition_output_dir.mkdir(parents=True, exist_ok=True)

            # Model creation
            model = MLPClassifier(
                hidden_layer_sizes = (md_params["hidden_neurons"],),
                learning_rate_init = md_params["learning_rate"],
                batch_size = md_params["batch_size"],
                max_iter = md_params["max_iter"],
                )
            
            # Utility creation
            utility = Utility(model, dataset, 'f1')

            # Compute the values and f1 scores for each method
            for method_name in md_params["methods"]:
                logger.info(f"{method_name=}")
                logger.info("Computing values")
                values = compute_values(
                    method_name, utility=utility
                )
                logger.info("Converting values to DataFrame")
                df = (
                    values.to_dataframe(column=method_name)
                    .drop(columns=[f"{method_name}_stderr"])
                    .T
                )
                logger.info("Computing f1 scores")
                # All score cambia en cada repeticion
                # Pero no en cada método
                scores[method_name].append(f1_misslabel(df.values[0]))
            
            logger.info("Saving results to disk")
            scores_df = pd.DataFrame(scores)
            # Are methods identified?
            scores_df.to_csv(repetition_output_dir / "scores.csv", index=False)
            try:
                scores_df.to_csv(repetition_output_dir / "scores.csv", index=False)
            except Exception as e:
                logger.error(f"Failed to save results to disk: {e}")

    logger.info("Finished mislabel detection experiment")


if __name__ == "__main__":
    run()