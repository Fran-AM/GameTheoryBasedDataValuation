from pathlib import Path

import pandas as pd
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
            'datasets': ['phoneme','wind'],
            'loss': 'log_loss',
            'data_points': 64,
            'test_points': 64,
            'methods': ["LOO"]
        }
    }

    # Params used in the experiment
    wacc_params =  params["weighted_acc"]
    n_repeat = wacc_params["n_repeat"]
    

    for dataset_name in wacc_params["datasets"]:
        # Create the output directory
        experiment_output_dir = (
            Path(Repo.find_root())
            / "output"
            / "weighted_acc"
            / f"dataset={dataset_name}"
        )
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        dataset = get_openML_data(
            dataset=dataset_name,
            n_data=wacc_params["data_points"],
            n_test=wacc_params["test_points"]
        )
        x_train, y_train = dataset.get_training_data()
        x_test, y_test = dataset.get_test_data()

        # Score dictionary creation
        scores = {method: [] for method in wacc_params["methods"]}
        scores["uniform"] = []

        for repetition in range(n_repeat):
            logger.info(f"{repetition=}")

            # Model creation
            model = SGDClassifier(
                loss = wacc_params["loss"]
                )
            
            # Utility creation
            utility = Utility(model, dataset, 'accuracy')

            # Compute the values
            for method_name in wacc_params["methods"]:
                logger.info(f"{method_name=}")
                logger.info("Computing values")
                values = compute_values(
                    method_name,
                    utility=utility
                )
                # Train the weighted model
                model.fit(x_train, y_train, values.values)
                y_weight = model.predict(x_test)
                w_acc = f1_score(y_test, y_weight, average='micro')
                scores[method_name].append(w_acc)
            
            # Train the normal model
            model.fit(x_train, y_train)
            y_unif = model.predict(x_test)
            u_acc = f1_score(y_test, y_unif, average='micro')
            scores["uniform"].append(u_acc)

        # Convert scores dictionary to DataFrame
        scores_df = pd.DataFrame(scores)
        # Save results to disk
        output_file = experiment_output_dir / "scores.csv"
        try:
            scores_df.to_csv(output_file, index=False)
            logger.info(f"Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to disk: {e}")

    logger.info("Finished weighted accuracy experiment")

if __name__ == "__main__":
    run()