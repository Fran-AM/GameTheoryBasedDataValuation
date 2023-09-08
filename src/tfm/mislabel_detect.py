from pathlib import Path

import yaml
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.utils import Utility

from sklearn.neural_network import MLPClassifier

from utils.utils import setup_logger, compute_values, f1_misslabel
from dataset import get_openML_data

logger = setup_logger()

def run():
    logger.info("Starting mislabel detection experiment")
    # params = params_show() Este para cuando esté en el dvc
    params = {
        'mislabel_detection': {
            'datasets': ['click','phoneme', 'wind', 'cpu', '2dplanes'],
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
            'methods': ["LOO"],
            'n_repeat': 1
        }
    }

    # Params used in the experiment
    md_params =  params["mislabel_detection"]
    n_repeat = md_params["n_repeat"]

    experiment_output_dir = (
        Path(Repo.find_root())
        / "output"
        / "mislabel_detection"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    model = MLPClassifier(
        hidden_layer_sizes = (md_params["hidden_neurons"],),
        learning_rate_init = md_params["learning_rate"],
        batch_size = md_params["batch_size"],
        max_iter = md_params["max_iter"],
        )

    all_results = {}

    for dataset_name in md_params["datasets"]:
        logger.info(f"{dataset_name=}")
        dataset_result = {}

        dataset = get_openML_data(
            dataset=dataset_name,
            n_data=md_params["data_points"],
            flip_ratio=0.1
        )

        utility = Utility(model, dataset, 'accuracy')

        for method_name in md_params["methods"]:
            logger.info(f"{method_name=}")
            method_results = {}
            # Compute the values and f1 as many times
            # as specified in n_repeat
            for repetition in range(n_repeat):
                logger.info(f"{repetition=}")
                logger.info("Computing values")
                values = compute_values(
                    method_name, utility=utility
                )
                logger.info("Computing f1 scores")
                method_results[repetition] = f1_misslabel(values.values)
            
            dataset_result[method_name] = method_results

        all_results[dataset_name] = dataset_result

    # Save results to yaml file
    output_file = experiment_output_dir / "results.yaml"
    with open(output_file, 'w') as file:
        yaml.dump(all_results, file, default_flow_style=False)
        logger.info(f"Saved results to {output_file}")

    logger.info("Finished mislabel detection experiment")

if __name__ == "__main__":
    run()