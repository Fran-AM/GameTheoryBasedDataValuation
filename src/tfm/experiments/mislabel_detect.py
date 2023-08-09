from pathlib import Path

import pandas as pd
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility

from tfm.utils import setup_logger

logger = setup_logger()

def run(dataset_name: str, budget: int):
    # TODO: Tenemos que añadir todos los parametros
    # necesarios al params.yaml
    logger.info("Starting mislabel detection experiment")
    params = params_show()
    logger.info(f"Using parameters:\n{params}")

    # Params used in the experiment
    mislabel_detect_params =  params["mislabel_detect"]
    n_repeat = mislabel_detect_params["n_repeat"]
    
    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root())
        / "output"
        / "mislabel_detection"
        / f"dataset={dataset_name}"
        / "results"
        / f"{budget=}"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    for repetition in range(n_repeat):
        logger.info(f"{repetition=}")

        repetition_output_dir = experiment_output_dir / f"{repetition=}"
        repetition_output_dir.mkdir(parents=True, exist_ok=True)

        all_values = []
        all_scores = []

        # Hay que construir las utilidades (Dataset, Modelo, Métrica)

            # Cogemos cada uno de los dataset (Los tenemos en los parámetros)
            
            # De



    logger.info("Finished mislabel detection experiment")


if __name__ == "__main__":
    run()
