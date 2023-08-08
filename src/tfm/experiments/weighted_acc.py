from pathlib import Path

import pandas as pd
from dvc.api import params_show
from dvc.repo import Repo
from pydvl.reporting.scores import compute_removal_score
from pydvl.utils import Utility

from tfm.utils import setup_logger

logger = setup_logger()

def run(dataset_name: str, budget: int):
    # TODO: Esto hay que meterlo
    params = params_show()
    
    # Create the output directory
    experiment_output_dir = (
        Path(Repo.find_root())
        / "output"
        / "weighted_acc"
        / f"dataset={dataset_name}"
        / "results"
        / f"{budget=}"
    )
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    
    logger.info("Finished weighted accuracy experiment")