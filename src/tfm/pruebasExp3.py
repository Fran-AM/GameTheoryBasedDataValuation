import pickle
import numpy as np
from utils.utils import setup_logger, undersamp_equilibration
from dataset import get_openML_data

logger = setup_logger()

def run():
    logger.info("Prueba")
    openML_path = '../../data/openML/'

    np.random.seed(999)

    phoneme = get_openML_data(
        dataset='phoneme',
        n_data=200,
        n_test=200,
        flip_ratio=0.0
    )


if __name__ == "__main__":
    run()