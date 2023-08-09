import os
import dataset
import pickle
import numpy as np
from utils.utils import setup_logger, oversamp_equilibration

logger = setup_logger()

def run():
    logger.info("Cargamos el fminst")
    mnist = dataset.create_fmnist()

    

if __name__ == "__main__":
    run()