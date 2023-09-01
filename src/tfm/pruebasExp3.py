import pickle
import numpy as np
from utils.utils import setup_logger, undersamp_equilibration

logger = setup_logger()

def run():
    logger.info("Prueba")
    openML_path = '../../data/openML/'

    np.random.seed(999)

    # Dictionary to map dataset names to their respective file names
    dataset_mapping = {
        'phoneme': 'phoneme_1489.pkl',
    }

    data_dict = pickle.load(open(openML_path + dataset_mapping['phoneme'], 'rb'))
    data, target = data_dict['X_num'], data_dict['y']
    target = (target == 1).astype(np.int32)
    elementos_unicos, conteo = np.unique(target, return_counts=True)
    print(f"Disponemos de {elementos_unicos}, ditribuidos en {conteo}")
    data, target = undersamp_equilibration(data, target)
    elementos_unicos, conteo = np.unique(target, return_counts=True)
    print(f"Disponemos de {elementos_unicos}, ditribuidos en {conteo}")
    

if __name__ == "__main__":
    run()