import dataset
from utils import setup_logger

logger = setup_logger()

def run():
    logger.info("Cargamos el fminst")
    trainset, testset = dataset.create_fmnist()

    logger.info("Comprobamos tamaño de trainset")
    print(trainset.data.shape)

    logger.info("Comprobamos tamaño de testset")
    print(len(testset.data.shape))

    
if __name__ == "__main__":
    run()