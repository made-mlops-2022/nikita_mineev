import hydra
from hydra.core.config_store import ConfigStore
from config_schemes.config import Config
from data.make_dataset import make_dataset
from models.predict_model import predict_model
import logging
import time


logging.basicConfig(format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
logger = logging.getLogger('train logger')
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name="train", node=Config)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def predict(cfg : Config) -> None:
    s_time = time.time()
    logger.info("Caclucate Predictions...")
    logger.info("Loading dataset...")
    X_test, = make_dataset(cfg.dataset.path.test,
                           train_size=0,
                           val_size=0,
                           test_size=0,
                           random_state=cfg.random_state)
    logger.info("Dataset loaded.")
    logger.info("Loading model and making predictions...")
    predict_model(X_test,
                  cfg.dataset.features.categorical_features,
                  cfg.model.save_path,
                  cfg.model.save_predict_path,
                  cfg.model.transform_dataset,
                  cfg.model.transformer_save_path)
    logger.info(f"Predicitons made in {time.time() - s_time:.3f}s")


if __name__ == "__main__":
    predict()
