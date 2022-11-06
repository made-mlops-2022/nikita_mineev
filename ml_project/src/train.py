import hydra
from hydra.core.config_store import ConfigStore
from config_schemes.config import Config
from data.make_dataset import make_dataset
from models.train_model import train_model
import logging
import time


logging.basicConfig(format="%(asctime)s\t%(levelname)s\t%(name)s\t%(message)s")
logger = logging.getLogger('train logger')
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name="train", node=Config)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(cfg : Config) -> None:
    s_time = time.time()
    logger.info("Start Training...")
    logger.info("Loading dataset...")
    X_train, X_val, X_test = make_dataset(cfg.dataset.path.train,
                                          cfg.dataset.split.train_size,
                                          cfg.dataset.split.val_size,
                                          cfg.dataset.split.test_size,
                                          cfg.random_state)
    logger.info("Dataset loaded.")
    logger.info("Fitting model...")
    train_model(X_train,
                X_val,
                X_test,
                cat_features=cfg.dataset.features.categorical_features,
                target_features=cfg.dataset.features.target_features,
                model_save_path=cfg.model.save_path,
                transform_dataset=cfg.model.transform_dataset,
                transformer_save_path=cfg.model.transformer_save_path,
                logger=logger,
                **cfg.model.params)
    logger.info("Model fitted.")
    logger.info(f"Training complete in {time.time() - s_time:.3}s")


if __name__ == "__main__":
    train()
