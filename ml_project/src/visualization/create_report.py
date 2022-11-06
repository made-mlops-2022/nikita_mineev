import dataprep.eda
import os
import pandas as pd
import hydra
from hydra.core.config_store import ConfigStore
from config_schemes.config import Config

cs = ConfigStore.instance()
cs.store(name="train", node=Config)

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def create_report(cfg : Config) -> None:
    df = pd.read_csv(cfg.dataset.path.raw)
    dataprep.eda.create_report(df, title=cfg.dataset.name)\
        .save(os.path.join(cfg.eda_reports_dir, "report"))


if __name__ == "__main__":
    create_report()
