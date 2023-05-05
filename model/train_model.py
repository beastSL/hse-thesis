from argparse import ArgumentParser
from pathlib import Path

from trainer import MultiTaskTrainer

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import json

def train_model(config_path):
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    batch_size = 32
    max_len = 512
    optimizer_params = {
        "lr": 2e-5,
        "betas": (0.9, 0.98)
    }
    shared_one_cycle_policy=False
    num_shared_epochs = 0
    scheduler_params = {
        "max_lr": 2e-4,
        "three_phase": True
    }
    multi_task_trainer = MultiTaskTrainer(
        config,
        optimizer_params,
        batch_size,
        max_len,
        shared_one_cycle_policy,
        num_shared_epochs,
        scheduler_params
    )
    multi_task_trainer.train(num_epochs_main=5)

if __name__ == "__main__":
    parser = ArgumentParser()
    data_group = parser.add_argument_group("Data paths")
    data_group.add_argument(
        "--config-path", type=Path, help="Paths to the config containing information about the data"
    )

    args = parser.parse_args()

    train_model(args.config_path)