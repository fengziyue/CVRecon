import argparse
import glob
import json
import os
import random
import subprocess

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.plugins import DDPPlugin

from cvrecon import collate, data, lightningmodel, utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--gpus", default=1)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    pl.seed_everything(config["seed"])
    
    if config['wandb_runid'] is not None:
        logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config, id=config['wandb_runid'], resume="must")
    else:
        logger = pl.loggers.WandbLogger(project=config["wandb_project_name"], config=config)
    subprocess.call(
        [
            "zip",
            "-q",
            os.path.join(str(logger.experiment.dir), "code.zip"),
            "config.yml",
            *glob.glob("cvrecon/*.py"),
            *glob.glob("scripts/*.py"),
        ]
    )
    
    ckpt_dir = os.path.join(str(logger.experiment.dir), "ckpts")
    checkpointer = pl.callbacks.ModelCheckpoint(
        save_last=True,
        dirpath=ckpt_dir,
        filename='{epoch}-{val/voxel_loss_medium:.4f}',
        verbose=True,
        save_top_k=20,
        monitor="val/voxel_loss_medium",
    )
    callbacks = [checkpointer, lightningmodel.FineTuning(config["initial_epochs"], config["cost_volume"])]
    
    if config["use_amp"]:
        amp_kwargs = {"precision": 16}
    else:
        amp_kwargs = {}
    
    model = lightningmodel.LightningModel(config)


    trainer = pl.Trainer(
        gpus=args.gpus,
        logger=logger,
        benchmark=True,
        max_epochs=config["initial_epochs"] + config["finetune_epochs"] + 300,
        check_val_every_n_epoch=5,
        detect_anomaly=False,
        callbacks=callbacks,
        reload_dataloaders_every_n_epochs=1,  # a hack so batch size can be adjusted for fine tuning
        strategy=DDPPlugin(find_unused_parameters=True),
        accumulate_grad_batches=1,
        num_sanity_val_steps=1,
        **amp_kwargs,
    )
    trainer.fit(model, ckpt_path=config["ckpt"])
