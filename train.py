import argparse
import yaml
from random import seed

from trainer import TrainingModule
import lightning as l
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch import loggers as pl_loggers
from dataset import get_loaders
import os
import time

def train_model(config=None, date_time=None):
    # Initialize parser
    gpu_mapping = {0: 2, 1: 0, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 1}
    parser = argparse.ArgumentParser(description="Train a model for Drug Combination Surface prediction.")
    # Adding arguments
    parser.add_argument('--config_path', type=str, default='config.yaml',
                        help='Path to the configuration file')

    # Parse arguments
    args = parser.parse_args()

    if config is None:
        with open(args.config_path, 'r') as file:
            config = yaml.safe_load(file)

    seed(10)
    if date_time is None:
        date_time = time.strftime('%Y_%m_%d_at_%H%M')
    save_dir = os.path.join(config["trainer_args"]["save_dir"], date_time + '_' + config["tag"])


    train_dataloader, val_dataloader = get_loaders(synergy_file=config["trainer_args"]["synergy_file"],
                                                   gene_exp_file=config["trainer_args"]["gene_exp_file"],
                                                   drug_smile_file=config["trainer_args"]["drug_smile_file"],
                                                   dose_response_file=config["trainer_args"]["dose_response_file"],
                                                   batch_size=config["trainer_params"]["batch_size"],
                                                   num_workers=config["trainer_args"]["num_workers"],)
                                                   #target_keys=config["trainer_args"]["target_keys"],)

    # Prepare the model checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor='val/dose_response_loss',
        mode='min',
        filename='{epoch:02d}-{val_loss_dose_response:.2f}',
        save_top_k=config["trainer_args"]["save_top_k"],
    )
    checkpoint_callback.FILE_EXTENSION = ".pth"

    # Initialize the trainer
    model = TrainingModule(config=config)

    if config["trainer_args"]["logger"] == 'wandb':
        logger = pl_loggers.WandbLogger(dir='/wandb_dir',
                                        entity='synba-team',
                                        project=config["tag"],
                                        name=config["tag"],
                                        save_dir=config["trainer_args"]["save_dir"],
                                        id=save_dir.split('/')[-1])

    elif config["trainer_args"]["logger"] == 'tensorboard':
        logger = pl_loggers.TensorBoardLogger(save_dir=config["trainer_args"]["save_dir"],
                                              name=date_time + '_' + config["tag"])

    gpus = config["trainer_params"]["gpus"]
    trainer = l.Trainer(devices=gpus,
                        accelerator=config["trainer_params"].get("accelerator", "auto"),
                        strategy=config["trainer_params"].get("strategy", "auto"),
                        precision=config["trainer_params"].get("precision"),
                        sync_batchnorm=True,
                        gradient_clip_val=config["trainer_params"]["grad_norm_clip"],
                        max_epochs=config["trainer_params"]["max_epochs"],
                        logger=logger,
                        callbacks=[checkpoint_callback],
                        log_every_n_steps=config["trainer_params"]["logging_interval"])


    # Train the model
    trainer.fit(model, train_dataloader, val_dataloader)
    logger.experiment.finish()



if __name__ == "__main__":
    train_model()

