from train import train_model
import yaml
import time
import os

os.environ["WANDB_DIR"] = "/mnt/hikuru_backup/wandb_dir"

with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

channels = [1024, 2048] # the number of channels in the gene expression
emb_sizes = [64, 128, 256] # the embedding size of the prediction head

for gene_channel in channels:
    config["model_params"]["gene_channel"] = gene_channel
    for emb_size in emb_sizes:
        config["model_params"]["emb_size"] = emb_size
        date_time = time.strftime('%Y_%m_%d_at_%H%M')
        train_model(config, date_time)

# how to run:
# python experimenter.py

