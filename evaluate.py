import argparse
import torch
from tqdm import tqdm
from trainer import TrainingModule
from dataset import get_test_loader
import os
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


def evaluate_model(checkpoint_path=None, save_output = False):

    gpu = 4
    device = torch.device('cuda:'+str(gpu))

    module = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
    module.config['trainer_params']['gpus'] = [gpu]
    print(f'Loaded weights from \n {checkpoint_path}')
    module.eval()

    module.to(device)
    model = module.model
    # get model config
    config = module.config

    test_dataloader = get_test_loader(synergy_file=config["trainer_args"]["synergy_file"],
                                      gene_exp_file=config["trainer_args"]["gene_exp_file"],
                                      drug_smile_file=config["trainer_args"]["drug_smile_file"],
                                      dose_response_file=config["trainer_args"]["dose_response_file"],
                                      batch_size=config["trainer_params"]["batch_size"],
                                      num_workers=0)


    output = {}
    output['drug1'] = []
    output['drug2'] = []
    output['drug1_dose'] = []
    output['drug2_dose'] = []
    output['cell'] = []

    output['dose_response'] = []
    output['dose_response_gt'] = []

    gt_flat = []
    pred_flat = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_dataloader)):
            drug1_smile = batch['drug1_smile'].to(device)
            drug2_smile = batch['drug2_smile'].to(device)
            cell_gex = batch['cell_gex'].to(device)
            drug1_dose = batch['drug1_dose'].to(device)
            drug2_dose = batch['drug2_dose'].to(device)
            predictions = model(drug1_smile, drug2_smile, cell_gex, drug1_dose, drug2_dose)
            output['drug1'].extend(batch['drug1'])
            output['drug2'].extend(batch['drug2'])
            output['cell'].extend(batch['cell'])
            output['dose_response'].extend(predictions['dose_response'])
            output['dose_response_gt'].extend(batch['dose_response'].to(device))
            output['drug1_dose'].extend(drug1_dose)
            output['drug2_dose'].extend(drug2_dose)
            gt_flat.extend(batch['dose_response'].flatten().tolist())
            pred_flat.extend(predictions['dose_response'].flatten().tolist())


    y_te = np.array(gt_flat)
    y_pred_te_ = np.array(pred_flat)
    RMSE = np.sqrt(mean_squared_error(y_te, y_pred_te_))
    RPearson = np.corrcoef(y_te, y_pred_te_)[0, 1]
    RSpearman, _ = spearmanr(y_te, y_pred_te_)

    output_path = os.path.join(config["trainer_args"]["save_dir"], checkpoint_path.split('/')[-4],
                               checkpoint_path.split('/')[-3])


    # Save output as pickle if needed
    if save_output:
        with open(output_path + '/output.pickle', 'wb') as handle:
            pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Output saved to %s" % output_path)
    print("RMSE: %f\nPearson correlation: %f\nSpearman correlation: %f" % (RMSE, RPearson, RSpearman))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a model for dose response prediction.")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint file')

    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    evaluate_model(checkpoint_path)