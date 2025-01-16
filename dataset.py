import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import json
import torch
import numpy as np
import pickle

class NciAlmanacDataset(Dataset):
    def __init__(self, synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys=None, split='train'):
        self.dose_matrix_root = os.path.join('/mnt/hikuru_backup/NCI_Almanac_synergy', 'combinations_full_notz/')
        # get synergy data
        self.synergy_data = self.get_splitted_data(synergy_file, split)
        self.loewe_bliss_gt = pd.read_csv('/mnt/hikuru_backup/NCI_Almanac_synergy/Loewe_Bliss_synergy_scores_ALL.csv')
        self.cell2gex = self.get_gene_expressions(gene_exp_file)

        self.drug2embedding = self.get_drug_embeddings(drug_smile_file)
        self.dose_response_data = self.get_dose_response_data(dose_response_file)
        self.target_keys = target_keys


    def get_dose_response_data(self, dose_response_file):
        with open(dose_response_file, 'rb') as handle:
            matrix = pickle.load(handle)
        return matrix

    def get_splitted_data(self,data_folder, split):
        full_dataset = pd.read_csv(data_folder)
        keep = full_dataset['alpha_mean'] <= 100
        full_dataset = full_dataset[keep]
        keep = np.logical_and(full_dataset['dHSA_mean'] <= 50, full_dataset['dHSA_mean'] >= - 50)
        full_dataset = full_dataset[keep]

        #filter based on train-validation-test splits
        split_column = 'split'
        # Split based on the values in the split_column
        if split == 'train':
            split_val = 1
        elif split == 'val':
            split_val = 2
        else:
            split_val = 3
        splitted_data = full_dataset[full_dataset[split_column] == split_val]
        return splitted_data

    @staticmethod
    def get_gene_expressions(filename):
        gexes = pd.read_csv(filename)
        cell_lines = gexes.columns[1:].values
        cell2gex = {}
        for ind in range(cell_lines.shape[0]):
            cell = cell_lines[ind]
            cell2gex[cell] = gexes[cell].values.astype('float32')

        return cell2gex

    @staticmethod
    def get_gene_exps_json(filename):
        cell2gex = {}
        with open(filename, 'r') as f:
            data = json.load(f)

        for cell in data:
            cell2gex[cell] = np.array(data[cell], dtype='float32')
        return cell2gex

    @staticmethod
    def get_drug_embeddings(filename):
        embs = torch.load(filename)
        return embs

    @staticmethod
    def convert_cell_name(cell):
        cell = cell.replace('-', '_').replace('/ATCC', '')
        cell = cell.replace(' ', '').replace('(TB)', '').replace('MDA_MB_468', 'MDA_N').replace('NCI/', 'NCI_')
        cell = cell.replace('RXF393', 'RXF_393').replace('T_47D', 'T47D')
        return cell

    def __len__(self):
        return len(self.synergy_data)

    @staticmethod
    def normalize_drug_name(drug_name):
        if drug_name[-1] == ' ':
            drug_name = drug_name[:-1]
        return drug_name

    def __getitem__(self, index):
        sample = self.synergy_data.iloc[index]
        response_key = sample['drug1'] + '_' + sample['drug2'] + '_' + sample['cell']

        dose_response = self.dose_response_data[response_key]
        gex = np.expand_dims(self.cell2gex[self.convert_cell_name(sample['cell'])], axis=1)

        data = {
            'drug1_smile': self.drug2embedding[self.normalize_drug_name(sample['drug1'])],
            'drug2_smile': self.drug2embedding[self.normalize_drug_name(sample['drug2'])],
            'cell_gex': gex, #torch.nn.functional.normalize(torch.from_numpy(gex), dim=0),
            'drug1': sample['drug1'],
            'drug2': sample['drug2'],
            'cell': sample['cell'],
            'dose_response': dose_response['matrix'].astype('float32'),
            'drug1_dose': dose_response['x1_dose'].astype('float32') * 1e6,
            'drug2_dose': dose_response['x2_dose'].astype('float32') * 1e6,
        }

        if self.target_keys is not None:
            # set target data
            for key in self.target_keys:
                data[key] = sample[key].astype('float32')

        return data


def get_loaders(synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys=None, batch_size=128, num_workers=16):
    train = NciAlmanacDataset(synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys, split='train')
    val = NciAlmanacDataset(synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys, split='val')
    trainloader = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valloader = DataLoader(val, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)

    print("The number of samples in train: {}".format(len(train)))
    print("The number of samples in validation: {}".format(len(val)))
    return trainloader, valloader

def get_test_loader(synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys=None, batch_size=128, num_workers=16):
    test = NciAlmanacDataset(synergy_file, gene_exp_file, drug_smile_file, dose_response_file, target_keys, split='test')
    testloader = DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)
    print("The number of samples in test: {}".format(len(test)))
    return testloader