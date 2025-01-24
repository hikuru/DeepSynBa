import torch
import torch.nn as nn
from blocks.drug_encoder import PretrainedDrugEncoder
from blocks.predictor import PredictionHead, DoseResponsePredictor
import numpy as np
import os

class DeepSynBa(nn.Module):
    def __init__(self, config):
        super(DeepSynBa, self).__init__()

        self.bias = True
        self.drug_encoder = PretrainedDrugEncoder()

        in_channels = config['model_params']['gene_dim'] + config['model_params']['drug_dim']
        self.drug_cell_agg = nn.Sequential(
            nn.Linear(in_channels, config['model_params']['gene_channel']),
            nn.BatchNorm1d(config['model_params']['gene_channel']),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.target_keys = config['trainer_args']['target_keys']
        self.predict_matrix = config['trainer_args']['predict_matrix']

        self.predictors = nn.ModuleDict()
        for target in self.target_keys:
            self.predictors[target] = PredictionHead(in_channels=config['model_params']['gene_channel'] * 2,
                                                     emb_size=config['model_params']['emb_size'],
                                                     apply_final_activation=config['model_params']['apply_final_activation'][target])

        if self.bias:
            self.bias_predictor1 = DoseResponsePredictor(in_channels=config['model_params']['gene_channel'] * 2,
                                                    emb_size=config['model_params']['emb_size'])

            self.bias_predictor2 = DoseResponsePredictor(in_channels=config['model_params']['gene_channel'] * 2,
                                                    emb_size=config['model_params']['emb_size'])

        self.dose_matrix_root = os.path.join(config["trainer_args"]["data_dir"], 'combinations_full_notz/')

        for name, param in self.drug_cell_agg.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


    def synba_likelihood_2d(self, x_1, x_2, e_1, e_2, e_3, logC_1, logC_2, h_1, h_2, sigma, alpha, add_noise=False):
        e_0 = 100
        x_1, x_2 = x_1 + 1e-6, x_2 + 1e-6
        #x_1, x_2 = torch.tensor(x_1, device=e_1.device), torch.tensor(x_2, device=e_1.device)
        # Use the log-sum-exp trick for stability

        max_exp = torch.maximum(logC_1 * h_1 + logC_2 * h_2,
                             torch.maximum(x_1.log() * h_1 + logC_2 * h_2,
                                        torch.maximum(logC_1 * h_1 + x_2.log() * h_2,
                                                   x_1.log() * h_1 + x_2.log() * h_2)))

        # Compute stable exponential terms
        exp_A = torch.exp(logC_1 * h_1 + logC_2 * h_2 - max_exp) * e_0
        exp_B = torch.exp(x_1.log() * h_1 + logC_2 * h_2 - max_exp) * e_1 * e_0
        exp_C = torch.exp(logC_1 * h_1 + x_2.log() * h_2 - max_exp) * e_2 * e_0
        exp_D = torch.exp(x_1.log() * h_1 + x_2.log() * h_2 - max_exp) * e_3 * e_0 * alpha
        numerator = exp_A + exp_B + exp_C + exp_D
        denominator = torch.exp(logC_1 * h_1 + logC_2 * h_2 - max_exp) + \
                      torch.exp(x_1.log() * h_1 + logC_2 * h_2 - max_exp) + \
                      torch.exp(logC_1 * h_1 + x_2.log() * h_2 - max_exp) + \
                      torch.exp(x_1.log() * h_1 + x_2.log() * h_2 - max_exp) * alpha
        if add_noise:
            z = np.random.normal(0, sigma, len(x_1))
        else:
            z = 0
        y = numerator / denominator + z

        return y

    def post_process(self, input, drug1_dose, drug2_dose):

        y_ = self.synba_likelihood_2d(drug1_dose,
                                      drug2_dose,
                                      input['e1_mean'].unsqueeze(dim=2),
                                      input['e2_mean'].unsqueeze(dim=2),
                                      input['e3_mean'].unsqueeze(dim=2),
                                      input['logC1_mean'].unsqueeze(dim=2),
                                      input['logC2_mean'].unsqueeze(dim=2),
                                      input['h1_mean'].unsqueeze(dim=2),
                                      input['h2_mean'].unsqueeze(dim=2),
                                      input['sigma_mean'].unsqueeze(dim=2),
                                      input['alpha_mean'].unsqueeze(dim=2),
                                      add_noise=False)

        return y_

    def forward(self, drug1_smile, drug2_smile, gene_expr, drug1_dose, drug2_dose):
        # Apply drug encoder for smiles to get drug features
        drug1_features = self.drug_encoder(drug1_smile) # [B, N, 768]
        drug2_features = self.drug_encoder(drug2_smile)  # [B, N, 768]

        b_size = drug1_features.shape[0]
        # gene_expr -> [B, 976, 1]
        if torch.isnan(gene_expr).any():
            gene_expr = torch.nan_to_num(gene_expr)
        if torch.isnan(drug1_features).any():
            drug1_features = torch.nan_to_num(drug1_features)
        if torch.isnan(drug2_features).any():
            drug2_features = torch.nan_to_num(drug2_features)


        drug1_features = torch.permute(drug1_features, (0, 2, 1))
        drug2_features = torch.permute(drug2_features, (0, 2, 1))

        in1 = torch.cat((gene_expr, drug1_features), dim=1).squeeze()
        in2 = torch.cat((gene_expr, drug2_features), dim=1).squeeze()

        conditional_cell_line1 = self.drug_cell_agg(in1)
        conditional_cell_line2 = self.drug_cell_agg(in2)


        multiview_cell = torch.cat((conditional_cell_line1, conditional_cell_line2), dim=1)
        output = {}

        for target in self.target_keys:
            output[target] = self.predictors[target](multiview_cell)

        drug1_dose = drug1_dose.reshape(b_size, 4, 1).repeat(1, 1, 4)
        drug2_dose = drug2_dose.reshape(b_size, 1, 4).repeat(1, 4, 1)
        #############
        if self.bias:
            out1 = self.bias_predictor1(multiview_cell)
            out2 = self.bias_predictor2(multiview_cell)

            out1 = out1.reshape(b_size, 4, 1).repeat(1, 1, 4)
            out2 = out2.reshape(b_size, 1, 4).repeat(1, 4, 1)
            bias = torch.mul(out1, drug1_dose) + torch.mul(out2, drug2_dose)
            output['bias'] = bias
        #############

        output['dose_response'] = self.post_process(output, drug1_dose, drug2_dose)
        if self.bias:
            output['dose_response'] = output['dose_response'] + bias

        return output