from torch import nn
# from fast_transformers.feature_maps import GeneralizedRandomFeatures
# from functools import partial
# from fast_transformers.masking import LengthMask
# from rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
# import yaml
# from rdkit import Chem
# from argparse import Namespace
# from tokenizer.tokenizer import MolTranBertTokenizer
# import numpy as np
# import torch

class PretrainedDrugEncoder(nn.Module):
    def __init__(self):
        super(PretrainedDrugEncoder, self).__init__()

    def forward(self, embedding):
        return embedding

# class DrugEncoder(nn.Module):
#     def __init__(self):
#         super(DrugEncoder, self).__init__()
#
#         self.molformer_config = self.get_molformer_config()
#         self.tokenizer = MolTranBertTokenizer('/home/halil/molformer/notebooks/pretrained_molformer/bert_vocab.txt')
#         ckpt = '/mnt/hikuru_backup/molformer_backup/data/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt'
#
#         self.drug_feature_extractor = self.get_drug_molformer(ckpt)
#
#     def get_drug_molformer(self, ckpt):
#         drug_feature_extractor = DrugMolformerEncoder(self.molformer_config, self.tokenizer.vocab)
#         checkpoint = torch.load(ckpt)
#         drug_feature_extractor.load_state_dict(checkpoint['state_dict'], strict=False)
#         for param in drug_feature_extractor.parameters():
#             param.requires_grad = False
#         return drug_feature_extractor.to(torch.device('cuda:1'))
#
#     def canonicalize(self, s):
#         return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)
#
#     def get_molformer_config(self):
#         with open('/mnt/hikuru_backup/molformer_backup/data/Pretrained MoLFormer/hparams.yaml', 'r') as f:
#             config = Namespace(**yaml.safe_load(f))
#         return config
#
#     def forward(self, smile):
#         # Transpose gene expression for compatibility with Transformer
#         smile_enc = self.tokenizer.batch_encode_plus(smile, padding=True, add_special_tokens=True)
#         idx = torch.tensor(smile_enc['input_ids'], device=torch.device('cuda:1'))
#         mask = torch.tensor(smile_enc['attention_mask'], device=torch.device('cuda:1'))
#         # Apply Molformer encoder for drug smiles
#         drug_features = self.drug_feature_extractor(idx, mask)  # [B, N, 768]
#
#         return drug_features




# class DrugMolformerEncoder(nn.Module):
#     def __init__(self, config, vocab):
#         super(DrugMolformerEncoder, self).__init__()
#         builder = rotate_builder.from_kwargs(
#             n_layers=config.n_layer,
#             n_heads=config.n_head,
#             query_dimensions=config.n_embd // config.n_head,
#             value_dimensions=config.n_embd // config.n_head,
#             feed_forward_dimensions=config.n_embd,
#             attention_type='linear',
#             feature_map=partial(GeneralizedRandomFeatures, n_dims=config.num_feats),
#             activation='gelu',
#         )
#         n_vocab, d_emb = len(vocab), config.n_embd
#         self.tok_emb = nn.Embedding(n_vocab, config.n_embd)
#
#         self.drop = nn.Dropout(config.d_dropout)
#         ## transformer
#         self.blocks = builder.get()
#         from pytorch_lightning.utilities import seed
#         seed.seed_everything(config.seed)
#
#
#     def forward(self, idx, mask):
#         b, t = idx.size()
#         token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
#         embeddings = self.blocks(token_embeddings, length_mask=LengthMask(mask.sum(-1)))
#         return embeddings
