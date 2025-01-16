import torch.nn as nn


class PredictionHead(nn.Module):
    def __init__(self, in_channels=1952, emb_size=1024, apply_final_activation=False):
        super(PredictionHead, self).__init__()
        self.loss_weight = 0.0 #torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        layers = [Block(in_channels, emb_size),
                 Block(emb_size, emb_size / 2),
                 Block(emb_size / 2, emb_size / 4),
                 nn.Linear(int(emb_size / 4), 1)]

        if apply_final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif apply_final_activation == 'relu':
            layers.append(nn.ReLU())

        self.prediction_head = nn.Sequential(*layers)

        for name, param in self.prediction_head.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)


    def forward(self, input):
        return self.prediction_head(input.squeeze())


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.block = nn.Sequential(nn.Linear(int(in_channels), int(out_channels)),
                                             nn.LayerNorm((int(out_channels),), eps=1e-05, elementwise_affine=True),
                                             nn.ReLU(),
                                             #SwishLayer(),
                                             nn.Dropout(0.2))

    def forward(self, x):
        return self.block(x)

class DoseResponsePredictor(nn.Module):
    def __init__(self, in_channels=1952, emb_size=1024):
        super(DoseResponsePredictor, self).__init__()
        self.loss_weight = 0.0 #torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

        layers = [Block(in_channels, emb_size),
                  Block(emb_size, emb_size / 2),
                  Block(emb_size / 2, emb_size / 4),
                  nn.Linear(int(emb_size / 4), 4),
                  nn.ReLU()]

        # convolution layers increase channel from 1 to 4
        # 1. 1952 -> 1024
        # 2. 1024 -> 512
        # 3. 512 -> 256
        # 4. 256 -> 4
        #self.conv_layers = nn.Conv1d(in_channels=1, out_channels=4, kernel_size=1)
        #self.final_relu = nn.ReLU()
        self.prediction_head = nn.Sequential(*layers)

        # nn.init.kaiming_normal_(self.conv_layers.weight)
        # if self.conv_layers.bias is not None:
        #     nn.init.zeros_(self.conv_layers.bias)

        for name, param in self.prediction_head.named_parameters():
            if 'weight' in name and len(param.data.shape) > 1:
                nn.init.kaiming_normal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, input):
        out = self.prediction_head(input.squeeze())
        #out = out.unsqueeze(1)
        #out = self.conv_layers(out)
        #out = self.final_relu(out)
        return out
