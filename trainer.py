import torch
import lightning as l

from deepsynba import DeepSynBa
from losses import mse_loss_new
from torchmetrics.regression import MeanSquaredError


class TrainingModule(l.LightningModule):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.eval_metrics = config['trainer_args']['eval_metrics']
        self.distributed = len(self.config.get("trainer_params",{}).get("gpus",0)) > 1
        print("self.distributed:",self.distributed)
        self.save_hyperparameters(config)
        self.weight_multiplier = self.config['trainer_args']['weight_multiplier']
        self.apply_final_activation = self.config['model_params']['apply_final_activation']

        self.optimizer_map = {
            "AdamW": torch.optim.AdamW,
            "Adam": torch.optim.Adam,
            "SGD": torch.optim.SGD
        }

        self.scheduler_map = {
            "LinearLR": torch.optim.lr_scheduler.LinearLR
        }

        self.target_keys = self.config['trainer_args']['target_keys']

        self._initialize_metrics(split='train')
        self._initialize_metrics(split='val')

        self.model = self._initialize_model(self.config)
        self.loss_fn = mse_loss_new

    @staticmethod
    def _initialize_model(config):
        return DeepSynBa(config)

    def _initialize_metrics(self, split='train'):
        setattr(self, f"{split}_mse_dose_response", MeanSquaredError())

    def forward(self, batch, split, is_train=True):
        predictions = self.model(batch['drug1_smile'], batch['drug2_smile'], batch['cell_gex'],
                                 batch['drug1_dose'], batch['drug2_dose'])

        batch_size = batch['drug1_smile'].shape[0]
        losses = self.compute_loss(predictions, batch)
        loss_all = losses['dose_response_loss']

        self.log(f'{split}_loss', loss_all, prog_bar=True,
                 sync_dist=self.distributed, batch_size=batch_size)
        self.log_losses(losses=losses, split=split, batch_size=batch_size)
#
        # Metrics
        if not self.trainer.sanity_checking:
            self.accumulate_metrics(predictions, batch, split=split)

        return predictions, loss_all

    def training_step(self, batch, batch_idx):
        predictions, loss = self.forward(batch, 'train', is_train=True)
        return loss

    def on_training_epoch_end(self):
        if not self.trainer.sanity_checking:
            self.log_metrics(split='train')

    def validation_step(self, batch, batch_idx):
        predictions, loss = self.forward(batch, 'val', is_train=False)
        return loss

    def on_validation_epoch_end(self):
        pass
        if not self.trainer.sanity_checking:
            self.log_metrics(split='val')

    def compute_loss(self, predictions, targets):
        loss, factor, loss_all = {}, {}, []

        pred = predictions['dose_response']
        target = targets['dose_response']
        loss['dose_response_loss'] = self.loss_fn(pred, target)
        return loss

    def log_losses(self, losses, batch_size, split='train'):
        keys = losses.keys()
        for key in keys:
            log_key = f"{split}/{key}"
            log_value = losses[key]
            self.log(log_key, log_value, prog_bar=True,
                     sync_dist=self.distributed, batch_size=batch_size)

    def accumulate_metrics(self, predictions, batch, split='train'):
        metric = getattr(self, f"{split}_mse_dose_response")
        metric(predictions['dose_response'], batch['dose_response'])

    def log_metrics(self, split='train'):
        metric = getattr(self, f"{split}_mse_dose_response")
        log_key = f"{split}_metric_dose_response"
        self.log(log_key, metric, sync_dist=self.distributed)


    def configure_optimizers(self):
        optimizer_params = self.config["optimizer"]
        optimizer_fn = self.optimizer_map[optimizer_params['name']]
        optimizer_params = {k: v for k, v in optimizer_params.items() if k != 'name'}
        optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)

        scheduler_params = self.config["scheduler"]
        scheduler_fn = self.scheduler_map[scheduler_params['name']]
        scheduler_params = {k: v for k, v in scheduler_params.items() if k != 'name'}
        scheduler = scheduler_fn(optimizer, **scheduler_params)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
