import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from hydra import main
from omegaconf import DictConfig
from pathlib import Path
import torch
 
from .models.hoam import HOAM, HOAMV2
from .losses.hybrid_margin import HybridMarginLoss
from pytorch_metric_learning.losses import SubCenterArcFaceLoss, ArcFaceLoss
from .data.transforms import build_transforms
from .data.statistics import DataStatistics
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
 
 
class HOAMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
 
    def setup(self, stage=None) -> None:
        mean, std = DataStatistics.get_mean_std(
            self.data_dir,
            self.image_size,
            cache_file="mean_std.json"
        )
        self.train_ds = ImageFolder(
            self.data_dir / 'train',
            transform=build_transforms('train', self.image_size, mean, std)
        )
        self.val_ds = ImageFolder(
            self.data_dir / 'val',
            transform=build_transforms('val', self.image_size, mean, std)
        )
 
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
 
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
 
 
class LightningModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
 
        # Dynamically determine number of classes from train folder
        train_path = Path(cfg.data.data_dir) / 'train'
        if not train_path.exists():
            raise FileNotFoundError(f"Training data directory not found: {train_path}")
        num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
 
        # Model selection
        model_map = {
            'HOAM': HOAM,
            'HOAMV2': HOAMV2,
        }
        model_cls = model_map.get(cfg.model.structure)
        if model_cls is None:
            raise ValueError(f"Unknown model structure: {cfg.model.structure}")
        self.model = model_cls(
            backbone_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            embedding_size=cfg.model.embedding_size
        )
 
        # Loss (criterion) selection
        loss_map = {
            'HybridMarginLoss': HybridMarginLoss,
            'SubCenterArcFaceLoss': SubCenterArcFaceLoss,
            'ArcFaceLoss': ArcFaceLoss,
        }
        loss_type = cfg.loss.type
        loss_cls = loss_map.get(loss_type)
        if loss_cls is None:
            raise ValueError(f"Unknown loss type: {loss_type}")
 
        if loss_type == 'HybridMarginLoss':
            self.criterion = loss_cls(
                num_classes=num_classes,
                embedding_size=cfg.model.embedding_size,
                subcenter_margin=cfg.loss.subcenter_margin,
                subcenter_scale=cfg.loss.subcenter_scale,
                sub_centers=cfg.loss.sub_centers,
                triplet_margin=cfg.loss.triplet_margin,
                center_loss_weight=cfg.loss.center_loss_weight
            )
        else:
            # SubCenterArcFaceLoss and ArcFaceLoss signatures
            params = {
                'num_classes': num_classes,
                'embedding_size': cfg.model.embedding_size,
            }
            # margin/scale only for ArcFace variants
            if hasattr(cfg.loss, 'subcenter_margin'):
                params['margin'] = cfg.loss.subcenter_margin
            if hasattr(cfg.loss, 'subcenter_scale'):
                params['scale'] = cfg.loss.subcenter_scale
            if loss_type == 'SubCenterArcFaceLoss' and hasattr(cfg.loss, 'sub_centers'):
                params['sub_centers'] = cfg.loss.sub_centers
 
            self.criterion = loss_cls(**params)
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
 
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        embeds = self(imgs)
        loss = self.criterion(embeds, labels)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss
 
    def validation_step(self, batch, batch_idx) -> None:
        imgs, labels = batch
        embeds = self(imgs)
        loss = self.criterion(embeds, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True)
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.training.lr
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.training.epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1
            }
        }
 
# Dynamically resolve config path relative to this file
@main(
    version_base="1.3",
    config_path=str(Path(__file__).parents[2] / "configs"),
    config_name="config"
)
def run(cfg: DictConfig) -> None:
    """
    Entry point for training via Hydra + PyTorch Lightning.
    """
    data_module = HOAMDataModule(
        data_dir=cfg.data.data_dir,
        image_size=cfg.data.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers
    )
 
    model = LightningModel(cfg)
 
    logger = TensorBoardLogger('logs', name=cfg.experiment.name)
    checkpoint = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename='{epoch:02d}-{val/loss:.4f}',
        monitor='val/loss',
        save_top_k=1,
        mode='min'
    )
    early_stop = EarlyStopping(
        monitor='val/loss',
        patience=cfg.training.patience,
        mode='min'
    )
 
    # Initialize Trainer (auto handles GPU/CPU)
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=[checkpoint, early_stop],
        accelerator="auto",
        devices=1
    )
 
    trainer.fit(model, datamodule=data_module)
 
if __name__ == '__main__':
    run()