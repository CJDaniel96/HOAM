import os
import torch
import shutil
import joblib
from pathlib import Path
from hydra import main
from omegaconf import DictConfig, OmegaConf
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.samplers import MPerClassSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
 
# Use Lightning umbrella (v2+) instead of pytorch_lightning
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
 
from .models.hoam import HOAM, HOAMV2
from .losses.hybrid_margin import HybridMarginLoss
from pytorch_metric_learning.losses import SubCenterArcFaceLoss, ArcFaceLoss
from .data.transforms import build_transforms
from .data.statistics import DataStatistics
 
# Dynamically set float32 matmul precision to leverage Tensor Cores when available
if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    precision = 'high' if major >= 8 else 'medium'
    torch.set_float32_matmul_precision(precision)
    

def build_sampler(dataset: ImageFolder, m_per_class: int, batch_size: int) -> MPerClassSampler:
    return MPerClassSampler(
        labels=[y for _, y in dataset.samples],
        m=m_per_class,
        batch_size=batch_size,
        length_before_new_iter=len(dataset) 
    )
 
 
class HOAMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        image_size: int,
        batch_size: int,
        num_workers: int
    ) -> None:
        super().__init__()
        import platform
        cpu_count = os.cpu_count() or 1
        # On Windows, use single worker to avoid spawn issues
        if platform.system() == "Windows":
            self.num_workers = 0
        else:
            self.num_workers = min(num_workers or cpu_count, cpu_count)
 
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
 
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
        sampler = build_sampler(self.train_ds, 4, self.batch_size)
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0),
            drop_last=True
        )
 
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )
 
 
class LightningModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.save_hyperparameters(cfg)
 
        # Determine classes
        train_path = Path(cfg.data.data_dir) / 'train'
        num_classes = len([d for d in train_path.iterdir() if d.is_dir()])
 
        # Model selection
        model_map = {'HOAM': HOAM, 'HOAMV2': HOAMV2}
        model_cls = model_map.get(cfg.model.structure)
        if model_cls is None:
            raise ValueError(f"Unknown model structure: {cfg.model.structure}")
        self.model = model_cls(
            backbone_name=cfg.model.backbone,
            pretrained=cfg.model.pretrained,
            embedding_size=cfg.model.embedding_size
        )
 
        # Loss selection
        loss_map = {
            'HybridMarginLoss': HybridMarginLoss,
            'SubCenterArcFaceLoss': SubCenterArcFaceLoss,
            'ArcFaceLoss': ArcFaceLoss
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
            params = {'num_classes': num_classes, 'embedding_size': cfg.model.embedding_size}
            if hasattr(cfg.loss, 'subcenter_margin'):
                params['margin'] = cfg.loss.subcenter_margin
            if hasattr(cfg.loss, 'subcenter_scale'):
                params['scale'] = cfg.loss.subcenter_scale
            if loss_type == 'SubCenterArcFaceLoss' and hasattr(cfg.loss, 'sub_centers'):
                params['sub_centers'] = cfg.loss.sub_centers
            self.criterion = loss_cls(**params)
            
        self.freeze_backbone = cfg.training.freeze_backbone_epochs
        self.ema = None
        
    def set_backbone_requies_grad(self, requires_grad: bool) -> None:
        for param in self.model.backbone.parameters():
            param.requires_grad = requires_grad
        
    def on_train_start(self):
        if self.freeze_backbone > 0:
            self.set_backbone_requies_grad(False)

    def on_train_epoch_end(self, outputs):
        if self.current_epoch == self.freeze_backbone:
            self.set_backbone_requies_grad(True)
            
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tb = self.logger.experiment
        tb.add_scalar('train_loss', avg_loss, self.current_epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
 
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        imgs, labels = batch
        loss = self.criterion(self(imgs), labels)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=imgs.size(0))
        return loss
 
    def validation_step(self, batch, batch_idx) -> None:
        imgs, labels = batch
        with torch.no_grad():
            loss = self.criterion(self(imgs), labels)
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
 
    def configure_optimizers(self):
        backbone_params = [p for n, p in self.named_parameters() if 'backbone' in n and p.requires_grad]
        head_params = [p for n, p in self.named_parameters() if 'backbone' not in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': self.hparams.training.lr * 0.1},
            {'params': head_params, 'lr': self.hparams.training.lr}
        ], weight_decay=self.hparams.training.weight_decay)
        
        epochs = self.hparams.training.epochs
        warmup = int(epochs * 0.05)
        
        def lr_lambda(current_epoch):
            if current_epoch < warmup:
                return float(current_epoch + 1) / float(warmup)
            # cosine decay to 0
            progress = (current_epoch - warmup) / max(1, epochs - warmup)
            progress_t = torch.tensor(progress)
            return 0.5 * (1. + torch.cos(torch.pi * progress_t)).item()
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}
        }
 
 
@main(
    version_base="1.3",
    config_path=str(Path(__file__).parents[2] / "configs"),
    config_name="config"
)
def run(cfg: DictConfig) -> None:
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
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        save_last='link',
        save_weights_only=True
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=cfg.training.patience, mode='min')
    swa = StochasticWeightAveraging(
        swa_lrs=[cfg.training.lr * 0.01, cfg.training.lr * 0.1],
        swa_epoch_start=0.75,
        annealing_epochs=10,
        annealing_strategy='cos'
    )
 
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        logger=logger,
        callbacks=[checkpoint, early_stop, swa]
    )
 
    trainer.fit(model, datamodule=data_module)

    # Save best and last weights
    best_ckpt = checkpoint.best_model_path
    last_ckpt = checkpoint.last_model_path
    if model:
        best_model = LightningModel.load_from_checkpoint(best_ckpt)
        torch.save(best_model.model.state_dict(), str(Path(cfg.training.checkpoint_dir) / 'best.pt'))
        
        last_model = LightningModel.load_from_checkpoint(last_ckpt)
        torch.save(last_model.model.state_dict(), str(Path(cfg.training.checkpoint_dir) / 'last.pt'))

    # Save config and mean_std
    OmegaConf.save(config=cfg, f=str(Path(cfg.training.checkpoint_dir) / 'config_used.yaml'))
    mean_src = Path(cfg.data.data_dir) / 'mean_std.json'
    if mean_src.exists():
        shutil.copy(str(mean_src), str(Path(cfg.training.checkpoint_dir) / 'mean_std.json'))
 
    # Optional KNN
    if cfg.knn.enable:
        emb_model = LightningModel.load_from_checkpoint(best_ckpt)
        emb_model.eval()
        mean, std = DataStatistics.get_mean_std(Path(cfg.data.data_dir), cfg.data.image_size)
        transforms = build_transforms('train', cfg.data.image_size, mean, std)
        dataset = ImageFolder(Path(cfg.data.data_dir) / 'train', transforms)
        match_finder = MatchFinder(distance=CosineSimilarity(), threshold=cfg.knn.threshold)
        inf_model = InferenceModel(emb_model.model, match_finder=match_finder)
        inf_model.train_knn(dataset)
        inf_model.save_knn_func(str(Path(cfg.training.checkpoint_dir) / cfg.knn.index_path))
        joblib.dump(dataset, str(Path(cfg.training.checkpoint_dir) / cfg.knn.dataset_pkl))
 
    print("Training run complete.")
 
 
if __name__ == '__main__':
    run()