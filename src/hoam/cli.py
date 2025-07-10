import click
from pathlib import Path
from hydra import initialize, compose
from omegaconf import DictConfig
from .train import run as hydra_run
from .evaluate import evaluate_model_on_testset
from .inference import main as inference_main, parse_opt as parse_infer_opt
from .data.statistics import DataStatistics
from .data.transforms import build_transforms
from torchvision.datasets import ImageFolder
import torch
 
 
@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    HOAM Project CLI: train, evaluate, infer
    """
    pass
 
 
@cli.command()
@click.option(
    "--config-dir",
    default="configs",
    type=click.Path(exists=True, file_okay=False),
    help="Directory containing Hydra configuration files"
)
@click.option(
    "--config-name",
    default="config",
    help="Base name of the Hydra config (without .yaml extension)"
)
def train(config_dir: str, config_name: str):  # noqa: D103
    """
    Train a model using Hydra config + PyTorch Lightning.
    """
    config_dir = Path(config_dir)
    with initialize(config_path=str(config_dir), job_name="hoam_train"):
        cfg: DictConfig = compose(config_name=config_name)
        hydra_run(cfg)
 
 
@cli.command()
@click.option("--model-path", "-m", type=click.Path(exists=True), required=True, help="Path to model checkpoint (.pt)")
@click.option("--test-data", "-d", type=click.Path(exists=True), required=True, help="Directory containing test images")
@click.option("--save-dir", "-s", type=click.Path(), required=True, help="Directory to save evaluation outputs")
@click.option("--batch-size", type=int, default=64, help="Batch size for evaluation")
def evaluate(model_path: str, test_data: str, save_dir: str, batch_size: int):  # noqa: D103
    """
    Evaluate a trained model on a test dataset.
    """
    # Load mean/std from cache or compute
    data_dir = Path(test_data)
    mean, std = DataStatistics.get_mean_std(data_dir, image_size=None)
 
    # Build dataset and dataloader
    transforms = build_transforms('val', image_size=None, mean=mean, std=std)
    test_ds = ImageFolder(str(data_dir), transforms)
 
    # Load model
    model = torch.load(model_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device).eval()
 
    # Run evaluation
    evaluate_model_on_testset(model, test_ds, save_dir, batch_size, device)
    click.echo(f"Evaluation results saved to {save_dir}")
 
 
@cli.command(context_settings={'ignore_unknown_options': True})
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
def infer(args):  # noqa: D103
    """
    Run inference (knn or match mode). Pass-through options to inference script.
    Example:
      hoam infer --mode knn --dataset-pkl dataset.pkl --model-path model.pt --data imgs/ --save-dir out/ --k 5
    """
    # Parse and delegate
    opt = parse_infer_opt(known=True)
    # overwrite parsed opts with CLI args
    import sys
    sys.argv = ['infer'] + list(args)
    opt = parse_infer_opt(known=False)
    inference_main(opt)
 
 
if __name__ == '__main__':
    cli()