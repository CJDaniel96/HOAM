import json

import click
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig
from .train import run as hydra_run
from .evaluate import evaluate_model_on_testset
from .inference import main as inference_main, parse_opt as parse_infer_opt
from .data.statistics import DataStatistics
from .data.transforms import build_transforms
from .utils import load_model
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
    # Resolve to an absolute path: initialize_config_dir avoids hydra.initialize's
    # "relative to the caller module" rule, which previously pointed at src/hoam/configs.
    config_dir = Path(config_dir).resolve()
    with initialize_config_dir(config_dir=str(config_dir), job_name="hoam_train"):
        cfg: DictConfig = compose(config_name=config_name)
        hydra_run(cfg)
 
 
@cli.command()
@click.option("--model-path", "-m", type=click.Path(exists=True), required=True, help="Path to model checkpoint (.pt)")
@click.option("--test-data", "-d", type=click.Path(exists=True), required=True, help="Directory containing test images")
@click.option("--save-dir", "-s", type=click.Path(), required=True, help="Directory to save evaluation outputs")
@click.option("--model-structure", type=click.Choice(["HOAM", "HOAMV2"]), default="HOAM", help="Model architecture to instantiate")
@click.option("--embedding-size", type=int, default=128, help="Embedding dimension used at training")
@click.option("--image-size", type=int, default=224, help="Image size for resizing")
@click.option("--mean-std-file", type=click.Path(exists=True), default=None, help="JSON of training mean/std; recommended for reproducible normalization")
@click.option("--batch-size", type=int, default=64, help="Batch size for evaluation")
@click.option("--knn-k", type=int, default=1, help="k for the k-NN classifier used in classification metrics")
def evaluate(model_path, test_data, save_dir, model_structure, embedding_size, image_size, mean_std_file, batch_size, knn_k):  # noqa: D103
    """
    Evaluate a trained model on a test dataset.
    """
    # Load mean/std from the supplied training-stats file, else fall back to
    # computing/caching from the dataset directory.
    data_dir = Path(test_data)
    if mean_std_file:
        with open(mean_std_file) as f:
            stats = json.load(f)
        mean, std = stats['mean'], stats['std']
    else:
        mean, std = DataStatistics.get_mean_std(data_dir, image_size=image_size)
 
    # Build dataset and dataloader
    transforms = build_transforms('val', image_size=image_size, mean=mean, std=std)
    test_ds = ImageFolder(str(data_dir), transforms)

    # Load model (instantiate architecture, then load the saved state_dict)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_structure, model_path, embedding_size, device=device)
 
    # Run evaluation
    evaluate_model_on_testset(model, test_ds, save_dir, batch_size, device, knn_k=knn_k)
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