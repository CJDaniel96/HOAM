import click
from pathlib import Path
from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf
from .train import run as hydra_run
from .evaluate import evaluate_model_on_testset
from .inference import main as inference_main, parse_opt as parse_infer_opt
from .knn import build_knn_index
from .visualize import create_eval_charts
from .data.statistics import DataStatistics
from .data.transforms import build_transforms
from .utils import load_model
from torchvision.datasets import ImageFolder
import torch
 
 
@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    HOAM Project CLI: train, evaluate, plot-eval, build-knn, infer
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
        mean, std = DataStatistics.load_mean_std(mean_std_file)
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
 
 
@cli.command("plot-eval")
@click.option("--eval-dir", "-e", type=click.Path(exists=True, file_okay=False), required=True, help="Directory containing test_metrics.json, classification_report.csv and confusion_matrix.csv")
@click.option("--output-dir", "-o", type=click.Path(), default=None, help="Directory to save charts; defaults to <eval-dir>/charts")
@click.option("--top-n", type=int, default=40, help="Maximum number of classes to show in per-class charts")
@click.option("--sort-by", type=click.Choice(["precision", "recall", "f1", "support"]), default="f1", help="Metric used to select/sort classes")
def plot_eval(eval_dir, output_dir, top_n, sort_by):  # noqa: D103
    """
    Create SVG charts and an HTML dashboard from evaluation CSV/JSON outputs.
    """
    paths = create_eval_charts(eval_dir, output_dir=output_dir, top_n=top_n, sort_by=sort_by)
    click.echo("Evaluation charts saved:")
    for path in paths:
        click.echo(f"  {path}")


def _read_config_defaults(config_file: str | None, model_path: str) -> DictConfig | None:
    config_path = Path(config_file) if config_file else Path(model_path).parent / "config_used.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    return None


@cli.command("build-knn")
@click.option("--model-path", "-m", type=click.Path(exists=True), required=True, help="Path to trained model state_dict (.pt)")
@click.option("--data-dir", "-d", type=click.Path(exists=True, file_okay=False), default=None, help="Dataset root containing train/ and val/")
@click.option("--save-dir", "-s", type=click.Path(), default=None, help="Directory to save knn.index and dataset.pkl; defaults to model checkpoint directory")
@click.option("--config-file", type=click.Path(exists=True), default=None, help="Training config_used.yaml. Defaults to <model-path>/../config_used.yaml when present")
@click.option("--model-structure", type=click.Choice(["HOAM", "HOAMV2"]), default=None, help="Model architecture. Inferred from config_used.yaml when omitted")
@click.option("--backbone", default=None, help="Backbone name used during training. Inferred from config_used.yaml when omitted")
@click.option("--embedding-size", type=int, default=None, help="Embedding dimension used during training. Inferred from config_used.yaml when omitted")
@click.option("--image-size", type=int, default=None, help="Image size used during training. Inferred from config_used.yaml when omitted")
@click.option("--mean-std-file", type=click.Path(exists=True), default=None, help="Training mean_std.json. Defaults to save-dir/model-dir mean_std.json when present")
@click.option("--index-path", default="knn.index", help="Output FAISS index filename/path")
@click.option("--dataset-pkl", default="dataset.pkl", help="Output pickled reference dataset filename/path")
@click.option("--threshold", type=float, default=0.5, help="Match threshold stored with the inference model helper")
@click.option("--batch-size", type=int, default=64, help="Batch size for embedding the reference dataset")
@click.option("--num-workers", type=int, default=0, help="Workers used only when mean/std must be computed")
def build_knn(
    model_path,
    data_dir,
    save_dir,
    config_file,
    model_structure,
    backbone,
    embedding_size,
    image_size,
    mean_std_file,
    index_path,
    dataset_pkl,
    threshold,
    batch_size,
    num_workers,
):  # noqa: D103
    """
    Build a KNN reference index from an already-trained checkpoint.
    """
    cfg = _read_config_defaults(config_file, model_path)
    model_dir = Path(model_path).parent
    save_path = Path(save_dir) if save_dir else model_dir

    if cfg is not None:
        data_dir = data_dir or cfg.data.data_dir
        model_structure = model_structure or cfg.model.structure
        backbone = backbone or cfg.model.backbone
        embedding_size = embedding_size or cfg.model.embedding_size
        image_size = image_size or cfg.data.image_size

    data_dir = data_dir or None
    model_structure = model_structure or "HOAM"
    embedding_size = embedding_size or 128
    image_size = image_size or 224

    if not data_dir:
        raise click.UsageError("Missing --data-dir and no data.data_dir found in config_used.yaml.")

    if mean_std_file is None:
        candidates = [save_path / "mean_std.json", model_dir / "mean_std.json"]
        mean_std_file = next((str(path) for path in candidates if path.exists()), None)

    index_file, dataset_file, mean_std_path = build_knn_index(
        model_structure=model_structure,
        model_path=model_path,
        data_dir=data_dir,
        save_dir=save_path,
        embedding_size=embedding_size,
        image_size=image_size,
        mean_std_file=mean_std_file,
        backbone_name=backbone,
        index_path=index_path,
        dataset_pkl=dataset_pkl,
        threshold=threshold,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    click.echo(f"KNN index saved to {index_file}")
    click.echo(f"Reference dataset saved to {dataset_file}")
    click.echo(f"Mean/std file: {mean_std_path}")


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
