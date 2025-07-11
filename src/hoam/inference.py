import argparse
import json
import joblib
import shutil
from pathlib import Path
 
import torch
import cv2
from PIL import Image
from torchvision.utils import save_image
 
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils.inference import InferenceModel, MatchFinder
 
from .utils import load_model, UnNormalize
from .data.statistics import DataStatistics
from .data.transforms import build_transforms
 
 
def parse_opt(known: bool = False):
    """
    Parse command line arguments for inference.
 
    Args:
        known: If True, return defaults without error on unknown; else parse all args.
 
    Returns:
        Namespace of parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['knn', 'match'], default='knn')
    parser.add_argument('--data', type=str, required=True, help='Image or directory for inference')
    parser.add_argument('--query-image', type=str, default='', help='Query image for match mode')
    parser.add_argument('--dataset-pkl', type=str, default='dataset.pkl', help='Pickled dataset for KNN')
    parser.add_argument('--faiss-index', type=str, default='', help='FAISS index path')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for match mode')
    parser.add_argument('--k', type=int, default=5, help='Top-K for KNN mode')
    parser.add_argument('--mean-std-file', type=str, default='mean_std.json', help='Mean/std cache')
    parser.add_argument('--model-structure', type=str, default='HOAM', choices=['HOAM', 'HOAMV2'], help='Model to load')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model .pt file')
    parser.add_argument('--embedding-size', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save results')
    if known:
        return parser.parse_known_args()[0]
    return parser.parse_args()
 
 
def create_inference_model(
    model_structure: str,
    model_path: str,
    embedding_size: int,
    faiss_index: str,
    threshold: float,
    device: torch.device
) -> InferenceModel:
    """
    Create a PML InferenceModel with KNN or match finder.
    """
    model = load_model(model_structure, model_path, embedding_size)
    match_finder = MatchFinder(distance=CosineSimilarity(), threshold=threshold)
    inf_model = InferenceModel(model, match_finder=match_finder)
    if faiss_index:
        inf_model.load_knn_func(faiss_index)
    return inf_model
 
 
def load_dataset(dataset_pkl: str):
    """Load dataset from pickle."""
    return joblib.load(dataset_pkl)
 
 
def process_image(
    image_path: str,
    transforms,
    device: torch.device
):
    """Load and transform image to tensor."""
    img = cv2.imread(image_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    tensor = transforms(img)
    return tensor.unsqueeze(0).to(device)
 
 
def knn_inference(
    data: str,
    inf_model: InferenceModel,
    dataset,
    classes,
    unnormalize,
    k: int,
    mean,
    std,
    save_dir: Path,
    device: torch.device
):
    """Run KNN inference on single image or directory."""
    top1 = {}
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if Path(data).is_file():
        tensor, label = _knn_single(data, inf_model, dataset, classes, unnormalize, k, mean, std, save_dir, device)
        top1[Path(data).stem] = label
    else:
        for img_path in Path(data).rglob('*.[jp][pn]g'):
            subdir = save_dir / img_path.stem
            subdir.mkdir(parents=True, exist_ok=True)
            _, label = _knn_single(str(img_path), inf_model, dataset, classes, unnormalize, k, mean, std, subdir, device)
            top1[img_path.stem] = label
    (save_dir / 'top1.json').write_text(json.dumps(top1, indent=2))
    return top1
 
 
def _knn_single(path, inf_model, dataset, classes, unnormalize, k, mean, std, save_dir, device):
    tensor = process_image(path, build_transforms('test', dataset.transform.transforms[0].size, mean, std), device)
    _, indices = inf_model.get_nearest_neighbors(tensor, k)
    idx = indices[0][0]
    label = classes[idx]
    result_img, _ = dataset[idx]
    img = unnormalize(result_img)
    save_image(img, save_dir / f"{label}_top1.jpg")
    return tensor, label
 
 
def match_inference(
    data: str,
    query: str,
    inf_model: InferenceModel,
    mean,
    std,
    save_dir: Path,
    device: torch.device
):
    """Run match inference single or directory."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    def single(src):
        t1 = process_image(src, build_transforms('test', None, mean, std), device)
        t2 = process_image(query, build_transforms('test', None, mean, std), device)
        is_match = inf_model.is_match(t1, t2)
        sub = save_dir / ('OK' if is_match else 'NG')
        sub.mkdir(exist_ok=True)
        shutil.copy(src, sub/Path(src).name)
    if Path(data).is_file():
        single(data)
    else:
        for img in Path(data).rglob('*.[jp][pn]g'):
            single(str(img))
 
 
def main(opt):
    """Entry point for inference."""
    mean_std_file = Path(opt.mean_std_file)
    mean, std = DataStatistics.get_mean_std(mean_std_file.parent, image_size=None)
    unnormalize = UnNormalize(mean, std)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    inf_model = create_inference_model(
        opt.model_structure,
        opt.model_path,
        opt.embedding_size,
        opt.faiss_index,
        opt.threshold,
        device
    )
 
    if opt.mode == 'knn':
        dataset = load_dataset(opt.dataset_pkl)
        classes = dataset.classes
        knn_inference(opt.data, inf_model, dataset, classes, unnormalize, opt.k, mean, std, Path(opt.save_dir), device)
    else:
        match_inference(opt.data, opt.query_image, inf_model, mean, std, Path(opt.save_dir), device)
 
 
if __name__ == '__main__':
    opt = parse_opt()
    main(opt)