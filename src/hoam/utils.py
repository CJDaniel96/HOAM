import torch
from typing import List, Type
from pathlib import Path
from .models.hoam import HOAM, HOAMV2
 
 
def load_model(
    model_structure: str,
    model_path: str,
    embedding_size: int
) -> torch.nn.Module:
    """
    Load a pretrained model for inference.
 
    Args:
        model_structure: Name of the model class (e.g., 'HOAM', 'HOAMV2').
        model_path: Path to the saved model state_dict (.pt file).
        embedding_size: Embedding dimension used at training.
 
    Returns:
        Initialized model in eval mode.
    """
    model_classes: dict[str, Type[torch.nn.Module]] = {
        'HOAM': HOAM,
        'HOAMV2': HOAMV2,
    }
    if model_structure not in model_classes:
        raise ValueError(f"Unsupported model structure: {model_structure}")
    model_cls = model_classes[model_structure]
    # instantiate with same args as training
    model = model_cls(embedding_size=embedding_size)
    state = torch.load(model_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state)
    model.eval()
    model.cuda()
    return model
 
 
class UnNormalize:
    """
    Unnormalize a tensor image using mean and std.
    """
    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = mean
        self.std = std
 
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor