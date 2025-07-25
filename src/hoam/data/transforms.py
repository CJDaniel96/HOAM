from typing import List
from torchvision import transforms
 
# Default normalization values
DEFAULT_MEAN: List[float] = [0.485, 0.456, 0.406]
DEFAULT_STD: List[float] = [0.229, 0.224, 0.225]
 
 
def build_transforms(
    mode: str,
    image_size: int,
    mean: List[float] = DEFAULT_MEAN,
    std: List[float] = DEFAULT_STD,
) -> transforms.Compose:
    """
    Build data augmentation and normalization transforms.
 
    Args:
        mode: One of 'train', 'val', 'test'. Determines augmentations.
        image_size: Target spatial size (image_size x image_size).
        mean: Channel-wise mean for normalization.
        std: Channel-wise std for normalization.
 
    Returns:
        A torchvision.transforms.Compose object.
    """
    assert mode in {'train', 'val', 'test'}, f"Unknown mode: {mode}"
 
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
 
    if mode == 'train':
        transform_list = [
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandAugment(num_ops=2, magnitude=9),
            *transform_list,
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
 
    return transforms.Compose(transform_list)