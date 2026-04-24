"""
Advanced data processing pipeline for medical images.
Includes augmentations, preprocessing, and dataset utilities.
"""

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging

logger = logging.getLogger(__name__)

class XRayDataset(Dataset):
    """Advanced chest X-ray dataset with comprehensive augmentations."""
    
    def __init__(
        self,
        csv_path: str,
        image_dir: Optional[str] = None,
        split: str = "train",
        image_size: Tuple[int, int] = (224, 224),
        augment: bool = True,
        normalize: bool = True,
        preprocessing_fn: Optional[Callable] = None,
    ):
        super().__init__()
        
        self.csv_path = Path(csv_path)
        self.image_dir = Path(image_dir) if image_dir else None
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == "train")
        self.normalize = normalize
        self.preprocessing_fn = preprocessing_fn
        
        # Load data
        self.data = self._load_data()
        self.classes = sorted(self.data['label'].unique().tolist())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
        # Create transforms
        self.transform = self._create_transforms()
        
        logger.info(f"Dataset initialized: {len(self.data)} samples, {self.num_classes} classes")
    
    def _load_data(self) -> pd.DataFrame:
        """Load and filter data based on split."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
        
        df = pd.read_csv(self.csv_path)
        
        # Simple split logic (you can enhance this)
        if 'split' in df.columns:
            df = df[df['split'] == self.split].reset_index(drop=True)
        else:
            # Create splits based on ratios
            if self.split == "train":
                df = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
            elif self.split == "val":
                df = df.drop(df.sample(frac=0.8, random_state=42).index)
                df = df.sample(frac=0.5, random_state=42).reset_index(drop=True)
            else:  # test
                df = df.drop(df.sample(frac=0.8, random_state=42).index)
                df = df.drop(df.sample(frac=0.5, random_state=42).index).reset_index(drop=True)
        
        return df
    
    def _create_transforms(self) -> A.Compose:
        """Create albumentations transforms."""
        transforms_list = []
        
        # Resize
        transforms_list.append(A.Resize(self.image_size[0], self.image_size[1]))
        
        if self.augment:
            # Geometric augmentations
            transforms_list.extend([
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.5
                ),
                A.ElasticTransform(
                    alpha=1,
                    sigma=50,
                    alpha_affine=50,
                    p=0.3
                ),
            ])
            
            # Intensity augmentations
            transforms_list.extend([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.2),
            ])
            
            # Medical-specific augmentations
            transforms_list.extend([
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
                A.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.2),
            ])
        
        # Normalization
        if self.normalize:
            transforms_list.append(
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225],   # ImageNet stds
                    max_pixel_value=255.0,
                )
            )
        
        transforms_list.append(ToTensorV2())
        
        return A.Compose(transforms_list)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get item by index."""
        row = self.data.iloc[idx]
        
        # Load image
        image_path = row['path']
        if self.image_dir:
            image_path = self.image_dir / image_path
        else:
            image_path = Path(image_path)
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image = np.array(image)
            
            # Apply preprocessing
            if self.preprocessing_fn:
                image = self.preprocessing_fn(image)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            # Get label
            label = self.class_to_idx[row['label']]
            
            return image_tensor, label
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return dummy data
            dummy_image = torch.zeros(3, *self.image_size)
            return dummy_image, 0
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced training."""
        class_counts = self.data['label'].value_counts()
        total_samples = len(self.data)
        
        weights = []
        for cls in self.classes:
            weight = total_samples / (len(self.classes) * class_counts[cls])
            weights.append(weight)
        
        return torch.FloatTensor(weights)
    
    def get_sample_weights(self) -> List[float]:
        """Get sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights()
        sample_weights = []
        
        for idx in range(len(self.data)):
            label = self.data.iloc[idx]['label']
            class_idx = self.class_to_idx[label]
            sample_weights.append(class_weights[class_idx].item())
        
        return sample_weights

def preprocess_image(
    image: Image.Image,
    size: Tuple[int, int] = (224, 224),
    normalize: bool = True,
) -> torch.Tensor:
    """
    Preprocess single image for inference.
    
    Args:
        image: PIL Image
        size: Target size
        normalize: Whether to normalize
        
    Returns:
        Preprocessed image tensor
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy
    image_np = np.array(image)
    
    # Create transforms
    transforms_list = [
        A.Resize(size[0], size[1]),
    ]
    
    if normalize:
        transforms_list.append(
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            )
        )
    
    transforms_list.append(ToTensorV2())
    
    transform = A.Compose(transforms_list)
    
    # Apply transforms
    transformed = transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def create_dataloaders(
    csv_path: str,
    image_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (224, 224),
    use_weighted_sampling: bool = True,
    **kwargs
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        csv_path: Path to CSV file
        image_dir: Directory containing images
        batch_size: Batch size
        num_workers: Number of worker processes
        image_size: Image size
        use_weighted_sampling: Use weighted sampling for training
        **kwargs: Additional dataset arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    # Create datasets
    train_dataset = XRayDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        split="train",
        image_size=image_size,
        augment=True,
        **kwargs
    )
    
    val_dataset = XRayDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        split="val",
        image_size=image_size,
        augment=False,
        **kwargs
    )
    
    test_dataset = XRayDataset(
        csv_path=csv_path,
        image_dir=image_dir,
        split="test",
        image_size=image_size,
        augment=False,
        **kwargs
    )
    
    # Create samplers
    train_sampler = None
    if use_weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes

class MedicalImageAugmentor:
    """Specialized augmentor for medical images."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def add_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Add medical imaging artifacts."""
        if np.random.random() < self.p:
            # Add grid pattern (common in X-rays)
            h, w = image.shape[:2]
            grid = np.zeros_like(image)
            
            # Vertical lines
            for i in range(0, w, 20):
                grid[:, i:i+1] = 255
            
            # Horizontal lines  
            for i in range(0, h, 20):
                grid[i:i+1, :] = 255
                
            image = cv2.addWeighted(image, 0.95, grid, 0.05, 0)
        
        return image
    
    def simulate_positioning_error(self, image: np.ndarray) -> np.ndarray:
        """Simulate patient positioning errors."""
        if np.random.random() < self.p:
            # Random perspective transform
            h, w = image.shape[:2]
            
            # Define source points
            src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            
            # Add random offset to destination points
            offset = int(min(h, w) * 0.05)
            dst_points = src_points + np.random.randint(-offset, offset, src_points.shape).astype(np.float32)
            
            # Apply perspective transform
            matrix = cv2.getPerspectiveTransform(src_points, dst_points)
            image = cv2.warpPerspective(image, matrix, (w, h))
        
        return image

def analyze_dataset(csv_path: str) -> Dict:
    """Analyze dataset statistics."""
    df = pd.read_csv(csv_path)
    
    analysis = {
        "total_samples": len(df),
        "num_classes": df['label'].nunique(),
        "class_distribution": df['label'].value_counts().to_dict(),
        "class_balance": df['label'].value_counts(normalize=True).to_dict(),
    }
    
    # Check for missing images
    missing_images = []
    for idx, row in df.iterrows():
        image_path = Path(row['path'])
        if not image_path.exists():
            missing_images.append(str(image_path))
    
    analysis["missing_images"] = len(missing_images)
    analysis["missing_image_paths"] = missing_images[:10]  # First 10
    
    return analysis
