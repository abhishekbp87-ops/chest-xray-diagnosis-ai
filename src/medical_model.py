"""
Advanced medical image classification models.
Includes multiple architectures and model management utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MedicalNet(nn.Module):
    """
    Advanced medical image classification network with multiple backbone options.
    """
    
    def __init__(
        self, 
        num_classes: int = 2,
        backbone: str = "resnet50",
        pretrained: bool = True,
        dropout_rate: float = 0.5,
        use_attention: bool = True
    ):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        
        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained)
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Attention mechanism
        if use_attention:
            self.attention = SpatialAttention(self.feature_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self, backbone: str, pretrained: bool) -> nn.Module:
        """Create backbone network."""
        if backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            # Remove final layers
            return nn.Sequential(*list(model.children())[:-2])
            
        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            return model.features
            
        elif backbone == "densenet121":
            model = models.densenet121(pretrained=pretrained)
            return model.features
            
        elif backbone == "mobilenet_v3":
            model = models.mobilenet_v3_large(pretrained=pretrained)
            return model.features
            
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension of backbone."""
        with torch.no_grad():
            x = torch.randn(1, 3, 224, 224)
            features = self.backbone(x)
            return features.shape[1]
    
    def _initialize_weights(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Extract features
        features = self.backbone(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Classification
        logits = self.classifier(features)
        
        return logits
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        features = self.backbone(x)
        if self.use_attention:
            features = self.attention(features)
        return features

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for medical images."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        return x * attention

class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved accuracy."""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble."""
        outputs = []
        for model in self.models:
            outputs.append(F.softmax(model(x), dim=1))
        
        # Weighted average
        ensemble_output = sum(w * out for w, out in zip(self.weights, outputs))
        return torch.log(ensemble_output + 1e-8)  # Convert back to log probabilities

def create_model(
    architecture: str = "resnet50",
    num_classes: int = 2,
    pretrained: bool = True,
    **kwargs
) -> MedicalNet:
    """
    Factory function to create medical models.
    
    Args:
        architecture: Model architecture name
        num_classes: Number of output classes
        pretrained: Use pretrained weights
        **kwargs: Additional model parameters
    
    Returns:
        Initialized model
    """
    return MedicalNet(
        num_classes=num_classes,
        backbone=architecture,
        pretrained=pretrained,
        **kwargs
    )

def load_model(
    checkpoint_path: str,
    num_classes: int = 2,
    device: torch.device = None,
    architecture: str = "resnet50",
    **kwargs
) -> MedicalNet:
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_classes: Number of classes
        device: Device to load model on
        architecture: Model architecture
        **kwargs: Additional model parameters
    
    Returns:
        Loaded model
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model(architecture, num_classes, pretrained=False, **kwargs)
    
    # Load checkpoint
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    accuracy: float,
    filepath: str,
    metadata: Optional[Dict] = None
):
    """
    Save model checkpoint with comprehensive metadata.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Training epoch
        loss: Current loss
        accuracy: Current accuracy
        filepath: Save path
        metadata: Additional metadata
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy,
        "model_architecture": getattr(model, "backbone_name", "unknown"),
        "num_classes": getattr(model, "num_classes", 2),
        "timestamp": torch.tensor(torch.get_num_threads()),  # Simple timestamp
    }
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")

class ModelTrainer:
    """Advanced model trainer with comprehensive features."""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        criterion: nn.Module = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=1e-4)
        self.scheduler = scheduler
        self.criterion = criterion or nn.CrossEntropyLoss()
        
        self.train_history = {"loss": [], "accuracy": []}
        self.val_history = {"loss": [], "accuracy": []}
    
    def train_epoch(self, dataloader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
