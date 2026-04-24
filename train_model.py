"""
Advanced training script with comprehensive logging, monitoring, and optimization.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from src.data_processor import create_dataloaders, analyze_dataset
from src.medical_model import create_model, save_checkpoint, ModelTrainer
from src.infer import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping implementation."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, metric: float, model: nn.Module) -> bool:
        """Check if training should stop."""
        if metric < self.best_metric - self.min_delta:
            self.best_metric = metric
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class AdvancedTrainer:
    """Advanced training class with comprehensive features."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        class_names: List[str],
        device: torch.device,
        config: Dict,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device
        self.config = config
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup monitoring
        self.writer = SummaryWriter(log_dir=f"runs/{config['experiment_name']}")
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 15),
            min_delta=config.get('min_delta', 0.001)
        )
        
        # Metrics tracking
        self.train_history = {"loss": [], "accuracy": []}
        self.val_history = {"loss": [], "accuracy": []}
        self.best_val_accuracy = 0.0
        self.best_model_path = None
        
        # Model evaluator
        self.evaluator = ModelEvaluator(self.model, device, class_names)
        
        # Initialize wandb if configured
        if config.get('use_wandb', False):
            import wandb
            wandb.init(
                project=config.get('wandb_project', 'chest-xray-classification'),
                name=config['experiment_name'],
                config=config
            )
            wandb.watch(self.model, log='all')
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optimizer_name = self.config.get('optimizer', 'adamw')
        lr = self.config.get('learning_rate', 1e-4)
        weight_decay = self.config.get('weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('momentum', 0.9)
            return optim.SGD(self.model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        scheduler_name = self.config.get('scheduler', 'cosine')
        
        if scheduler_name.lower() == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=self.config.get('min_lr', 1e-6)
            )
        elif scheduler_name.lower() == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function."""
        loss_name = self.config.get('loss', 'crossentropy')
        
        if loss_name.lower() == 'crossentropy':
            # Use class weights if available
            class_weights = self.config.get('class_weights')
            if class_weights:
                weights = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
                return nn.CrossEntropyLoss(weight=weights)
            else:
                return nn.CrossEntropyLoss()
        elif loss_name.lower() == 'focalloss':
            return self._focal_loss
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
    
    def _focal_loss(self, inputs, targets, alpha=1, gamma=2):
        """Focal loss implementation."""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config['epochs']}",
            leave=False
        )
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip'):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100.0 * correct_predictions / total_samples
            progress_bar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
            
            # Log batch metrics
            global_step = epoch * len(self.train_loader) + batch_idx
            self.writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            
            if self.config.get('use_wandb'):
                import wandb
                wandb.log({'train/batch_loss': loss.item()}, step=global_step)
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for data, targets in tqdm(self.val_loader, desc="Validating", leave=False):
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += targets.size(0)
                correct_predictions += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100.0 * correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def train(self) -> Dict:
        """Main training loop."""
        logger.info(f"Starting training for {self.config['epochs']} epochs")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        
        start_time = time.time()
        
        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # Scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            logger.info(
                f"Epoch {epoch+1}/{self.config['epochs']} - "
                f"Time: {epoch_time:.2f}s - "
                f"LR: {current_lr:.2e} - "
                f"Train Loss: {train_loss:.4f} - "
                f"Train Acc: {train_acc:.2f}% - "
                f"Val Loss: {val_loss:.4f} - "
                f"Val Acc: {val_acc:.2f}%"
            )
            
            # TensorBoard logging
            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Wandb logging
            if self.config.get('use_wandb'):
                import wandb
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/accuracy': train_acc,
                    'val/loss': val_loss,
                    'val/accuracy': val_acc,
                    'learning_rate': current_lr,
                    'epoch_time': epoch_time
                })
            
            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self.best_model_path = self._save_checkpoint(epoch, val_loss, val_acc, is_best=True)
                logger.info(f"New best model saved with validation accuracy: {val_acc:.2f}%")
            
            # Regular checkpoint
            if (epoch + 1) % self.config.get('checkpoint_interval', 10) == 0:
                self._save_checkpoint(epoch, val_loss, val_acc, is_best=False)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        total_training_time = time.time() - start_time
        logger.info(f"Training completed in {total_training_time:.2f} seconds")
        
        # Final evaluation
        final_results = self._final_evaluation()
        
        # Close logging
        self.writer.close()
        if self.config.get('use_wandb'):
            import wandb
            wandb.finish()
        
        return {
            'best_val_accuracy': self.best_val_accuracy,
            'best_model_path': self.best_model_path,
            'training_time': total_training_time,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'final_results': final_results
        }
    
    def _save_checkpoint(self, epoch: int, val_loss: float, val_acc: float, is_best: bool = False) -> str:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        if is_best:
            filename = checkpoint_dir / f"{self.config['experiment_name']}_best.pth"
        else:
            filename = checkpoint_dir / f"{self.config['experiment_name']}_epoch_{epoch+1}.pth"
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_loss,
            accuracy=val_acc,
            filepath=str(filename),
            metadata={
                'config': self.config,
                'class_names': self.class_names,
                'train_history': self.train_history,
                'val_history': self.val_history,
            }
        )
        
        return str(filename)
    
    def _final_evaluation(self) -> Dict:
        """Perform final evaluation on test set."""
        if self.test_loader is None:
            return {}
        
        logger.info("Performing final evaluation on test set...")
        
        # Load best model if available
        if self.best_model_path and Path(self.best_model_path).exists():
            checkpoint = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from {self.best_model_path}")
        
        # Evaluate on test set
        test_results = self.evaluator.evaluate_dataset(self.test_loader)
        
        logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
        if test_results['auc'] is not None:
            logger.info(f"Test AUC: {test_results['auc']:.4f}")
        
        # Save confusion matrix
        cm_path = Path(self.config.get('output_dir', '.')) / f"{self.config['experiment_name']}_confusion_matrix.png"
        self.evaluator.plot_confusion_matrix(test_results['confusion_matrix'], str(cm_path))
        
        return test_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train chest X-ray classification model")
    
    # Data arguments
    parser.add_argument('--data_csv', type=str, required=True, help='Path to dataset CSV file')
    parser.add_argument('--image_dir', type=str, help='Directory containing images')
    parser.add_argument('--image_size', type=int, default=224, help='Image size for training')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, default='resnet50', 
                       choices=['resnet50', 'efficientnet_b0', 'densenet121', 'mobilenet_v3'],
                       help='Model architecture')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'plateau', 'none'])
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    
    # Regularization
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--grad_clip', type=float, help='Gradient clipping max norm')
    
    # Experiment
    parser.add_argument('--experiment_name', type=str, required=True, help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # Monitoring
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='chest-xray-classification', help='Wandb project name')
    
    # Early stopping
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--min_delta', type=float, default=0.001, help='Minimum improvement for early stopping')
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config_path = Path(args.output_dir) / f"{args.experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Analyze dataset
    logger.info("Analyzing dataset...")
    dataset_stats = analyze_dataset(args.data_csv)
    logger.info(f"Dataset statistics: {dataset_stats}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        csv_path=args.data_csv,
        image_dir=args.image_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        use_weighted_sampling=True,
    )
    
    # Create model
    logger.info(f"Creating {args.architecture} model...")
    model = create_model(
        architecture=args.architecture,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout_rate=args.dropout,
    )
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Add class weights to config
    if hasattr(train_loader.dataset, 'get_class_weights'):
        class_weights = train_loader.dataset.get_class_weights().tolist()
        config['class_weights'] = class_weights
        logger.info(f"Using class weights: {class_weights}")
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        class_names=class_names,
        device=device,
        config=config,
    )
    
    # Start training
    results = trainer.train()
    
    # Save final results
    results_path = Path(args.output_dir) / f"{args.experiment_name}_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif key == 'final_results' and isinstance(value, dict):
            # Handle nested numpy arrays in final_results
            serializable_final = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    serializable_final[k] = v.tolist()
                else:
                    serializable_final[k] = v
            serializable_results[key] = serializable_final
        else:
            serializable_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Training completed! Best validation accuracy: {results['best_val_accuracy']:.2f}%")
    logger.info(f"Results saved to {results_path}")
    logger.info(f"Best model saved to {results['best_model_path']}")

if __name__ == "__main__":
    main()
