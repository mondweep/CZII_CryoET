import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..core.data_loader import CryoETDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

class CryoET3DCNN(nn.Module):
    def __init__(self, num_classes=6):  # Keep as 6 total classes (5 particles + background)
        super(CryoET3DCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def calculate_metrics(y_true, y_pred, particle_types):
    """Calculate precision, recall, and f1-score for each class"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    metrics_df = pd.DataFrame({
        'Particle Type': particle_types,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # Also calculate averaged metrics
    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    
    return metrics_df, (avg_precision, avg_recall, avg_f1)

def train_model(model=None, optimizer=None, start_epoch=0, num_epochs=50, metrics_history=None):
    if model is None:
        model = CryoET3DCNN()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters())
    
    if metrics_history is None:
        metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [],
            'train_precision': [], 'train_recall': [], 'train_f1': [],
            'val_precision': [], 'val_recall': [], 'val_f1': []
        }
    
    # Initialize dataset
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    
    # Get training data and create data loaders
    X, y = dataset.create_training_patches()
    
    # Debug prints
    print("\nDebugging create_training_patches output:")
    print(f"Type of X: {type(X)}")
    print(f"Type of y: {type(y)}")
    if hasattr(X, 'shape'):
        print(f"Shape of X: {X.shape}")
    else:
        print(f"X does not have shape attribute. Content sample: {X[:2]}")
    if hasattr(y, 'shape'):
        print(f"Shape of y: {y.shape}")
    else:
        print(f"y does not have shape attribute. Content sample: {y[:2]}")
    
    # Convert numpy arrays to tensors if they aren't already
    if not isinstance(X, torch.Tensor):
        X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension here
    if not isinstance(y, torch.Tensor):
        y = torch.LongTensor(y)
    
    # Print shapes for verification
    print(f"After initial conversion - X shape: {X.shape}, y shape: {y.shape}")
    
    X_train, X_val, y_train, y_val = train_test_split(X.numpy(), y.numpy(), test_size=0.2)
    
    # Convert split results back to tensors
    X_train = torch.FloatTensor(X_train)  # Should already have channel dimension
    X_val = torch.FloatTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    # Print shapes for verification
    print(f"Final X_train shape: {X_train.shape}")
    print(f"Final y_train shape: {y_train.shape}")
    
    # After creating data loaders
    print("Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    print("Setting up train and validation loaders...")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print("Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Moving model to device...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    print("Starting training loop...")
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        # Training phase
        model.train()
        train_loss = 0
        batch_count = 0
        print("Training phase...")
        for batch_X, batch_y in train_loader:
            print(f"Processing batch {batch_count+1}", end='\r')
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_count += 1
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_name = f'checkpoint_epoch_{epoch+1}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics_history': metrics_history,
            }, f'models/{checkpoint_name}')
    
    return model, metrics_history

# Make sure to export the function
__all__ = ['CryoET3DCNN', 'train_model']
  