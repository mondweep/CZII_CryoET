import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..core.data_loader import CryoETDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from ..core.model import CryoET3DCNN
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

def train_model():
    # Initialize dataset
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    
    # Get training data
    X, y = dataset.create_training_patches()
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X).unsqueeze(1)  # Add channel dimension
    y = torch.LongTensor(y)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize model
    model = CryoET3DCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(
        weight=torch.FloatTensor([1, 1, 1, 2, 2, 1]).to(device)  # weights for all 6 classes
    )
    optimizer = optim.Adam(model.parameters())
    
    # Define particle types
    particle_types = [
        'apo-ferritin', 'ribosome', 'thyroglobulin',
        'beta-galactosidase', 'virus-like-particle', 'beta-amylase'
    ]

    # Initialize metrics tracking
    metrics_history = {
        'epoch': [], 'train_loss': [], 'val_loss': [],
        'train_precision': [], 'train_recall': [], 'train_f1': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }

    print("\nStarting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_predictions = []
        train_targets = []
        batch_count = 0
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("Training batches: ", end='', flush=True)
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_predictions.extend(predicted.cpu().numpy())
            train_targets.extend(batch_y.cpu().numpy())
            batch_count += 1
            
            if batch_count % 5 == 0:
                print("🔄", end='', flush=True)
        
        # Calculate training metrics
        train_metrics_df, (train_prec, train_rec, train_f1) = calculate_metrics(
            train_targets, train_predictions, particle_types
        )
        
        avg_train_loss = train_loss / batch_count
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        print("\nTraining Metrics:")
        print(train_metrics_df)
        
        # Validation
        model.eval()
        val_loss = 0
        val_predictions = []
        val_targets = []
        print("Validating: ", end='', flush=True)
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = outputs.max(1)
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
                print("✓", end='', flush=True)
        
        # Calculate validation metrics
        val_metrics_df, (val_prec, val_rec, val_f1) = calculate_metrics(
            val_targets, val_predictions, particle_types
        )
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Store metrics
        metrics_history['epoch'].append(epoch + 1)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['train_precision'].append(train_prec)
        metrics_history['train_recall'].append(train_rec)
        metrics_history['train_f1'].append(train_f1)
        metrics_history['val_precision'].append(val_prec)
        metrics_history['val_recall'].append(val_rec)
        metrics_history['val_f1'].append(val_f1)
        
        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        print("\nValidation Metrics:")
        print(val_metrics_df)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = f'models/checkpoint_epoch_{epoch+1}_{timestamp}.pth'
            
            # Create models directory if it doesn't exist
            os.makedirs('models', exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics_history': metrics_history,
                'train_metrics_df': train_metrics_df.to_dict(),
                'val_metrics_df': val_metrics_df.to_dict(),
            }, checkpoint_path)
            
            # Save metrics history to CSV
            metrics_df = pd.DataFrame(metrics_history)
            metrics_df.to_csv(f'models/metrics_history_{timestamp}.csv', index=False)
            
            print(f"Saved checkpoint and metrics: {checkpoint_path}")
    
    return model, metrics_history

if __name__ == "__main__":
    model, metrics_history = train_model()
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(metrics_history['epoch'], metrics_history['train_loss'], label='Train Loss')
    plt.plot(metrics_history['epoch'], metrics_history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot precision
    plt.subplot(1, 3, 2)
    plt.plot(metrics_history['epoch'], metrics_history['train_precision'], label='Train Precision')
    plt.plot(metrics_history['epoch'], metrics_history['val_precision'], label='Val Precision')
    plt.title('Precision History')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot F1 Score
    plt.subplot(1, 3, 3)
    plt.plot(metrics_history['epoch'], metrics_history['train_f1'], label='Train F1')
    plt.plot(metrics_history['epoch'], metrics_history['val_f1'], label='Val F1')
    plt.title('F1 Score History')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.close()
    
    # Save final model
    torch.save(model.state_dict(), 'models/cryo_et_model.pth')
  