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
    
    print("\nStarting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
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
            batch_count += 1
            
            # Print progress indicator
            if batch_count % 5 == 0:  # Show progress every 5 batches
                print("🔄", end='', flush=True)
        
        avg_train_loss = train_loss / batch_count
        print(f"\nAverage training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        print("Validating: ", end='', flush=True)
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
                print("✓", end='', flush=True)
                
        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        print(f"\nValidation Loss: {avg_val_loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f'model_checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    return model

if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), 'cryo_et_model.pth') 