import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging

from data_loader import CryoETDataLoader
from data_preprocessing import CryoETPreprocessor
from model import CryoETNet, FocalLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryoETDataset(Dataset):
    def __init__(self, data_loader, preprocessor, exp_runs, transform=None):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.transform = transform
        
        self.patches = []
        self.labels = []
        self.protein_types = set()
        
        # Load all data
        self._load_data(exp_runs)
        
        # Convert protein types to indices
        self.protein_to_idx = {protein: idx for idx, protein in enumerate(sorted(self.protein_types))}
        self.idx_to_protein = {idx: protein for protein, idx in self.protein_to_idx.items()}
        
    def _load_data(self, exp_runs):
        for exp_run in exp_runs:
            try:
                # Load tomogram
                tomogram = self.data_loader.load_tomogram(exp_run)
                tomogram_data = np.array(tomogram)
                
                # Preprocess tomogram
                tomogram_data = self.preprocessor.normalize_tomogram(tomogram_data)
                tomogram_data = self.preprocessor.denoise_tomogram(tomogram_data)
                
                # Load labels
                labels_dict = self.data_loader.load_labels(exp_run)
                
                # Process each protein type
                for protein_type, coordinates in labels_dict.items():
                    self.protein_types.add(protein_type)
                    
                    # Extract patches for positive samples
                    patches, valid_coords = self.preprocessor.extract_patches(
                        tomogram_data, coordinates['coordinates'])
                    
                    self.patches.extend(patches)
                    self.labels.extend([protein_type] * len(valid_coords))
                    
            except Exception as e:
                logger.error(f"Error processing {exp_run}: {str(e)}")
                
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        label = self.protein_to_idx[self.labels[idx]]
        
        if self.transform:
            patch = self.transform(patch)
            
        return torch.FloatTensor(patch).unsqueeze(0), torch.tensor(label)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        logger.info(f'Epoch {epoch+1}: '
                   f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.3f}% | '
                   f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.3f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize components
    data_loader = CryoETDataLoader("/kaggle/input/czii-cryo-et-object-identification")
    preprocessor = CryoETPreprocessor(patch_size=64)
    
    # Get list of experiment runs
    train_runs = ["TS_6_4", "TS_6_6", "TS_86_3"]  # Add more runs as needed
    
    # Create datasets
    dataset = CryoETDataset(data_loader, preprocessor, train_runs)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    model = CryoETNet(num_classes=len(dataset.protein_types))
    criterion = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
