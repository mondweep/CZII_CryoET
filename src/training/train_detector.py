import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import json

from data_loader import CryoETDataLoader
from data_preprocessing import CryoETPreprocessor
from detection_model import CryoETDetector, DetectionLoss

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CryoETDetectionDataset(Dataset):
    def __init__(self, data_loader, preprocessor, exp_runs, patch_size=64, stride=32):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.patch_size = patch_size
        self.stride = stride
        
        self.samples = []
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
                
                # Extract overlapping patches and corresponding labels
                patches, labels = self._extract_patches_with_labels(
                    tomogram_data, labels_dict)
                
                self.samples.extend(list(zip(patches, labels)))
                
            except Exception as e:
                logger.error(f"Error processing {exp_run}: {str(e)}")
    
    def _extract_patches_with_labels(self, tomogram, labels_dict):
        patches = []
        patch_labels = []
        
        # Get dimensions
        depth, height, width = tomogram.shape
        
        # Extract overlapping patches
        for z in range(0, depth - self.patch_size + 1, self.stride):
            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    patch = tomogram[z:z+self.patch_size, 
                                   y:y+self.patch_size, 
                                   x:x+self.patch_size]
                    
                    # Get labels for this patch
                    patch_label = self._get_labels_for_patch(
                        labels_dict, x, y, z, self.patch_size)
                    
                    patches.append(patch)
                    patch_labels.append(patch_label)
        
        return patches, patch_labels
    
    def _get_labels_for_patch(self, labels_dict, x, y, z, size):
        labels = []
        
        for protein_type, data in labels_dict.items():
            self.protein_types.add(protein_type)
            
            for coord in data['coordinates']:
                cx, cy, cz = coord
                
                # Check if coordinate is within patch
                if (x <= cx < x + size and 
                    y <= cy < y + size and 
                    z <= cz < z + size):
                    
                    # Convert to patch-relative coordinates
                    rel_x = (cx - x) / size
                    rel_y = (cy - y) / size
                    rel_z = (cz - z) / size
                    
                    labels.append({
                        'class_id': self.protein_to_idx[protein_type],
                        'center': [rel_x, rel_y, rel_z],
                        'size': data.get('size', 0.1)  # Default size if not provided
                    })
        
        return labels
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        patch, labels = self.samples[idx]
        
        # Convert patch to tensor
        patch_tensor = torch.FloatTensor(patch).unsqueeze(0)
        
        # Convert labels to tensor format
        target = self._convert_labels_to_target(labels)
        
        return patch_tensor, target
    
    def _convert_labels_to_target(self, labels):
        # Convert labels to the format expected by the model
        # This is a placeholder - implement based on model requirements
        return {
            'boxes': torch.tensor([label['center'] + [label['size']] for label in labels]),
            'labels': torch.tensor([label['class_id'] for label in labels])
        }

def train_detector(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
    model = model.to(device)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}')):
            inputs = inputs.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss, loss_dict = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}: Loss={loss.item():.4f} '
                          f'(cls={loss_dict["cls_loss"]:.4f}, '
                          f'box={loss_dict["box_loss"]:.4f}, '
                          f'obj={loss_dict["obj_loss"]:.4f})')
        
        train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                outputs = model(inputs)
                loss, _ = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        
        logger.info(f'Epoch {epoch+1}: '
                   f'Train Loss: {train_loss:.4f} | '
                   f'Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_detector.pth')

if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize components
    data_loader = CryoETDataLoader("data")
    preprocessor = CryoETPreprocessor(patch_size=64)
    
    # Get list of experiment runs
    train_runs = ["TS_6_4", "TS_6_6", "TS_86_3"]  # Add more runs as needed
    
    # Create datasets
    dataset = CryoETDetectionDataset(data_loader, preprocessor, train_runs)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)
    
    # Initialize model, criterion, and optimizer
    model = CryoETDetector(num_classes=len(dataset.protein_types))
    criterion = DetectionLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    train_detector(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
