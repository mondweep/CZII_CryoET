import os
import torch
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import glob

from ..core.model import CryoET3DCNN
from ..core.data_loader import CryoETDataset

def test_model(model_path):
    """Test the trained model and visualize results"""
    print(f"\nTesting model: {model_path}")
    print("Loading model...")
    
    # Load the checkpoint dictionary
    checkpoint = torch.load(model_path, weights_only=True)
    
    # Initialize model and load state
    model = CryoET3DCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test data
    print("Loading test data...")
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    X, y = dataset.create_training_patches()
    
    # Process test data in batches
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    batch_size = 32
    all_predictions = []
    
    print("Making predictions in batches...")
    for i in range(0, len(X), batch_size):
        batch_end = min(i + batch_size, len(X))
        print(f"Processing batch {i//batch_size + 1}/{len(X)//batch_size + 1}")
        
        # Process batch
        batch_X = torch.FloatTensor(X[i:batch_end]).unsqueeze(1).to(device)
        
        with torch.no_grad():
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            all_predictions.extend(predicted.cpu().numpy())
        
        # Clear GPU memory
        del batch_X
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Convert predictions to numpy array
    y_pred = np.array(all_predictions)
    y_true = y
    
    # Print classification report
    particle_types = [
        'apo-ferritin',
        'ribosome', 
        'thyroglobulin',
        'beta-galactosidase',
        'virus-like-particle',
        'background'
    ]
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=particle_types))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=particle_types,
                yticklabels=particle_types)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('confusion_matrix.png')
    print("\nConfusion matrix saved as 'confusion_matrix.png'")
    
    # Print additional training information
    print(f"\nTraining information from checkpoint:")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get project root directory (two levels up from this file)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Look for checkpoints in the models directory
    checkpoint_pattern = os.path.join(project_root, 'models', 'model_checkpoint_epoch_*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No model checkpoints found in {os.path.join(project_root, 'models')}!")
        print("Please ensure model checkpoint files exist in the models directory.")
        exit(1)
    
    print(f"Found {len(checkpoint_files)} model checkpoints:")
    for f in checkpoint_files:
        print(f"- {os.path.basename(f)}")
    
    # Test the latest checkpoint
    latest_checkpoint = max(checkpoint_files)
    print(f"\nUsing latest checkpoint: {os.path.basename(latest_checkpoint)}")
    
    try:
        test_model(latest_checkpoint)
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()