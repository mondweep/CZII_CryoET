import os
import torch
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import glob

from ..core.model import CryoET3DCNN
from ..core.data_loader import CryoETDataset

def idx_to_particle_type(idx):
    """Convert model output index to particle type name"""
    particle_types = [
        'apo-ferritin',
        'ribosome', 
        'thyroglobulin',
        'beta-galactosidase',
        'virus-like-particle',
        'beta-amylase'
    ]
    return particle_types[idx]

def test_model(model_path):
    """Test the trained model and generate submission file"""
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
        raw_path=os.getenv('TEST_DATASET_PATH'),
        annotated_path=None  # No annotations needed for test data
    )
    
    # Get all test experiments
    test_experiments = dataset.get_experiment_ids()
    print(f"Found {len(test_experiments)} test experiments")
    
    # Initialize list to store all predictions
    all_predictions = []
    prediction_id = 0
    
    # Process each experiment
    for exp_id in test_experiments:
        print(f"\nProcessing experiment: {exp_id}")
        
        # Load tomogram
        tomogram = dataset.load_tomogram(exp_id)
        if tomogram is None:
            print(f"Skipping {exp_id} - Could not load tomogram")
            continue
        
        # Define patch size (same as training)
        patch_size = 64
        stride = 16  # Smaller stride for denser scanning
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Slide through the tomogram
        for z in range(0, tomogram.shape[0] - patch_size + 1, stride):
            for y in range(0, tomogram.shape[1] - patch_size + 1, stride):
                for x in range(0, tomogram.shape[2] - patch_size + 1, stride):
                    # Extract patch
                    patch = tomogram[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    
                    # Normalize patch
                    patch = (patch - patch.mean()) / (patch.std() + 1e-6)
                    patch_tensor = torch.FloatTensor(patch).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = model(patch_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted = probabilities.max(1)
                        
                        # If confident prediction, add to results
                        if confidence.item() > 0.3:  # Lower threshold for more detections
                            prediction = {
                                'id': prediction_id,
                                'experiment': exp_id,
                                'particle_type': idx_to_particle_type(predicted.item()),
                                'x': x + patch_size//2,  # Center coordinates
                                'y': y + patch_size//2,
                                'z': z + patch_size//2
                            }
                            all_predictions.append(prediction)
                            prediction_id += 1
                    
                    # Clear GPU memory
                    del patch_tensor
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"Found {len(all_predictions) - prediction_id} predictions in {exp_id}")
    
    # Create submission DataFrame
    if not all_predictions:
        print("\nWarning: No predictions were made. This might indicate:")
        print("1. Confidence threshold might be too high (currently 0.5)")
        print("2. Model might not be detecting any particles")
        print("3. Input processing might need adjustment")
        
        # Create empty DataFrame with correct columns
        submission_df = pd.DataFrame(columns=['id', 'experiment', 'particle_type', 'x', 'y', 'z'])
    else:
        submission_df = pd.DataFrame(all_predictions)
        submission_df = submission_df[['id', 'experiment', 'particle_type', 'x', 'y', 'z']]
    
    # Save submission file
    submission_path = 'submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved as: {submission_path}")
    print(f"Total predictions: {len(submission_df)}")
    print("\nSample of predictions:")
    print(submission_df.head())
    
    return submission_df

def find_latest_checkpoint():
    models_dir = 'models'
    checkpoint_files = []
    for root, dirs, files in os.walk(models_dir):
        for file in files:
            if file.startswith('checkpoint_epoch_') and file.endswith('.pth'):
                checkpoint_files.append(os.path.join(root, file))
    
    if not checkpoint_files:
        raise FileNotFoundError("No model checkpoints found!")
    
    # Sort by epoch number and timestamp
    latest_checkpoint = sorted(checkpoint_files)[-1]
    return latest_checkpoint

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Look for checkpoints in the models directory
    checkpoint_pattern = os.path.join(project_root, 'models', 'checkpoint_epoch_*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        print(f"No model checkpoints found in {os.path.join(project_root, 'models')}!")
        print("Please ensure model checkpoint files exist in the models directory.")
        exit(1)
    
    # Test the latest checkpoint
    latest_checkpoint = max(checkpoint_files)
    print(f"\nUsing latest checkpoint: {os.path.basename(latest_checkpoint)}")
    
    try:
        submission_df = test_model(latest_checkpoint)
    except Exception as e:
        print(f"Error testing model: {str(e)}")
        import traceback
        traceback.print_exc()