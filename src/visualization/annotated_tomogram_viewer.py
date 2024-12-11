import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ..core.data_loader import CryoETDataset
from dotenv import load_dotenv, find_dotenv
import zarr
import napari
import json
import numpy as np

# Debug: Print current working directory and location of .env file
print(f"Current working directory: {os.getcwd()}")
print(f"Found .env file at: {find_dotenv()}")

# Force reload of .env file
load_dotenv(find_dotenv(), override=True)

# Get both dataset paths
dataset_path = os.getenv('DATASET_PATH')
annotated_path = os.getenv('ANNOTATED_DATASET_PATH')

# Print paths and verify they exist
print(f"\nDataset path: {dataset_path}")
print(f"Dataset path exists: {os.path.exists(dataset_path)}")
print(f"Annotations path: {annotated_path}")
print(f"Annotations path exists: {os.path.exists(annotated_path)}")

# Define paths for tomogram data
zarr_paths = {
    "denoised": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "denoised.zarr"),
    "isonetcorrected": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "isonetcorrected.zarr"),
    "ctfdeconvolved": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "ctfdeconvolved.zarr"),
    "wbp": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "wbp.zarr")
}

# Define path for annotations
picks_path = os.path.join(annotated_path, "TS_86_3", "Picks")

# Create a napari viewer with 3D display mode
viewer = napari.Viewer(ndisplay=3)

# Load and display tomogram data
for name, path in zarr_paths.items():
    try:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
            
        print(f"Attempting to load {name} from {path}")
        zarr_file = zarr.open(path, mode='r')
        tomogram_data = zarr_file['0'][:]
        print(f"Successfully loaded {name} with shape: {tomogram_data.shape}")
        
        # Calculate better contrast limits
        p2, p98 = np.percentile(tomogram_data, (2, 98))
        
        # Add tomogram data to the viewer
        viewer.add_image(tomogram_data, name=name, colormap='gray')
    except Exception as e:
        print(f"Failed to load {name} from {path}: {e}")

# Load and display annotations
picks_path = os.path.join(annotated_path, "TS_86_3", "Picks")
print(f"\nLooking for annotations in: {picks_path}")

if not os.path.exists(picks_path):
    print(f"Error: Picks directory not found at {picks_path}")
else:
    json_files = [f for f in os.listdir(picks_path) if f.endswith('.json')]
    print(f"Found {len(json_files)} JSON files: {json_files}")

    for json_file in sorted(json_files):
        try:
            json_path = os.path.join(picks_path, json_file)
            print(f"\nProcessing {json_file}...")
            
            with open(json_path, 'r') as f:
                data = json.load(f)
                protein_name = json_file.replace('.json', '')
                print(f"Loaded data for {protein_name}")
                
                # Extract points from the JSON structure
                if 'points' in data:
                    points_list = []
                    for point in data['points']:
                        if 'location' in point:
                            loc = point['location']
                            # Convert from angstroms to voxels (divide by 10)
                            x = loc['x'] / 10
                            y = loc['y'] / 10
                            z = loc['z'] / 10
                            points_list.append([z, y, x])  # napari expects ZYX order
                    
                    points = np.array(points_list)
                    print(f"Found {len(points)} points for {protein_name}")
                    
                    if len(points) > 0:
                        viewer.add_points(
                            points,
                            name=protein_name,
                            size=10,
                            face_color=np.random.rand(3),
                            opacity=0.7,
                            symbol='disc'
                        )
                        print(f"Added points layer for {protein_name}")
                    else:
                        print(f"No valid points found for {protein_name}")
                else:
                    print(f"No points found in {json_file}")
                
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
            import traceback
            traceback.print_exc()

napari.run()