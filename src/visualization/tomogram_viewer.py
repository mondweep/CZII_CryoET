import os
from dotenv import load_dotenv, find_dotenv
import zarr
import napari

# Debug: Print current working directory and location of .env file
print(f"Current working directory: {os.getcwd()}")
print(f"Found .env file at: {find_dotenv()}")

# Force reload of .env file
load_dotenv(find_dotenv(), override=True)

# Get the dataset path from the environment variable
dataset_path = os.getenv('DATASET_PATH')
print(f"Dataset path from .env: {dataset_path}")

# Verify the dataset path exists
if not os.path.exists(dataset_path):
    print(f"WARNING: Dataset path does not exist: {dataset_path}")
else:
    print(f"Contents of dataset path: {os.listdir(dataset_path)}")

# Update paths - using absolute paths for clarity
zarr_paths = {
    "denoised": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "denoised.zarr"),
    "isonetcorrected": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "isonetcorrected.zarr"),
    "ctfdeconvolved": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "ctfdeconvolved.zarr"),
    "wbp": os.path.join(dataset_path, "TS_86_3", "VoxelSpacing10.000", "wbp.zarr")
}

# Debug: Check each component of the path
for name, path in zarr_paths.items():
    print(f"\nChecking path components for {name}:")
    current_path = dataset_path
    print(f"Base path exists: {os.path.exists(current_path)}")
    
    for component in ["TS_86_3", "VoxelSpacing10.000"]:
        current_path = os.path.join(current_path, component)
        print(f"After adding {component}: {current_path}")
        print(f"Path exists: {os.path.exists(current_path)}")
        if os.path.exists(current_path):
            print(f"Contents: {os.listdir(current_path)}")

# Create a napari viewer with 3D display mode
viewer = napari.Viewer(ndisplay=3)

# Load and display each processing stage
for name, path in zarr_paths.items():
    try:
        if not os.path.exists(path):
            print(f"Path does not exist: {path}")
            continue
            
        print(f"Attempting to load {name} from {path}")
        zarr_file = zarr.open(path, mode='r')
        tomogram_data = zarr_file['0'][:] #['0] is looking at the highest resolution image layer. The Zarr files are there at 0, 1 and 2 levels
        print(f"Successfully loaded {name} with shape: {tomogram_data.shape}")
        viewer.add_image(
            tomogram_data,
            name=name,
            contrast_limits=[tomogram_data.min(), tomogram_data.max()],
            rendering='mip',  # Maximum intensity projection
            opacity=0.5  # Semi-transparent for better 3D visualization
        )
    except Exception as e:
        print(f"Error accessing {name}: {str(e)}")

napari.run()