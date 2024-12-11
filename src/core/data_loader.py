import os
import numpy as np
import pandas as pd
import zarr
from dotenv import load_dotenv
from pathlib import Path

class CryoETDataset:
    def __init__(self, raw_path, annotated_path):
        load_dotenv()
        self.raw_path = Path(raw_path) if raw_path else None
        self.annotated_path = Path(annotated_path) if annotated_path else None
        
        # Define particle types and their indices
        self.particle_types = [
            'apo-ferritin',      # index 0
            'ribosome',          # index 1
            'thyroglobulin',     # index 2
            'beta-galactosidase',# index 3
            'virus-like-particle'# index 4
        ]
        
    def load_tomogram(self, experiment_id, processing_stage='denoised'):
        """
        Load a single tomogram
        Args:
            experiment_id: e.g., 'TS_5_4'
            processing_stage: one of ['denoised', 'isonetcorrected', 'wbp', 'ctfdeconvolved']
        """
        path = self.raw_path / experiment_id / "VoxelSpacing10.000" / f"{processing_stage}.zarr"
        try:
            if not path.exists():
                raise FileNotFoundError(f"No zarr file found at {path}")
                
            print(f"Loading tomogram from {path}")
            zarr_file = zarr.open(str(path), mode='r')
            return zarr_file['0'][:]  # Load full resolution data
            
        except Exception as e:
            print(f"Error loading tomogram {experiment_id}: {str(e)}")
            return None
        
    def load_annotations(self, experiment_id):
        """Load particle annotations for a given experiment"""
        if not self.annotated_path:
            return None
            
        picks_dir = self.annotated_path / experiment_id / "Picks"
        print(f"Looking for annotations in: {picks_dir}")
        
        annotations = []
        try:
            if picks_dir.exists():
                print(f"Found Picks directory. Contents: {list(picks_dir.glob('*'))}")
                
                # Process each JSON file
                for json_file in picks_dir.glob("*.json"):
                    print(f"\nReading JSON file: {json_file}")
                    particle_type = json_file.stem  # Get filename without extension
                    
                    with open(json_file, 'r') as f:
                        import json
                        data = json.load(f)
                        print(f"Found {len(data.get('points', []))} points in {particle_type}")
                        
                        # Extract points from the JSON structure
                        if 'points' in data:
                            for point in data['points']:
                                if 'location' in point:
                                    loc = point['location']
                                    annotations.append({
                                        'particle_type': particle_type,
                                        'x': loc['x'],
                                        'y': loc['y'],
                                        'z': loc['z']
                                    })
                
                if annotations:
                    df = pd.DataFrame(annotations)
                    print(f"\nSuccessfully loaded {len(df)} total annotations")
                    print("\nParticle distribution:")
                    print(df['particle_type'].value_counts())
                    
                    # Scale coordinates to match tomogram dimensions
                    scale_factor = 10.0  # VoxelSpacing10.000
                    df['x'] = (df['x'] / scale_factor).astype(int)
                    df['y'] = (df['y'] / scale_factor).astype(int)
                    df['z'] = (df['z'] / scale_factor).astype(int)
                    
                    return df
                else:
                    print("No valid annotations found in JSON files")
            else:
                print(f"Picks directory does not exist: {picks_dir}")
            return None
            
        except Exception as e:
            print(f"Error loading annotations for {experiment_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_experiment_ids(self):
        """Get list of all experiment IDs"""
        return [d.name for d in self.raw_path.iterdir() 
                if d.is_dir() and d.name.startswith('TS_')]
    
    def create_training_patches(self):
        """Create training patches from tomograms and annotations"""
        X = []  # patches
        y = []  # labels
        
        total_experiments = len(list(self.get_experiment_ids()))
        for exp_idx, exp_id in enumerate(self.get_experiment_ids(), 1):
            print(f"\nProcessing experiment: {exp_id} ({exp_idx}/{total_experiments})")
            
            # Load tomogram and annotations
            tomogram = self.load_tomogram(exp_id, 'isonetcorrected')
            annotations = self.load_annotations(exp_id)
            
            if tomogram is None or annotations is None:
                continue
            
            # Create patches around each particle
            patch_size = 64
            half_size = patch_size // 2
            
            # Process each particle type
            for particle_type in self.particle_types:
                particle_annotations = annotations[annotations['particle_type'] == particle_type]
                print(f"Processing {particle_type}: {len(particle_annotations)} particles")
                
                for idx, (_, row) in enumerate(particle_annotations.iterrows(), 1):
                    if idx % 5 == 0:  # Show progress every 5 particles
                        print(".", end='', flush=True)
                    
                    x, y_pos, z = int(row['x']), int(row['y']), int(row['z'])
                    
                    # Check if we can extract a full patch
                    if (x-half_size >= 0 and x+half_size < tomogram.shape[2] and
                        y_pos-half_size >= 0 and y_pos+half_size < tomogram.shape[1] and
                        z-half_size >= 0 and z+half_size < tomogram.shape[0]):
                        
                        # Extract patch
                        patch = tomogram[
                            z-half_size:z+half_size,
                            y_pos-half_size:y_pos+half_size,
                            x-half_size:x+half_size
                        ]
                        
                        X.append(patch)
                        y.append(self.particle_types.index(particle_type))
                
                print()  # New line after processing each particle type
            
            num_negatives = len(y) // 2
            print(f"\nAdding {num_negatives} negative samples...")
            for i in range(num_negatives):
                if i % 10 == 0:  # Show progress every 10 negative samples
                    print(".", end='', flush=True)
                # Random position
                x = np.random.randint(half_size, tomogram.shape[2]-half_size)
                y_pos = np.random.randint(half_size, tomogram.shape[1]-half_size)
                z = np.random.randint(half_size, tomogram.shape[0]-half_size)
                
                # Check if far enough from any particle
                too_close = False
                for _, row in annotations.iterrows():
                    if (abs(x-row['x']) < half_size and 
                        abs(y_pos-row['y']) < half_size and 
                        abs(z-row['z']) < half_size):
                        too_close = True
                        break
                
                if not too_close:
                    patch = tomogram[
                        z-half_size:z+half_size,
                        y_pos-half_size:y_pos+half_size,
                        x-half_size:x+half_size
                    ]
                    X.append(patch)
                    y.append(len(self.particle_types))  # Background class
                    break
            
            print(f"\nCompleted experiment {exp_id}")
        
        print(f"\nTotal patches created: {len(X)}")
        print(f"Positive samples: {len(X) - num_negatives}")
        print(f"Negative samples: {num_negatives}")
        
        return np.array(X), np.array(y)
    
    @staticmethod
    def extract_patch(tomogram, x, y, z, size):
        """Extract a cubic patch around a point"""
        half = size // 2
        try:
            patch = tomogram[
                max(0, z-half):min(tomogram.shape[0], z+half),
                max(0, y-half):min(tomogram.shape[1], y+half),
                max(0, x-half):min(tomogram.shape[2], x+half)
            ]
            if patch.shape == (size, size, size):
                return patch
        except:
            pass
        return None
    
    def generate_negative_samples(self, tomogram, annotations, size, num_samples):
        """Generate random negative samples away from annotated particles"""
        negative_patches = []
        half = size // 2
        
        # Create a set of existing particle locations
        particle_locations = set(
            (int(row['x']), int(row['y']), int(row['z'])) 
            for _, row in annotations.iterrows()
        )
        
        while len(negative_patches) < num_samples:
            # Random location
            x = np.random.randint(half, tomogram.shape[2]-half)
            y = np.random.randint(half, tomogram.shape[1]-half)
            z = np.random.randint(half, tomogram.shape[0]-half)
            
            # Check if far enough from existing particles
            if not any(self.is_close_to_particle((x,y,z), loc, threshold=size) 
                      for loc in particle_locations):
                patch = self.extract_patch(tomogram, x, y, z, size)
                if patch is not None:
                    negative_patches.append(patch)
                    
        return negative_patches
    
    @staticmethod
    def is_close_to_particle(point1, point2, threshold):
        """Check if two points are within threshold distance"""
        return sum((a-b)**2 for a,b in zip(point1, point2)) < threshold**2

if __name__ == "__main__":
    print("Testing CryoETDataset...")
    
    # Initialize dataset
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    
    # List available experiments
    exp_ids = dataset.get_experiment_ids()
    print(f"\nFound experiments: {exp_ids}")
    
    if exp_ids:
        # Test with first experiment
        test_id = exp_ids[0]
        print(f"\nTesting with experiment: {test_id}")
        
        # Test tomogram loading
        print("\nTesting tomogram loading:")
        for stage in ['denoised', 'isonetcorrected', 'wbp', 'ctfdeconvolved']:
            print(f"\nTrying to load {stage} tomogram...")
            tomogram = dataset.load_tomogram(test_id, stage)
            if tomogram is not None:
                print(f"Success! Shape: {tomogram.shape}")
                print(f"Value range: [{tomogram.min():.2f}, {tomogram.max():.2f}]")
            else:
                print(f"Failed to load {stage} tomogram")
        
        # Test annotation loading
        print("\nTesting annotation loading:")
        annotations = dataset.load_annotations(test_id)
        if annotations is not None:
            print(f"Success! Found {len(annotations)} annotations")
            print("\nParticle types:")
            print(annotations['particle_type'].value_counts())
        else:
            print("Failed to load annotations")
    else:
        print("No experiments found!")
