from data_loader import CryoETDataset
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

def visualize_tomogram_with_annotations(tomogram, annotations, slice_idx=None, slice_range=5):
    """Visualize a tomogram slice with nearby annotations"""
    if slice_idx is None:
        slice_idx = tomogram.shape[0] // 2
    
    plt.figure(figsize=(20, 10))
    
    # Create main plot
    plt.subplot(1, 2, 1)
    plt.imshow(tomogram[slice_idx], cmap='gray', vmin=-5, vmax=2)  # Adjust contrast
    plt.title(f'Tomogram Slice {slice_idx}', fontsize=12)
    plt.colorbar(label='Intensity')
    
    # Plot annotations near this slice
    nearby_annotations = annotations[
        (annotations['z'] >= slice_idx - slice_range) & 
        (annotations['z'] <= slice_idx + slice_range)
    ]
    
    # Plot different particle types with different colors and sizes
    particle_sizes = {
        'virus-like-particle': 200,  # Larger marker for bigger particles
        'ribosome': 150,
        'thyroglobulin': 150,
        'beta-galactosidase': 100,
        'apo-ferritin': 100,
        'beta-amylase': 100
    }
    
    for particle_type in nearby_annotations['particle_type'].unique():
        mask = nearby_annotations['particle_type'] == particle_type
        plt.scatter(
            nearby_annotations[mask]['x'],
            nearby_annotations[mask]['y'],
            label=f"{particle_type} ({sum(mask)})",  # Add count to legend
            alpha=0.8,
            marker='x',
            s=particle_sizes.get(particle_type, 100),
            linewidth=2
        )
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Add histogram of particle z-positions
    plt.subplot(1, 2, 2)
    for particle_type in annotations['particle_type'].unique():
        plt.hist(
            annotations[annotations['particle_type'] == particle_type]['z'],
            bins=30,
            orientation='horizontal',
            alpha=0.5,
            label=particle_type
        )
    
    plt.axhline(y=slice_idx, color='r', linestyle='--', label='Current slice')
    plt.ylabel('Z position')
    plt.xlabel('Count')
    plt.title('Particle Distribution along Z-axis')
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

def main():
    print("Starting test script...")
    
    # Initialize dataset
    dataset = CryoETDataset(
        raw_path=os.getenv('DATASET_PATH'),
        annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
    )
    
    # Get first experiment
    exp_ids = dataset.get_experiment_ids()
    if exp_ids:
        test_id = exp_ids[0]
        print(f"\nTesting with experiment: {test_id}")
        
        # Load data
        tomogram = dataset.load_tomogram(test_id, processing_stage='isonetcorrected')
        annotations = dataset.load_annotations(test_id)
        
        if tomogram is not None and annotations is not None:
            print("\nVisualizing middle slice...")
            visualize_tomogram_with_annotations(tomogram, annotations)
            
            # Find and show the slice with the most particles
            z_counts = annotations['z'].value_counts()
            busy_slice = z_counts.index[0]
            print(f"\nVisualizing busy slice {busy_slice} with {z_counts[busy_slice]} particles...")
            visualize_tomogram_with_annotations(tomogram, annotations, slice_idx=busy_slice)
            
            # Print some statistics
            print("\nParticle statistics:")
            print(f"Total particles: {len(annotations)}")
            print("\nParticles per type:")
            print(annotations['particle_type'].value_counts())
            print("\nZ-range: {:.1f} to {:.1f}".format(
                annotations['z'].min(),
                annotations['z'].max()
            ))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()