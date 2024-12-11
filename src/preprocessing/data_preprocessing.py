import numpy as np
from ..core.data_loader import CryoETDataset
from scipy.ndimage import gaussian_filter
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CryoETPreprocessor:
    def __init__(self, patch_size=64):
        """
        Initialize the preprocessor
        Args:
            patch_size (int): Size of the 3D patches to extract
        """
        self.patch_size = patch_size
        self.augmentation = self.get_augmentation_pipeline()
        
    def normalize_tomogram(self, tomogram):
        """
        Normalize the tomogram to zero mean and unit variance
        Args:
            tomogram (np.ndarray): Input tomogram
        Returns:
            np.ndarray: Normalized tomogram
        """
        mean = np.mean(tomogram)
        std = np.std(tomogram)
        return (tomogram - mean) / (std + 1e-6)
    
    def denoise_tomogram(self, tomogram, sigma=1.0):
        """
        Apply Gaussian denoising to the tomogram
        Args:
            tomogram (np.ndarray): Input tomogram
            sigma (float): Standard deviation for Gaussian kernel
        Returns:
            np.ndarray: Denoised tomogram
        """
        return gaussian_filter(tomogram, sigma=sigma)
    
    def extract_patches(self, tomogram, coordinates, patch_size=None):
        """
        Extract patches around given coordinates
        Args:
            tomogram (np.ndarray): Input tomogram
            coordinates (list): List of (x, y, z) coordinates
            patch_size (int): Size of patches to extract (optional)
        Returns:
            np.ndarray: Extracted patches
        """
        if patch_size is None:
            patch_size = self.patch_size
            
        half_size = patch_size // 2
        patches = []
        valid_coords = []
        
        for coord in coordinates:
            x, y, z = map(int, coord)
            
            # Check if patch is within bounds
            if (x - half_size >= 0 and x + half_size < tomogram.shape[0] and
                y - half_size >= 0 and y + half_size < tomogram.shape[1] and
                z - half_size >= 0 and z + half_size < tomogram.shape[2]):
                
                patch = tomogram[x-half_size:x+half_size,
                               y-half_size:y+half_size,
                               z-half_size:z+half_size]
                patches.append(patch)
                valid_coords.append(coord)
        
        return np.array(patches), valid_coords
    
    def get_augmentation_pipeline(self):
        """
        Create an augmentation pipeline for 2D slices
        Returns:
            A.Compose: Augmentation pipeline
        """
        return A.Compose([
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                A.GridDistortion(p=0.5),
                A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
            ToTensorV2(),
        ])
    
    def augment_slice(self, slice_2d):
        """
        Apply augmentation to a 2D slice
        Args:
            slice_2d (np.ndarray): Input 2D slice
        Returns:
            torch.Tensor: Augmented slice
        """
        augmented = self.augmentation(image=slice_2d)
        return augmented['image']

if __name__ == "__main__":
    # Example usage
    preprocessor = CryoETPreprocessor()
    
    # Create a dummy tomogram
    dummy_tomogram = np.random.randn(100, 100, 100)
    
    # Normalize and denoise
    normalized = preprocessor.normalize_tomogram(dummy_tomogram)
    denoised = preprocessor.denoise_tomogram(normalized)
    
    # Extract patches
    dummy_coords = [(50, 50, 50), (60, 60, 60)]
    patches, valid_coords = preprocessor.extract_patches(denoised, dummy_coords)
    
    print(f"Number of extracted patches: {len(patches)}")
    print(f"Patch shape: {patches[0].shape}")
    
    # Test augmentation on a 2D slice
    slice_2d = denoised[50, :, :]
    augmented_slice = preprocessor.augment_slice(slice_2d)
    print(f"Augmented slice shape: {augmented_slice.shape}")
