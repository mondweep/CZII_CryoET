from src.core.data_loader import CryoETDataset
import os

# Test data loading
dataset = CryoETDataset(
    raw_path=os.getenv('DATASET_PATH'),
    annotated_path=os.getenv('ANNOTATED_DATASET_PATH')
)

# Load a single tomogram to verify access
test_exp = next(dataset.raw_path.iterdir()).name
print(f"Testing with experiment: {test_exp}")
tomogram = dataset.load_tomogram(test_exp)
print(f"Loaded tomogram shape: {tomogram.shape}")

# Test annotation loading
annotations = dataset.load_annotations(test_exp)
print(f"Number of annotations: {len(annotations) if annotations is not None else 0}") 