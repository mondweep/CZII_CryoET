# AWS Tools

A collection of Python tools for AWS account management and monitoring.

## Setup

1. Clone the repository: 
bash
git clone [your-repo-url]
cd aws-tools

2. Create a virtual environment:
```bash
python -m venv aws-tools-env
source aws-tools-env/bin/activate  On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your AWS credentials:
```bash
AWS_ACCESS_KEY_ID=your_access_key_id
AWS_SECRET_ACCESS_KEY=your_secret_access_key
AWS_DEFAULT_REGION=your_region
```

## Features

- AWS Permissions checking
- Free Tier usage monitoring
- Cost Explorer integration

## Usage

Example usage:
```bash
python check_free_tier_services.py
```


# Cryo-ET Particle Detection

This project implements a solution for detecting and classifying particles in cryo-electron tomography (cryo-ET) data. The goal is to identify specific particle types in 3D tomographic data with high recall accuracy.

## Project Overview

### Objective
Detect and classify particles in cryo-ET data with the following characteristics:
- 5 particles of interest with different weights
- Emphasis on recall over precision (F-beta score with β=4)
- Handling of both "easy" and "hard" particle types

### Particle Types and Weights
- **Easy Particles** (Weight: 1)
  - Ribosome
  - Virus-like particles
  - Apo-ferritin

- **Hard Particles** (Weight: 2)
  - Thyroglobulin
  - β-galactosidase

- **Non-scored Particles** (Weight: 0)
  - Beta-amylase

## Data Structure

### Data Paths
DATASET_PATH=/.../train/static/ExperimentRuns # Raw training data
ANNOTATED_DATASET_PATH=/.../train/overlay/ExperimentRuns # Annotated training data
TEST_DATASET_PATH=/.../test/static/ExperimentRuns # Test data


### File Structure
Each experiment contains:
- Raw tomogram data in Zarr format
- Multiple processing stages (denoised, isonetcorrected, ctfdeconvolved, wbp)
- Annotation files for training data

## Solution Components

### 1. Data Loading (`data_loader.py`)
- Handles Zarr file reading
- Creates training patches
- Implements data augmentation
- Manages positive and negative sample generation

### 2. Model Training (`train_model.py`)
- Implements 3D CNN architecture
- Handles model training and validation
- Includes weighted loss function
- Supports GPU acceleration

### 3. Prediction (`predict.py`)
- Processes test data
- Generates predictions in required format
- Implements post-processing steps

## Submission Format

csv
id,experiment,particle_type,x,y,z
0,TS_5_4,beta-amylase,2983.596,3154.13,764.124
1,TS_5_4,beta-galactosidase,2983.596,3154.13,764.124



## Evaluation Metric

The model is evaluated using F-beta score with β=4, which heavily prioritizes recall over precision. A particle is considered correctly identified if it lies within 0.5× the particle's radius.

### Scoring Formula
F-beta = (1 + beta²) (precision recall) / (beta² precision + recall)
where beta = 4



## Installation and Usage

1. Install required packages:
bash
pip install torch numpy pandas zarr scikit-learn


2. Set up environment variables:
bash
export DATASET_PATH=/path/to/training/data
export ANNOTATED_DATASET_PATH=/path/to/annotated/data
export TEST_DATASET_PATH=/path/to/test/data


3. Train the model:
bash
python train_model.py


4. Generate predictions:
bash
python predict.py


## Potential Improvements

1. **Data Augmentation**
   - Random rotations
   - Noise injection
   - Contrast adjustment

2. **Model Architecture**
   - Try ResNet3D or UNet3D
   - Implement attention mechanisms
   - Use pretrained models

3. **Training Optimizations**
   - Learning rate scheduling
   - Cross-validation
   - Model ensembling
   - Mixed precision training

4. **Post-processing**
   - Non-maximum suppression
   - Confidence thresholding
   - Particle clustering

## Contributing

Feel free to submit issues and enhancement requests!

## License

[MIT](LICENSE)

Update (end of day Tue Dec 10)
# CryoET Particle Visualization Project

This project provides tools for loading and visualizing CryoET (Cryo-Electron Tomography) data with particle annotations.

## Project Structure

```
CZII_CryoET/
├── data_loader.py      # Core data loading functionality
├── test_loading.py     # Visualization and testing scripts
└── README.md          # This documentation
```

## Features Implemented

### Data Loading (`data_loader.py`)
- Created `CryoETDataset` class for handling tomogram and annotation data
- Supports multiple tomogram processing stages:
  - denoised
  - isonetcorrected
  - wbp (Weighted Back Projection)
  - ctfdeconvolved
- Loads particle annotations from JSON files
- Handles multiple particle types:
  - apo-ferritin (46 particles)
  - ribosome (31 particles)
  - thyroglobulin (30 particles)
  - beta-galactosidase (12 particles)
  - virus-like-particle (11 particles)
  - beta-amylase (10 particles)

### Visualization (`test_loading.py`)
- Interactive visualization of tomogram slices with particle annotations
- Features:
  - 2D slice view with particle markers
  - Particle distribution histogram along Z-axis
  - Different marker sizes for different particle types
  - Color-coded particle annotations
  - Contrast adjustment for better visibility
  - Particle counts in legend

## Usage

1. Set up environment variables:
```bash
DATASET_PATH=/path/to/tomograms
ANNOTATED_DATASET_PATH=/path/to/annotations
```

2. Run the visualization:
```bash
python test_loading.py
```

3. View specific tomogram slices:
```python
# In test_loading.py
visualize_tomogram_with_annotations(tomogram, annotations, slice_idx=92)  # View slice 92
```

## Data Format

### Tomograms
- 3D volumes with dimensions (184, 630, 630)
- Stored in .zarr format
- Multiple processing stages available

### Annotations
- Stored in JSON format
- Structure per particle type:
  ```json
  {
    "pickable_object_name": "particle_type",
    "points": [
      {
        "location": {
          "x": float,
          "y": float,
          "z": float
        }
      }
    ]
  }
  ```

## Visualization Guide

The visualization shows two panels:
1. Left Panel: Tomogram slice with particle annotations
   - Gray-scale image showing one Z-slice of the tomogram
   - Colored X markers indicating different particle types
   - Legend showing particle types and counts

2. Right Panel: Particle distribution
   - Histogram showing particle distribution along Z-axis
   - Color-coded by particle type
   - Red dashed line indicating current slice position

## Dependencies
- Python 3.x
- numpy
- matplotlib
- pandas
- zarr (for tomogram loading)
- python-dotenv (for environment variables)
```

## Future Improvements
- Add support for additional tomogram formats
- Implement 3D visualization capabilities
- Add particle size analysis tools
- Enhance annotation tools
```

Would you like me to add or modify any sections of this documentation?

