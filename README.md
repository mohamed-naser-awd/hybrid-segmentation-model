# Hybrid Image Segmentation Project

This project implements a hybrid image segmentation algorithm that combines convolutional neural networks (CNNs) with transformer architectures. The model uses a DarkNet53 backbone, Feature Pyramid Network (FPN), and a Transformer decoder to perform instance segmentation on images.

## Project Overview

The project is designed to segment objects in images using a hybrid approach that leverages:
- **DarkNet53 Backbone**: For extracting multi-scale features from images
- **Feature Pyramid Network (FPN)**: For combining features at different scales
- **Transformer Decoder**: For generating segmentation masks using query-based detection

## Project Structure

### Core Model Files

#### `network.py`
The main network architecture file containing the `HybirdSegmentationAlgorithm` class. This is the primary model that combines all components:
- **Backbone**: DarkNet53 for feature extraction
- **FPN**: Feature Pyramid Network for multi-scale feature fusion
- **Patchify Layer**: Converts feature maps to tokens for the transformer
- **Transformer Decoder**: Processes queries and generates segmentation predictions
- **Outputs**: 
  - `pred_logits`: Class predictions for each query (shape: B, Q, num_classes+1)
  - `pred_masks`: Segmentation masks for each query (shape: B, Q, H, W)

#### `backbone.py`
Implements the DarkNet53 backbone architecture:
- **ResidualBlock**: Basic residual block with 1x1 and 3x3 convolutions
- **ResidualStage**: A stage containing multiple residual blocks with downsampling
- **DarkNet53**: The main backbone that extracts features at three scales (c3, c4, c5)

#### `fpn.py`
Implements the Feature Pyramid Network:
- Combines features from different scales (c3, c4, c5) from the backbone
- Uses upsampling and concatenation to create multi-scale feature maps
- Outputs three pyramid levels (p3, p4, p5) with unified channel dimensions

### Data Handling

#### `dataset.py`
Dataset classes for loading and preprocessing image segmentation data:
- **P3M10kDataset**: Loads images and masks from the P3M-10k dataset
  - Supports custom image and mask directories
  - Resizes images and masks to a specified size (default: 640x640)
  - Handles both JPG and PNG image formats
- **P3MMemmapDataset**: Efficient dataset loader using memory-mapped files
  - Loads preprocessed data from `.mmap` files for faster I/O
  - Supports multi-worker data loading
  - Used for training with combined datasets
- **PetSegmentationDataset**: Legacy dataset wrapper for Oxford-IIIT Pet dataset (for reference)

#### Datasets Used
The project uses two main datasets:
1. **P3M-10k Dataset**: 
   - Training images: `dataset/P3M-10k/train/blurred_image`
   - Training masks: `dataset/P3M-10k/train/mask`
   - Validation images: `dataset/P3M-10k/validation/P3M-500-P/blurred_image`
   - Validation masks: `dataset/P3M-10k/validation/P3M-500-P/mask`
2. **Supervisely Person Clean 2667 Dataset**:
   - Training images: `dataset/supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/images`
   - Training masks: `dataset/supervisely_person_clean_2667_img/supervisely_person_clean_2667_img/masks`

The datasets are combined and preprocessed into memory-mapped files (`.mmap`) for efficient training using the `set_data_set.py` script.

### Training and Testing

#### `train_p3.py`
Main training script for the segmentation model using P3M-10k and Supervisely datasets:
- **`train_p3m10k()`**: Full training pipeline:
  - Uses `P3MMemmapDataset` for efficient data loading
  - Supports training and validation splits
  - Uses AdamW optimizer with CrossEntropyLoss for classification and BCEWithLogitsLoss for masks
  - Implements mixed precision training with GradScaler
  - Saves best model checkpoint based on validation loss
  - Training dataset: ~12,088 samples (combined P3M-10k and Supervisely)
  - Validation dataset: 500 samples from P3M-500-P

#### `train.py`
Legacy training script for single image overfitting (for testing):
- **`load_single_sample()`**: Loads a single image-mask pair and preprocesses them
- **`train_on_single_image()`**: Trains the model on a single image (overfitting):
  - Loads image and mask
  - Initializes the model with specified parameters
  - Uses AdamW optimizer with CrossEntropyLoss for classification and BCEWithLogitsLoss for masks
  - Trains for a specified number of steps
  - Saves the trained model checkpoint

#### `test.py`
Testing script for evaluating trained models:
- Loads a pre-trained model checkpoint
- Tests the model on a single image
- Measures inference time for different components (backbone, FPN, decoder)
- Uses segmentation functions to generate and save segmented images
- Runs multiple inference passes for performance testing

### Segmentation and Export

#### `segement.py`
Contains functions for post-processing model outputs:
- **`segment_class_on_image()`**: 
  - Takes model outputs (pred_logits, pred_masks) and an image
  - Filters queries by class ID and confidence score
  - Combines multiple masks for the same class using max operation
  - Applies thresholding to create binary masks
  - Returns segmented image and binary mask
  - Note: The mask is inverted (1 - combined_mask) so the object is visible and background is black

- **`save_segmented_image()`**: 
  - Saves segmented images to disk
  - Handles different tensor shapes (with/without batch dimension)
  - Converts tensors to PIL images and saves them

#### `export.py`
Utility script for exporting sample data:
- Loads a sample from the PetSegmentationDataset
- Exports the first image and mask to the `export/` directory
- Used to prepare test data for training and inference

### Main Entry Point

#### `main.py`
Simple example script showing how to use the model:
- Loads an image
- Runs inference
- Segments a specific class
- Saves the segmented result

## Usage

### Training the Model

For full training on the combined datasets:

```bash
python train_p3.py
```

This will:
1. Load training data from memory-mapped files (`train_640_fp16_images.mmap` and `train_640_fp16_masks.mmap`)
2. Train the model for multiple epochs with validation
3. Save the best model checkpoint to `hybrid_seg_p3m10k.pt`

You can modify the training parameters:
- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Batch size (default: 20)
- `lr`: Learning rate (default: 1e-4)
- `num_workers`: Number of data loading workers (default: 4)

For single image overfitting (testing):

```bash
python train.py
```

This will:
1. Load `export/image.png` and `export/mask.png`
2. Train the model for 100 steps (default)
3. Save the model to `hybrid_seg_single_overfit.pt`

### Testing a Trained Model

```bash
python test.py
```

This will:
1. Load the trained model from `hybrid_seg_single_overfit.pt`
2. Run inference on `export/image.png`
3. Generate and save `segmented_image.png`
4. Print timing information for each component

### Preparing Datasets

Before training, you need to prepare the datasets using `set_data_set.py`:

```bash
python set_data_set.py
```

This script:
1. Combines images from P3M-10k and supervisely_person_clean_2667_img datasets
2. Resizes all images and masks to 640x640
3. Converts them to memory-mapped files (`.mmap`) for efficient loading
4. Creates `train_640_fp16_images.mmap` and `train_640_fp16_masks.mmap` for training
5. Creates `val_640_fp16_images.mmap` and `val_640_fp16_masks.mmap` for validation

### Exporting Sample Data

```bash
python export.py
```

This will export sample data to the `export/` directory for testing purposes.

## Model Architecture Details

### Input Processing
- Images are resized to 640x640 pixels
- Images are normalized to [0, 1] range

### Feature Extraction Pipeline
1. **Backbone (DarkNet53)**: Extracts features at three scales
   - c3: 256 channels, 80x80 spatial size
   - c4: 512 channels, 40x40 spatial size
   - c5: 1024 channels, 20x20 spatial size

2. **FPN**: Combines multi-scale features
   - Upsamples and concatenates features
   - Outputs unified 256-channel features at p3 level

3. **Patchify**: Converts feature maps to tokens
   - Uses 2x2 convolution with stride 2
   - Reduces spatial dimensions and projects to d_model (default: 384)

4. **Transformer Decoder**: 
   - Uses 100 learnable query embeddings
   - 6-layer decoder with 6 attention heads
   - Generates class predictions and mask features

5. **Mask Generation**:
   - Uses einsum to combine query features with pixel features
   - Bilinear upsampling to original image size

### Output Format
- **pred_logits**: (B, 100, num_classes+1) - Class predictions including background
- **pred_masks**: (B, 100, H, W) - Segmentation masks for each query

## Dependencies

- PyTorch
- torchvision
- PIL (Pillow)
- numpy

## File Organization

```
train/
├── network.py          # Main model architecture
├── backbone.py         # DarkNet53 backbone
├── fpn.py             # Feature Pyramid Network
├── dataset.py        # Dataset loading
├── train_p3.py        # Main training script (P3M-10k + Supervisely)
├── train.py           # Legacy single image training script
├── set_data_set.py    # Dataset preprocessing and memmap creation
├── test.py            # Testing script
├── segement.py        # Segmentation post-processing
├── export.py          # Data export utility
├── main.py            # Example usage
├── export/            # Exported test data
│   ├── image.png
│   └── mask.png
├── hybrid_seg_single_overfit.pt  # Trained model checkpoint
└── segmented_image.png           # Output segmentation result
```

## Notes

- The model is designed for foreground/background segmentation (person/object vs background)
- Training uses combined datasets: P3M-10k (~10k images) and Supervisely Person Clean 2667 (~2.6k images)
- The segmentation output inverts the mask so the object is visible (background is black)
- Model uses CUDA if available, otherwise falls back to CPU
- All images are processed at 640x640 resolution
- Data is preprocessed into memory-mapped files for efficient loading during training
- Training dataset contains ~12,088 combined samples
- Validation uses 500 samples from P3M-500-P subset

## Future Improvements

- Add support for multi-class segmentation
- Implement proper training loop with validation
- Add data augmentation
- Support for different image sizes
- Batch processing support

