# Facial Expression Recognition using AlexNet (FER-CNN)

This repository implements a robust Facial Expression Recognition (FER) system leveraging a modified **AlexNet** architecture. Developed using **PyTorch**, the project focuses on classifying human facial expressions into multiple emotion categories by training on three benchmark datasets: CK, CK+, and the JAFFE dataset.

## Project Description
The system utilises deep Convolutional Neural Networks (CNN) to extract high-level features from facial images. By using the AlexNet backbone, the model is capable of recognising complex patterns in facial muscle movements associated with different emotions. The pipeline includes comprehensive data preprocessing, augmentation via `torchvision.transforms.v2`, and performance evaluation using confusion matrices.

## Key Features
* **Architecture:** Based on AlexNet for deep feature extraction and classification.
* **Framework:** Built entirely using PyTorch and Torchvision.
* **Data Augmentation:** Utilizes the latest `torchvision.v2` transforms for improved model generalization.
* **Evaluation:** Includes built-in support for Scikit-Learn confusion matrices and Matplotlib for loss/accuracy visualisation.

## Required Libraries
To run this project, ensure you have the following libraries installed:
* **OpenCV (cv2):** Image processing and loading.
* **NumPy:** Numerical operations and matrix handling.
* **PyTorch & Torchvision:** Core deep learning framework and model utilities.
* **Matplotlib:** Plotting training curves and data visualization.
* **Scikit-Learn:** Metrics calculation and confusion matrix generation.

## Datasets
The model is designed to work with the following datasets:
1.  **CK (Cohn-Kanade):** A standard dataset for facial expression analysis.
2.  **CK+ (Extended Cohn-Kanade):** An enhanced version of CK with more labeled sequences.
3.  **JAFFE (Japanese Female Facial Expression):** A dataset containing images of 10 Japanese female models posing 7 primary facial expressions.

## Installation and Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jamesharrison2005/FER-CNN-AlexNet.git
   cd FER-CNN-AlexNet
   ```

2. **Install Dependencies:**
   ```bash
   pip install torch torchvision numpy opencv-python matplotlib scikit-learn
   ```

## Dataset Structure
For the code to correctly load the data using `ImageFolder`, organise your dataset directories as follows:
```text
/data
    /train
        /angry
        /happy
        /surprise
        ...
    /test
        /angry
        /happy
        /surprise
        ...
```

## Model Architecture
The project utilises the `torchvision.models.alexnet` implementation, modified for the specific requirements of facial expression recognition:
* **Input Layer:** Adjusted to handle the specific image dimensions (e.g., grayscale or RGB).
* **Feature Extractor:** Five convolutional layers with Max Pooling.
* **Classifier:** Fully connected layers with Dropout for regularisation, ending in a Softmax output corresponding to the number of emotion classes.

## Usage
### Training the Model
To begin the training process, execute the main training script. This script will handle data loading, transformations, and weight updates:
```bash
python train.py
```

### Evaluation
Once training is complete, run the evaluation script to generate performance metrics and a confusion matrix:
```bash
python evaluate.py
```

## Visualization
The repository includes scripts to visualise:
* **Training History:** Plots for training/validation loss and accuracy over epochs.
* **Confusion Matrix:** A visual representation of class-wise performance using `ConfusionMatrixDisplay` to identify which expressions the model distinguishes most effectively.


