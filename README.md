# 3D Brain MRI Segmentation with SwinUNETR

This repository contains the implementation of a 3D brain MRI segmentation model using the SwinUNETR architecture. The project focuses on segmenting lesions related to Hypoxic Ischemic Encephalopathy (HIE) and includes comprehensive preprocessing, data augmentation techniques, and model training using a robust loss function to address class imbalance.

## Overview

Hypoxic Ischemic Encephalopathy (HIE) is a severe neonatal brain injury requiring precise and timely diagnosis. The segmentation of HIE lesions in brain MRI scans is challenging due to their subtle and diffuse nature. This project leverages deep learning techniques to improve segmentation accuracy and model generalization.

## Methodology

- **Model**: `SwinUNETR` is used for 3D segmentation, taking advantage of its feature extraction and spatial hierarchy capabilities.
- **Preprocessing**: MRI images must be converted to a `MONAI`-compatible format. Code for this conversion is located at the top of `train.py`.
- **Loss Function**: `Focal Tversky Loss` is implemented to address class imbalance.
- **Ensemble Voting (Experimental)**: The `test_ensemble.py` script implements a voting-based ensemble with three models trained on different window-level settings. This method was tested but ultimately not used for final results due to inconsistencies observed on the competition platform.

## Dataset Preparation

The dataset must be formatted to be compatible with `MONAI`. Code for data preprocessing and conversion is provided in `train.py`.

## Models

- Model files are located in the `models` directory.
- A Google Drive link for downloading these models is included in the `models` section of the code.

## Getting Started

### Prerequisites

- `Python 3.8+`
- `PyTorch 1.10+`
- `MONAI`
- `SimpleITK`
- `NumPy`
- Additional dependencies listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone

# Install required packages
pip install -r requirements.txt
