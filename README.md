# Face Attribute Detection Project

## Overview

This repository contains the implementation of a face attribute detection project focusing on three main tasks: face mask detection, gender prediction, and glasses or no glasses detection. The project leverages deep learning techniques, including Convolutional Neural Networks (CNNs) and transfer learning, with models like Xception, DenseNet201, VGG19, ResNet50, and EfficientNetB7.

## Table of Contents

- [Introduction](#introduction)
- [Datasets](#datasets)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Techniques](#modeling-techniques)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Usage](#usage)
- [License](#license)

## Introduction

In today's digital age, face attribute detection plays a crucial role in various applications, from security to healthcare. This project focuses on three specific tasks:

1. **Face Mask Detection**: Utilizing the Xception model, this task aims to identify whether an individual is wearing a face mask or not.

2. **Gender Prediction**: Leveraging transfer learning with various pre-trained models, including DenseNet201, VGG19, ResNet50, and EfficientNetB7, the project predicts the gender of individuals based on facial features.

3. **Glasses or No Glasses Detection**: The EfficientNetB7 model is employed to detect whether an individual is wearing glasses or not.

## Datasets

- [men-women-classification](https://www.kaggle.com/datasets/playlist/men-women-classification/data): A dataset consists of 3,354 images (in JPG format), categorized into men (1,414 files) and women (1,940 files)
- [Face Mask 12K Images Dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset): A dataset comprising nearly 12,000 images for face mask detection.
- [Glasses or No Glasses Dataset](https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses): A dataset generated using a Generative Adversarial Neural Network (GAN) with 512-dimensional latent vectors and 5000 images.

## Data Preprocessing

The data preprocessing pipeline includes:

- Data augmentation for improved model generalization.
- Normalization of pixel values for consistency.
- Dataset restructuring for efficient model training.

## Modeling Techniques

Various pre-trained models are employed for gender prediction:

- Xception
- DenseNet201
- VGG19
- ResNet50
- EfficientNetB7

## Model Evaluation

The models are evaluated based on key metrics such as accuracy, precision, recall, and F1 score. The results are summarized in a comprehensive table for comparison.

## Results

The project achieves remarkable accuracy in gender prediction, with EfficientNetB7 leading with an accuracy of 89.8%.

## Usage

Follow these steps to reproduce and explore the project:

1. Clone the repository:

```bash
git clone https://github.com/your-username/face-attribute-detection.git
```

2. Run the notebooks and save the accurate trained model (.h5 - .hdf5)

3. Run the experimentation:

```bash
python Facial_Attributes_Recognition_Experimentation.py
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
