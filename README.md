# Plant Disease Detection Using Deep Learning (PyTorch)

## Project Overview

This project is a Deep Learning based Plant Disease Detection System built using PyTorch and the PlantVillage dataset.

The model classifies plant leaf images into healthy or diseased categories using a Convolutional Neural Network (CNN).

---

# Features

- Plant disease classification
- CNN model using PyTorch
- Automatic dataset loading
- Training and validation
- Disease prediction from images
- Model saving and loading

---

# Dataset

Dataset used:
PlantVillage Dataset

# Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- PIL (Pillow)

---

# Installation

## 1. Clone Repository

git clone <repository-url>

cd plant-disease-pytorch

---

## 2. Install Dependencies

pip install -r requirements.txt

---

# Train the Model

Run:

python train.py

The trained model will be saved in:

models/plant_disease_model.pth

---



# CNN Architecture

The CNN model contains:

- 3 Convolutional Layers
- ReLU Activation
- MaxPooling Layers
- Fully Connected Layers
- Dropout Layer

---

# How It Works

Leaf Image
    ↓
Image Preprocessing
    ↓
CNN Feature Extraction
    ↓
Disease Classification
    ↓
Prediction Output

---

# Future Improvements

- Transfer Learning (ResNet18 / MobileNet)
- Streamlit GUI
- Real-time Camera Detection
- Disease Remedy Suggestions
- Better Data Augmentation
- Model Deployment

---

# Accuracy

Expected accuracy on PlantVillage dataset:
90% to 98%

Depends on:
- Epochs
- Learning rate
- Data augmentation
- Hardware

---

# Author

Arpit Kumar

---

# License

This project is for educational purposes.
