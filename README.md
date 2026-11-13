# Lab 07: Face Gender Classification with Deep Convolutional Neural Networks (CNN)

**Author:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Institution:** The University of Lahore (UOL)  
**Course:** Artificial Intelligence Lab (AI-101L)  
**Dataset:** UTKFace Benchmark (`crop_part1`)  

---

## 📌 Project Overview
This repository contains the complete deep learning pipeline for **Lab 07: Gender Classification using Convolutional Neural Networks (CNN)**. The project implements end-to-end binary facial classification on 9,779 facial images from the UTKFace benchmark, utilizing an optimized TensorFlow/Keras pipeline with data streaming, multi-layer feature extraction, regularization, and visual inference.

---

## 📂 Repository Structure
```
AI-Lab07-Gender-Classification-CNN/
├── Gender_Classification_CNN.ipynb   # Complete Jupyter Notebook with execution outputs
├── report.md                        # Academic laboratory report with full technical analysis
├── requirements.txt                 # Python dependencies
└── README.md                        # Setup and execution guide
```

---

## 🧠 Model Architecture
A sequential Convolutional Neural Network designed for spatial feature hierarchy:
- **Input Dimension:** $64 \times 64 \times 3$ (RGB)
- **Conv Block 1:** `Conv2D(32, (3, 3))` + `MaxPooling2D((2, 2))`
- **Conv Block 2:** `Conv2D(64, (3, 3))` + `MaxPooling2D((2, 2))`
- **Conv Block 3:** `Conv2D(128, (3, 3))` + `MaxPooling2D((2, 2))`
- **Dense Head:** `Flatten` -> `Dense(128, activation='relu')` -> `Dropout(0.5)` -> `Dense(1, activation='sigmoid')`
- **Total Parameters:** 683,329 parameters (~2.61 MB)

---

## 📊 Dataset: UTKFace (`crop_part1`)
- **Total Images:** 9,779
- **Classes:** 
  - `0`: Male (4,372 images)
  - `1`: Female (5,407 images)
- **Filename convention:** `[age]_[gender]_[race]_[date].chip.jpg`
- **Dataset Partitioning:** Stratified split into 70% Train (6,845), 10% Validation (978), and 20% Test (1,956).

> **Note on Dataset:** The UTKFace dataset (`crop_part1`) can be downloaded directly from [Kaggle UTKFace Dataset](https://www.kaggle.com/datasets/jangedoo/utkface-new). Extract the `crop_part1` folder directly into the root of this repository. The notebook already includes all pre-computed outputs and charts for review.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
```bash
jupyter notebook Gender_Classification_CNN.ipynb
```

The notebook runs sequentially:
1. Installs/verifies required libraries (`tensorflow`, `scikit-learn`, `matplotlib`, `seaborn`).
2. Scans and labels images from the dataset folder.
3. Builds an asynchronous, autotuned `tf.data.Dataset` pipeline.
4. Compiles and fits the CNN model with `EarlyStopping`.
5. Visualizes accuracy and loss progression curves.
6. Computes test set predictions, confusion matrix, and visualizes individual sample inference.
7. Saves the trained model weights as `utkface_gender_cnn.keras`.
