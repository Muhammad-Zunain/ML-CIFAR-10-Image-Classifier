# CIFAR-10 Binary Image Classification – Animal vs Vehicle

This project focuses on converting the CIFAR-10 dataset into binary classes (Vehicle vs Animal) and comparing the performance of different machine learning models including **K-Nearest Neighbors (KNN)**, **Logistic Regression**, and **Convolutional Neural Networks (CNN)**.

---


## 🎯 Objective


The primary objective of this project is to classify images from the **CIFAR-10** dataset into two broad **binary categories**:

- **Vehicle** → Classes: `airplane`, `automobile`, `ship`, `truck`
- **Animal** → Classes: `bird`, `cat`, `deer`, `dog`, `frog`, `horse`

---

This project walks through a complete machine learning pipeline using both traditional ML and deep learning:

###  Machine Learning Models:
- **K-Nearest Neighbors (KNN)**
- **Logistic Regression**

###  Deep Learning Models:
- **Convolutional Neural Networks (CNN)**  
  - Simple CNN  
  - Deeper CNN  
  - Advanced CNN (with BatchNorm & Dropout)

---

## 📁 Dataset Overview

- **Dataset**: CIFAR-10 (60,000 32x32 RGB images, 10 classes)
- **Binary Conversion**:
  - **Animals**: bird, cat, deer, dog, frog, horse
  - **Vehicles**: airplane, automobile, ship, truck
- **Split**:
  - Training: 45,000 images
  - Validation: 5,000 images
  - Test: 10,000 images


## 🔄 Preprocessing Steps

- Binary label mapping
- Class balancing via undersampling
- Flattening + PCA (for KNN & Logistic Regression)
- Normalization (0–1)
- 3D shape retained for CNNs

---

## 🧠 Implemented Models

### K-Nearest Neighbors (KNN)
- Distance: Euclidean
- Values of k: 1, 3, 5, 7, 9
- PCA reduced input features from 3072 to 100

### Logistic Regression
- Solvers: `liblinear`, `lbfgs`
- Penalty: `l1`, `l2`
- Regularization Strength (C): 0.1, 1.0, 10.0

### CNN Architectures
- **Simple CNN**: 2 conv layers, 1 dense
- **Deeper CNN**: More conv layers, bigger dense layer
- **Advanced CNN**: + BatchNorm, Dropout, deeper filters

---

## 📊 Results Summary

| Model               | Accuracy | ROC AUC | PR AUC |
|---------------------|----------|---------|--------|
| KNN                 | 82%      | 0.893   | 0.924  |
| Logistic Regression | 81%      | 0.884   | 0.903  |
| CNN (Advanced)      | **95%**  | **0.960** | **0.950** |

---

## 📈 Evaluation Techniques

- Confusion Matrix
- ROC & Precision-Recall Curves
- Learning Curves
- Model Saving (Persistence)

---

## ▶️ Running the Project

```bash
# Clone the repository
git clone https://github.com/Muhammad-Zunain/ML-CIFAR-10-Image-Classifier.git
cd ML-CIFAR-10-Image-Classifier
```
---

## 🧾 How to Download and Set Up the CIFAR-10 Dataset

To run this project, you must download and organize the CIFAR-10 dataset locally. Here's how:

###  Step 1: Download the Dataset

1. Go to the official CIFAR-10 page:  
    [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

2. Scroll down and **download** the following file:  
    `cifar-10 python version`

---

###  Step 2: Extract the Dataset

Once downloaded, extract the compressed `.tar.gz` file:

```bash
tar -xzvf cifar-10-python.tar.gz
```

 ### Step 3: Organize the Data Directory
 ```
 ML-CIFAR-10-Image-Classifier/
    ├── Machine-Learing.ipynb
    ├── ML-Report.pdf
    ├── saved_models/
    ├── data_batch_1
    ├── data_batch_2
    ├── data_batch_3
    ├── data_batch_4
    ├── data_batch_5
    ├── test_batch
    └── batches.meta
```

---

## 📷 Sample Visualizations

### Class Distribution

| Before Balancing                                                | After Balancing                                               |
| --------------------------------------------------------------- | ------------------------------------------------------------- |
|![class_dist_before](https://github.com/user-attachments/assets/9d55f853-efa4-4757-8230-1fa675984896)| ![class_dist_after](https://github.com/user-attachments/assets/23060627-f62e-43b5-a22e-2d866ac8da7f)|


### PCA Visualization

| PCA Before                                 | PCA After                                |
| ------------------------------------------ | ---------------------------------------- |
| ![pca_before_balancing](https://github.com/user-attachments/assets/7b9c9e23-d995-4064-a37e-449fe13eafe6)|![pca_after_balancing](https://github.com/user-attachments/assets/80d35a15-b648-410e-b3f4-42dc3d42e7ad)|

### Pixel Distribution

![pixel_distribution](https://github.com/user-attachments/assets/fd2569e2-947c-4727-b0d3-aefcfed3b138)

### ROC Curves

| KNN                                  | Logistic Regression                       | CNN                                  |
| ------------------------------------ | ----------------------------------------- | ------------------------------------ |
|![roc_knn](https://github.com/user-attachments/assets/93673acd-6d44-4f8f-9223-22de7c22cf99)|![roc_logistic](https://github.com/user-attachments/assets/ad16ceff-087d-4930-9084-c3dc79ed97a9)| ![roc_cnn](https://github.com/user-attachments/assets/e368f7ff-6e47-4a11-9cca-7d5e0186df57)|

### CNN Training Curves
![cnn_training_curves](https://github.com/user-attachments/assets/462ac8f8-00a1-4a61-86ab-67d6af09f4a5)

### Confusion Matrices

| KNN                                                | Logistic Regression                                     | CNN                                                |
| -------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| ![knn_conf_matrix](https://github.com/user-attachments/assets/5fdcd1f3-5916-431d-81f1-d574bc2448f1)| ![log_conf_matrix](https://github.com/user-attachments/assets/2fd8ecd3-9f29-4b8c-af3f-a18ab51ffb22)|![cnn_conf_matrix](https://github.com/user-attachments/assets/6ccb3f38-f6f5-4c00-80b5-316169da23f8)|

---

## 📚 Report
A detailed analysis of all steps, results, and comparisons is available in ML-Report.pdf.

---

## 🤝 Contributors
- [**Muhammad Zunain**](https://github.com/Muhammad-Zunain)  
- [**Muhammad Owais**][github.com/MuhammadOwais03](https://github.com/MuhammadOwais03) 
- **Zuhaib Noor**

---

Enjoy using this project! Contributions and feedback are welcome. 😊

