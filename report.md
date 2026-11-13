# Laboratory Report: Lab 07 — Face Gender Classification Using Convolutional Neural Networks (CNN)

**Course:** Artificial Intelligence Lab (AI-101L)  
**Student Name:** Mohsin Raza  
**SAP ID:** 70177136  
**Email:** 70177136@student.uol.edu.pk  
**Department:** Department of Computer Science & Information Technology  
**Institution:** The University of Lahore (UOL)  

---

## 1. Introduction and Theoretical Foundation

Face attribute analysis, specifically binary gender classification from unconstrained facial imagery, is a foundational task in biometric analysis, human-computer interaction (HCI), and computational demographic estimation. Unlike classical machine learning pipelines that rely on handcrafted feature descriptors (such as Histogram of Oriented Gradients [HOG] or Local Binary Patterns [LBP]) paired with Support Vector Machines (SVM), deep Convolutional Neural Networks (CNNs) learn hierarchical, translation-invariant representations directly from raw pixel arrays.

### 1.1 Objectives
The primary objectives of this laboratory assignment are:
1. Parse and extract ground-truth demographic labels from the UTKFace cropped faces dataset (`crop_part1`) using filename tokenization.
2. Build an efficient, asynchronous data feeding pipeline using TensorFlow `tf.data.Dataset` with parallel mapping, caching, shuffling, and autotuned prefetching.
3. Design and implement a sequential multi-stage Deep Convolutional Neural Network tailored for binary facial classification.
4. Train and optimize the model using the Adam optimizer and Binary Cross-Entropy loss with Early Stopping to prevent overfitting.
5. Evaluate model performance using confusion matrices, classification metrics (accuracy, loss curves), and test-set single-sample visual inference.

---

## 2. Dataset Analysis and Preprocessing Pipeline

### 2.1 Dataset Composition
The dataset utilized is the **UTKFace** benchmark (`crop_part1`), consisting of **9,779** cropped facial images under varying poses, illumination conditions, expressions, and ethnic backgrounds.
- **Filename Format:** `[age]_[gender]_[race]_[date].chip.jpg`
  - `age`: Integer representing estimated chronological age.
  - `gender`: Binary integer label (`0` for Male, `1` for Female).
  - `race`: Categorical integer identifier.
- **Distribution:**
  - Total Valid Images: **9,779**
  - Male Samples (`0`): **4,372** (44.7%)
  - Female Samples (`1`): **5,407** (55.3%)

```
Total images: 9779
Male (0): 4372
Female (1): 5407
```

### 2.2 Stratified Partitioning
To maintain balanced demographic proportions across splits, `sklearn.model_selection.train_test_split` was applied with stratified sampling:
- **Training Set (70%):** 6,845 images
- **Validation Set (10%):** 978 images
- **Testing Set (20%):** 1,956 images

### 2.3 `tf.data` Ingestion and Preprocessing Pipeline
To maximize GPU/CPU throughput and eliminate memory bottlenecks:
1. **Dynamic Disk Reading:** `tf.io.read_file` loads byte streams on demand.
2. **Decoding & Shape Assertion:** `tf.io.decode_image(channels=3, expand_animations=False)` decodes JPEG/PNG buffers into 3D tensors `[H, W, 3]`.
3. **Resizing:** Bilinear resizing to fixed dimensions of $64 \times 64 \times 3$ (`IMG_SIZE = 64`).
4. **Min-Max Scaling:** Pixel intensities scaled to $[0.0, 1.0]$ by division by $255.0$:
   $$X_{\text{norm}} = \frac{X}{255.0}$$
5. **Autotuned Optimization:** Datasets were batched (`BATCH_SIZE = 32`), shuffled with a buffer size of 2,000 for training, and chained with `prefetch(tf.data.AUTOTUNE)`.

---

## 3. Deep Convolutional Neural Network Architecture

The architecture was formulated as a feedforward sequential model with three convolutional feature extraction blocks followed by a dense classification head:

```
Input Tensor: (64, 64, 3)
      │
┌─────▼──────────────────────────┐
│ Conv2D (32 filters, 3x3, valid)│ -> (None, 62, 62, 32) [896 params]
│ MaxPooling2D (pool_size=(2, 2))│ -> (None, 31, 31, 32)
└─────┬──────────────────────────┘
      │
┌─────▼──────────────────────────┐
│ Conv2D (64 filters, 3x3, valid)│ -> (None, 29, 29, 64) [18,496 params]
│ MaxPooling2D (pool_size=(2, 2))│ -> (None, 14, 14, 64)
└─────┬──────────────────────────┘
      │
┌─────▼──────────────────────────┐
│ Conv2D (128 filters, 3x3, valid│ -> (None, 12, 12, 128) [73,856 params]
│ MaxPooling2D (pool_size=(2, 2))│ -> (None, 6, 6, 128)
└─────┬──────────────────────────┘
      │
┌─────▼──────────────────────────┐
│ Flatten                        │ -> (None, 4608)
│ Dense (128 units, ReLU)        │ -> (None, 128) [589,952 params]
│ Dropout (rate = 0.50)          │ -> (None, 128) [Regularization]
│ Dense (1 unit, Sigmoid)        │ -> (None, 1) [129 params]
└────────────────────────────────┘
```

- **Total Trainable Parameters:** 683,329 parameters (~2.61 MB).
- **Mathematical Form of Activation Functions:**
  - ReLU (Hidden Layers):
    $$f(z) = \max(0, z)$$
  - Sigmoid (Output Unit):
    $$\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]$$

---

## 4. Training Strategy and Optimization

The objective function minimized during optimization is the Binary Cross-Entropy loss:
$$\mathcal{L}(y, \hat{y}) = -\frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]$$

- **Optimizer:** Adam (Adaptive Moment Estimation) with standard initial learning rate $\alpha = 0.001$.
- **Early Stopping Callback:** Monitored `val_loss` with `patience=3` and `restore_best_weights=True` to halt training when generalisation stalled.

### 4.1 Convergence Trajectory Across Epochs

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Epoch Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.6086 | 65.73% | 0.5198 | 75.87% | 34s |
| **2** | 0.4835 | 76.80% | 0.4006 | 80.06% | 26s |
| **3** | 0.4149 | 80.22% | 0.3710 | 80.88% | 24s |
| **4** | 0.3822 | 81.93% | 0.3598 | **82.72%** | 24s |
| **5** | **0.3514** | **83.59%** | **0.3528** | 82.21% | 39s |

Both training and validation losses smoothly decayed across epochs, demonstrating consistent gradient propagation without numerical divergence or extreme variance.

---

## 5. Experimental Results and Discussion

### 5.1 Learning Curves Analysis
- **Accuracy Progression:** Validation accuracy accelerated rapidly from 75.87% to a peak of 82.72% in Epoch 4.
- **Loss Progression:** Validation loss decreased steadily from 0.5198 down to 0.3528, closely tracking the training loss curve and confirming that the 50% Dropout layer effectively suppressed overfitting.

### 5.2 Confusion Matrix Evaluation
Predictions were computed across the held-out test cohort of 1,956 images ($y_{\text{pred}} = \mathbb{I}[P(\text{Female}) \ge 0.5]$). The resulting confusion matrix indicated strong discriminative separation between both classes, with balanced precision and recall across both male and female faces.

### 5.3 Test Sample Visual Inference
A custom inference routine `predict_gender(image_path)` was deployed to evaluate unseen inputs:
- Given test input `X_test[0]`, the network predicted:
  - **Predicted Gender Label:** `0`
  - **Predicted Class:** `Male`
  - **Male Probability:** **75.51%**
  - **Female Probability:** **24.49%**
- Visual plotting verified the prediction was accurate against the ground truth facial morphology.

---

## 6. Model Serialization and Artifacts

The final trained weights and computational graph were persisted to disk:
```python
model.save("utkface_gender_cnn.keras")
```
This self-contained Keras v3 format allows immediate reloading for inference or integration into web APIs (e.g., FastAPI) without needing architecture re-specification.

---

## 7. Conclusions and Future Work

1. **Conclusion:** A lightweight 3-stage CNN trained on $64 \times 64$ facial crops achieved **82.7%+ validation accuracy** on the UTKFace benchmark within 5 training epochs.
2. **Future Enhancements:**
   - Incorporating real-time data augmentations (horizontal flips, slight rotations, brightness jitter) to further improve robustness to unconstrained real-world lighting.
   - Exploring transfer learning with pre-trained backbones (e.g., MobileNetV3-Small or ResNet50) for fine-grained feature reuse.
   - Multi-task demographic learning jointly predicting both gender and continuous chronological age from the same convolutional backbone.
