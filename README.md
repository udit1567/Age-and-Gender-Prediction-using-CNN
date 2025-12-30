# Age and Gender Prediction using CNN

This project focuses on predicting **human age** and **gender** from facial images using **deep learning with PyTorch**.  
The task is formulated as a **multi-task learning problem**, where a single convolutional neural network jointly performs:
- **Age estimation** as a regression task
- **Gender prediction** as a binary classification task

Two modeling strategies are explored to analyze the performance impact of **training from scratch** versus **transfer learning using pretrained models**.

---

## Project Objective

The primary objectives of this project are to:
- Design and train deep learning models for **facial age and gender prediction**
- Compare a **custom CNN trained from scratch** with a **fine-tuned pretrained CNN**
- Study the effectiveness of **transfer learning** in improving model performance
- Build a clean and reproducible **training and inference pipeline**

---

## Modeling Approaches

### Approach 1: Scratch CNN

A Convolutional Neural Network was designed and trained entirely from scratch to serve as a baseline model.

#### Architecture
- Three convolutional blocks:
  - `Conv2D → BatchNorm → ReLU → MaxPooling`
- Shared feature extractor
- Two task-specific fully connected heads:
  - **Gender head:** 2-class classification
  - **Age head:** continuous regression output

#### Training Setup
- **Loss Functions**
  - Gender: `CrossEntropyLoss`
  - Age: `L1Loss`
- **Optimizer:** Adam (`lr = 1e-4`)
- **Data Augmentation**
  - Random horizontal flips
  - Random rotations
  - Image normalization using `torchvision.transforms`

#### Results
- **Public Leaderboard Score:** `0.509`

This model learned basic facial representations but was limited by its shallow architecture and reduced feature extraction capacity.

---

### Approach 2: Fine-Tuned CNN (ResNet18)

To improve performance, a **ResNet18 model pretrained on ImageNet** was fine-tuned for the age and gender prediction tasks.

#### Model Modifications
- Replaced the original fully connected layer with:
  - A gender classification head (2 classes)
  - An age regression head (continuous output)

#### Training Strategy
- **Stage 1:**  
  - Backbone frozen
  - Train only the task-specific heads  
  - Epochs: 10  
  - Learning rate: `1e-4`

- **Stage 2:**  
  - Unfreeze all layers
  - Fine-tune the entire network  
  - Epochs: 10  
  - Learning rate: `1e-5`

#### Loss Functions
- Gender: `CrossEntropyLoss`
- Age: `L1Loss`
- Joint optimization of both objectives

#### Results
- **Public Leaderboard Score:** `0.8111`

This approach showed a significant performance improvement due to **transfer learning**, benefiting from robust pretrained feature representations.

---

## Model Comparison and Results

<img width="952" height="509" alt="Screenshot 2025-12-30 at 4 32 56 PM" src="https://github.com/user-attachments/assets/5951108f-f7cc-4932-ba21-a6aa0ca3084f" />


| Model | Architecture | Loss Function | Epochs | Public Score | Remarks |
|------|-------------|---------------|--------|--------------|--------|
| Scratch CNN | Custom CNN | CE + L1 | 20 | 0.509 | Baseline model |
| Fine-Tuned CNN | ResNet18 (Pretrained) | CE + L1 | 20 | 0.8111 | Significant improvement via transfer learning |

---

## Key Features

- Multi-task learning framework:
  - Age regression
  - Gender classification
- Shared CNN backbone with task-specific heads
- Data augmentation for improved generalization
- Clean training and inference pipelines
- Modular PyTorch implementation
- Experiment tracking support (TrackIO)

---

## Key Learnings and Takeaways

- Multi-task learning enables effective feature sharing between related tasks
- Transfer learning dramatically improves performance on limited datasets
- Freezing and gradually unfreezing pretrained backbones stabilizes training
- Proper loss balancing is crucial when combining regression and classification tasks
- Data augmentation plays a vital role in improving generalization


