# CNN-in-Action-Real-World-Image-Classification-Scenarios
### Aim
To develop an image classification model using a custom Convolutional Neural Network (CNN) in PyTorch that classifies images into three categories: cats, dogs, and panda. The project demonstrates basic deep learning techniques for visual recognition and explores model training, evaluation, and results visualization.

### Procedure
Dataset Preparation

Each subfolder contains respective class images for training/testing.

### Environment Setup

Ensure Python 3.6+ is installed.

Install required libraries:

bash
```
pip install torch torchvision matplotlib numpy seaborn scikit-learn
```
Use Google Colab or local setup with GPU if possible.

### Model Building

Open and run cnn.ipynb notebook.

Update the dataset path as needed.

### Training

The notebook loads images, applies data augmentation, normalizes inputs.

Trains a custom CNN on the training dataset.

Tracks and prints epoch-wise loss and accuracy.

### Evaluation

Evaluates model on the test dataset.

Displays confusion matrix and class-wise performance.

Plots loss and accuracy curves.

### Algorithm
Data Augmentation & Loading

Apply RandomHorizontalFlip, Resize, ToTensor, and Normalize using torchvision.transforms.

Custom CNN Model (PyTorch)

Layer 1: Convolution + ReLU + MaxPooling

Layer 2: Convolution + ReLU + MaxPooling

Flatten: Input for fully connected layers

FC1: Fully connected + ReLU

FC2: Fully connected to get logits for 3 classes

Training Loop

For each epoch:

Forward pass on train data

Calculate loss (CrossEntropyLoss)

Backward pass and optimizer step (Adam/SGD)

Track accuracy

Validation/Test Loop

After each epoch or at the end, run model on test data

Calculate test loss, accuracy

Generate confusion matrix

### Visualization

Plot loss and accuracy curves using matplotlib

Display confusion matrix with seaborn heatmap

### Output
Training and Validation Accuracy/Loss: Printed per epoch.

Plots:

Training/Validation loss vs. epochs

Training/Validation accuracy vs. epochs

Confusion Matrix: Visual class performance on test data.

### Sample Output:

Final test accuracy (e.g., ≈86% on sample dataset, may vary)

Visualization of class-wise predictions
