# Fake Image Detection using CNN
This project focuses on detecting fake images using a Convolutional Neural Network (CNN) architecture. The model is designed to identify manipulated images with high accuracy, leveraging deep learning techniques in computer vision.

Table of Contents
* Overview
* Dataset
* Model Architecture
* Data Preparation
* Training
* Dependencies
* Results
* Acknowledgments

### Overview
Fake image detection is crucial in identifying digitally altered images, which can lead to misinformation. This project implements a CNN-based approach to detect fake images by analyzing key features using multiple convolutional layers and pooling operations.

### Dataset
The dataset consists of real and fake images in .jpg format. The data is split into training and validation sets in an 80-20 ratio. The images are resized and preprocessed before being fed into the model.

### Model Architecture
The CNN architecture comprises two primary convolutional layers, each with 32 filters of size 5x5, followed by a max-pooling layer and fully connected (FC) layers. The architecture is designed as follows:

1. Layer 1:
* Conv Layer 1: 32 filters, 5x5 kernel
* Conv Layer 2: 32 filters, 5x5 kernel
* Max Pooling: 2x2 kernel
* Dropout: 25% to prevent overfitting
2. Layer 2:
* Flatten: The output from the convolutional layers is flattened.
* Fully Connected (FC) Layer: 256 units
* Dropout: 50% to prevent overfitting
* Softmax Activation: For output classification.
* Optimizer: RMSprop with a learning rate of 0.0005

<!-- Update with actual path if uploading the image to GitHub -->

### Data Preparation
The data preparation steps are as follows:

1. Data Input: The input consists of .jpg images.
2. Error Level Analysis: Applied to detect possible manipulations.
3. Resize: All images are resized to 128x128 pixels.
4. Normalization: Pixel values are normalized.
5. Label Encoding: Labels are encoded for classification.
6. Train-Validation Split: The data is split into training (80%) and validation (20%) sets.

### Training
The model is trained with the following parameters:
* Optimizer: RMSprop with a learning rate of 0.0005
* Loss Function: Categorical Crossentropy
* Metrics: Accuracy
* Epochs: 20
* Batch Size: 50
* Early Stopping: Monitors validation accuracy to prevent overfitting, with a patience of 2 epochs

### Dependencies
* Python 3.x
* TensorFlow or Keras
* NumPy
* Matplotlib
* scikit-learn

### Results
The model achieves high accuracy on the validation set, and the use of dropout helps to reduce overfitting.

### Acknowledgments
* This project is inspired by recent advancements in image forensics and fake image detection.
* Special thanks to open-source contributors and the machine learning community for providing valuable resources.
