# MNIST Handwritten Digit Classification

This project demonstrates handwritten digit classification using the MNIST dataset and a simple neural network built with TensorFlow and Keras. It is also designed as a learning tool to understand neural networks and their optimization.

## Overview

The notebook [notebook.ipynb](notebook.ipynb) contains the code for:

1.  **Loading and Preprocessing the Data:** The MNIST dataset is loaded using `keras.datasets.mnist.load_data()`. The pixel values are scaled to the range of 0 to 1 by dividing by 255. The training and testing datasets are flattened from a 28x28 image into a 784-element array.
2.  **Building a Neural Network:** A sequential model is created using Keras. It consists of:
    *   An input layer that flattens the 28x28 images.
    *   A dense (fully connected) layer with 100 neurons and ReLU activation.
    *   An output layer with 10 neurons (one for each digit) and sigmoid activation.
3.  **Training the Model:** The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the metric. The model is trained using the `fit()` method for 5 epochs.
4.  **Evaluating the Model:** The model's accuracy is evaluated on the test dataset using the `evaluate()` method.
5.  **Making Predictions:** The model is used to predict the digit for sample images from the test dataset.
6.  **Confusion Matrix:** The confusion matrix is plotted using the `seaborn` library to visualize the performance of the classification model.

## Key Concepts

*   **Neural Networks:** A fundamental machine learning model used for various tasks, including image classification.
*   **MNIST Dataset:** A widely used dataset of handwritten digits for training and testing image classification models.
*   **TensorFlow and Keras:** Open-source libraries for building and training machine learning models.
*   **Data Preprocessing:** Scaling and flattening the image data to improve model performance.
*   **Activation Functions:** ReLU and Sigmoid activation functions are used in the neural network layers.
*   **Optimizer:** Adam optimizer is used to update the model's weights during training.
*   **Loss Function:** Sparse categorical cross-entropy is used to measure the difference between predicted and actual outputs.
*   **Epochs:** The number of times the model iterates over the entire training dataset.
*   **Evaluation Metrics:** Accuracy is used to measure the model's performance.
*   **Confusion Matrix:** A table that summarizes the performance of a classification model by showing the counts of true positive, true negative, false positive, and false negative predictions.
*   **Optimization:** Understanding how different optimizers, loss functions, and network architectures affect model performance.

## How to Run

1.  Make sure you have Python 3 installed.
2.  Install the necessary libraries:

    ```bash
    pip install tensorflow keras matplotlib numpy seaborn
    ```
3.  Open and run the [notebook.ipynb](notebook.ipynb) file in a Jupyter Notebook environment.

## Potential Improvements

The notebook suggests several ways to improve the model and further your understanding:

*   Add more hidden layers to explore deeper network architectures.
*   Try different activation functions to see their impact on learning.
*   Experiment with different optimizers to understand their convergence properties.
*   Use different loss functions and analyze their effect on the model's learning process.
*   Adjust the number of epochs to find the optimal training duration.