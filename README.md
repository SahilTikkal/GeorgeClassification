GeorgeClassification
==============================

Classifying if the image contains St. George or not.

Google colab link: https://colab.research.google.com/drive/18pfl9zIVTqtgC3bOnPpg8_-IA3XDHM_i#scrollTo=AaynBc9dPj3h

### St. George's Image Classification Project Documentation
### Overview
The St. George's Image Classification Project is a deep learning project aimed at classifying images as either "George" or "Not George." The project uses a deep learning model to analyze and predict whether an input image contains an image of St. George.

### Table of Contents
- Introduction
- Installation
- Usage
- Dataset
- Model
- Results
- Contributing
- License

### Introduction
This project employs Deep learning techniques to perform binary image classification. Specifically, it addresses the task of identifying images of St. George from a collection of images. The underlying deep learning model has been trained on a dataset of labeled images to make predictions.

### Installation
To use this project, follow these installation steps:

Clone the repository:

bash
```
git clone https://github.com/SahilTikkal/GeorgeClassification.git
cd st-georges-image-classification
```
Install the required dependencies:
```
pip install -r requirements.txt
```
### Usage
After installation, you can use the project as follows:

Prepare your image data in the appropriate format. You can organize your images into separate folders: george and not_george.

Train the model by running the training script:
```
python train.py
```
After training, you can make predictions on new images using the prediction script:

```
python predict.py path/to/your/image.jpg
```

### Dataset
The dataset used for this project consists of images categorized as either "George" or "Not George." The images should be organized into two folders: george and not_george, each containing images corresponding to their respective categories.

### Model
The image classification model is built using a deep learning architecture. It employs convolutional neural networks (CNNs) to extract features from input images and make predictions. The model has been trained on the provided dataset to achieve accurate classification.

### Results
The model's performance on the test dataset is as follows:

Accuracy: XX%
Precision: XX%
Recall: XX%
F1-Score: XX%

### Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, feel free to submit a pull request or open an issue on the project repository.
