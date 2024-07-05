# Image Manipulation Detection using Error Level Analysis (ELA)

## Introduction

This project aims to detect image manipulation using Error Level Analysis (ELA) and Convolutional Neural Networks (CNN). The model is trained on a dataset of real and fake images, converting them into ELA images to highlight discrepancies introduced by manipulation. The primary goal is to classify images as either real or fake based on their ELA features.

## Running Instructions

### Prerequisites

Ensure you have the following libraries installed:
- `numpy`
- `matplotlib`
- `PIL`
- `keras`
- `scikit-learn`
- `tensorflow`
- `streamlit`
- `sqlite3`

You can install these libraries using pip:

```bash
pip install numpy matplotlib pillow keras scikit-learn tensorflow streamlit sqlite3
```

## Steps to Run the Project

### Download the Model Code and Streamlit App Code
Save the provided model code and Streamlit app code in a directory.

### Download the Dataset
The dataset used here is the CASIA dataset from Kaggle. Download it from [this link](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) and extract it.

### Run the Model Code

1. Ensure all paths are correct in the code.
2. Execute the Python script to train the model and generate a `.h5` file:

    ```bash
    python image_manipulation_detection.py
    ```

### Run the Streamlit App

1. Execute the following command to start the Streamlit app:

    ```bash
    streamlit run app.py
    ```

