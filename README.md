
# Image Manipulation Detection using ELA and Multi-Input CNN

## Introduction

This project detects image manipulation by leveraging Error Level Analysis (ELA) combined with a multi-input Convolutional Neural Network (CNN). In this version, the model processes both the original image and its corresponding ELA-transformed version through two parallel branches. Each branch uses a pre-trained network (EfficientNetV2B0 by default with a fallback to ResNet50V2) to extract features. The extracted features are then concatenated, followed by dense layers for final classification. With this enhanced architecture, our model achieved a **93% accuracy** in classifying images as real or fake.

## Running Instructions

### Prerequisites

Ensure you have the following libraries installed:
- `numpy`
- `matplotlib`
- `PIL` (Pillow)
- `tensorflow`
- `keras`
- `scikit-learn`
- `seaborn`
- `streamlit`
- `sqlite3`

Install the required libraries via pip:

```bash
pip install numpy matplotlib pillow tensorflow keras scikit-learn seaborn streamlit sqlite3
```

### Project Structure

Place the provided Python model code (e.g., `cnn-model.py`) and the Streamlit app code (e.g., `app.py`) in the same project directory.

### Dataset

Download the CASIA dataset from [this Kaggle link](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) and extract it. The dataset should include separate directories for real and fake images.

### Running the Model Code

1. **Update Paths if Necessary**:  
   Ensure that the file paths in the model code (e.g., for the CASIA dataset and the directories where ELA images are stored) are set correctly.

2. **Execute the Model Training Script**:  
   Run the following command to train the model and generate a saved model file (with a `.keras` extension):

   ```bash
   python cnn-model.py
   ```

   The script includes:
   - **Preprocessing**: Converting original images to their ELA representations.
   - **Data Augmentation**: Using image generators to augment both original and ELA images.
   - **Multi-Input Model Construction**: Two branches extract features from the original and ELA images. EfficientNetV2B0 is used by default with a fallback to ResNet50V2.
   - **Training Phases**: Initial training of top layers followed by fine-tuning by unfreezing the pre-trained layers.
   - **Evaluation**: Detailed metrics with confusion matrix, ROC, and precision-recall curves.
   - **Saving the Model**: The final model is saved for later use.

### Running the Streamlit App

1. **Start the Streamlit Interface**:  
   Launch the web app using:

   ```bash
   streamlit run app.py
   ```

   This app allows users to interactively upload and evaluate images against the trained model.

## Model and Approach Differences

- **Dual-Branch Architecture**:  
  The new model accepts two inputs—original images and their ELA-transformed counterparts. Features are extracted independently from both branches and concatenated before making the final classification.

- **Pre-Trained Backbone Networks**:  
  Each branch attempts to load EfficientNetV2B0 pre-trained on ImageNet. In case of any loading issues, the model automatically falls back to ResNet50V2. This ensures robustness across different environments.

- **Training Strategy**:  
  The training process is divided into two phases:
  1. **Training Top Layers**: Only the newly added layers are trained while the base models remain frozen.
  2. **Fine-Tuning**: A subset of the base model layers are unfrozen and fine-tuned with a lower learning rate to further boost performance.

- **Performance**:  
  The implemented approach achieved a **93% accuracy**, along with high precision and recall, demonstrating the effectiveness of combining original and ELA features for detecting image manipulation.

## Final Notes

This updated version significantly improves upon the previous implementation by integrating advanced transfer learning techniques and a dual-input strategy. The detailed evaluation metrics and visualizations (confusion matrix, ROC, and precision-recall curves) help in understanding the model’s performance comprehensively.

Happy coding!

---
