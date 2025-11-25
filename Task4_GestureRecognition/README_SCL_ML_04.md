# Task 4 — Hand Gesture Recognition (Classical ML)

**Problem statement**

Build a hand gesture recognition system using only classical machine learning methods on the Sign Language MNIST dataset. The goal is to classify 28×28 grayscale hand images into letter labels.

**Dataset**

- Source: Sign Language MNIST (Kaggle). CSV files contain a `label` column and 784 pixel columns for 28×28 images.
- Files used here: `sign_mnist_train.csv`, `sign_mnist_test.csv` (place them in `dataset/` or the project root).

**Why this dataset works for classical ML**

The images are small (28×28) and grayscale. Hand-crafted features such as HOG (Histogram of Oriented Gradients) capture edge and shape information that is highly informative for classical classifiers like SVM. This keeps the solution lightweight and interpretable without requiring deep learning.

**Approach summary (HOG + SVM)**

- Preprocess images by normalizing pixel values and reshaping to 28×28.
- Extract HOG descriptors for each image.
- Train an SVM (support vector machine) with a scaler in a pipeline.
- Evaluate with accuracy, classification report, and confusion matrix.

**Preprocessing steps**

- Load CSV files (`label` + 784 pixels).
- Normalize pixel values to [0,1] by dividing by 255.
- Reshape flattened vectors to `(28, 28)` images.
- Compute HOG features using `skimage.feature.hog`.

**Files in this project**

Task4_GestureRecognition/
 ├── SCL_ML_04.py
 ├── README_SCL_ML_04.md
 ├── requirements.txt
 ├── gesture_svm.pkl (after training)
 └── dataset/
      ├── sign_mnist_train.csv
      └── sign_mnist_test.csv

**How to run**

1. Create a Python environment and install dependencies:

```bash
python -m venv .venv; .\.venv\Scripts\Activate; pip install -r requirements.txt
```

2. Train the model and evaluate on the test set (saves `gesture_svm.pkl`):

```bash
python SCL_ML_04.py --train --model_out gesture_svm.pkl
```

3. Predict a single image from the test set by index (after training or with `--model_in`):

```bash
# predict test index 10 using the model just trained
python SCL_ML_04.py --predict_index 10

# or using a pre-saved model
python SCL_ML_04.py --predict_index 10 --model_in gesture_svm.pkl
```

**Example results (expected)**

- Accuracy: (depends on train settings and sklearn SVM defaults) often in a good range for HOG+SVM on this dataset, but exact number will be produced by `SCL_ML_04.py` during evaluation.
- Classification report and confusion matrix printed to stdout.

**Future improvements**

- Tune HOG parameters (`pixels_per_cell`, `cells_per_block`, `orientations`).
- Use cross-validation and grid search for SVM hyperparameters.
- Try other classical classifiers like RandomForest or KNN for comparison.
- Add more classical features (LBP, Zernike, PCA on raw pixels) and ensemble models.

**Notes**

- This project purposely avoids any deep learning / CNN method to meet the task requirement.
- The mapping of numeric labels to letters is dataset-specific; this script works directly with numeric labels provided in the CSV files.
