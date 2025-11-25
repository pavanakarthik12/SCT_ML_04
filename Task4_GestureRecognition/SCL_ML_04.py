#!/usr/bin/env python3
"""
SCL_ML_04.py

Hand Gesture Recognition using classical ML (HOG + SVM) on Sign Language MNIST.

Functions:
 - load_data()
 - preprocess()
 - extract_hog_features()
 - train_model()
 - evaluate_model()
 - predict_single()

Usage examples in README_SCL_ML_04.md
"""
import argparse
import os
try:
    import joblib
except Exception:
    import pickle

    class _JoblibFallback:
        @staticmethod
        def dump(obj, path):
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

        @staticmethod
        def load(path):
            with open(path, 'rb') as f:
                return pickle.load(f)

    joblib = _JoblibFallback()
import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def load_data(train_path: str, test_path: str):
    """Load sign language mnist CSVs and return train/test features and labels.

    Each CSV is expected to have a `label` column and 784 pixel columns.
    """
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    y_train = train['label'].values
    X_train = train.drop('label', axis=1).values

    y_test = test['label'].values
    X_test = test.drop('label', axis=1).values

    return X_train, y_train, X_test, y_test


def preprocess(X: np.ndarray):
    """Normalize and reshape flattened 784-length arrays into (n,28,28).

    Returns images as float32 in range [0,1].
    """
    Xf = X.astype(np.float32) / 255.0
    Xf = Xf.reshape((-1, 28, 28))
    return Xf


def extract_hog_features(images: np.ndarray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    """Extract HOG descriptors for a collection of 2D images.

    images: array-like of shape (n_samples, height, width)
    Returns: ndarray (n_samples, n_features)
    """
    feats = []
    for img in images:
        fd = hog(img,
                 orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 block_norm='L2-Hys',
                 visualize=False,
                 feature_vector=True)
        feats.append(fd)
    return np.array(feats)


def train_model(X_hog: np.ndarray, y: np.ndarray, kernel='linear', C=1.0):
    """Train an SVM pipeline (scaler + SVC) and return the fitted pipeline."""
    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel=kernel, C=C, probability=True, random_state=42))
    ])
    clf.fit(X_hog, y)
    return clf


def evaluate_model(model, X_hog_test: np.ndarray, y_test: np.ndarray):
    """Evaluate model and return metrics dict and print outputs."""
    y_pred = model.predict(X_hog_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}\n")
    print("Classification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    return {'accuracy': acc, 'report': report, 'confusion_matrix': cm}


def predict_single(model, image: np.ndarray, hog_params=None):
    """Predict single 28x28 image using model and HOG params.

    image: shape (28,28) or flattened (784,)
    hog_params: dict or None
    Returns: predicted label and probabilities (if available)
    """
    if image.ndim == 1:
        image = image.reshape(28, 28)
    if hog_params is None:
        hog_params = {'orientations': 9, 'pixels_per_cell': (8, 8), 'cells_per_block': (2, 2)}

    fd = hog(image,
             orientations=hog_params.get('orientations', 9),
             pixels_per_cell=hog_params.get('pixels_per_cell', (8, 8)),
             cells_per_block=hog_params.get('cells_per_block', (2, 2)),
             block_norm='L2-Hys',
             visualize=False,
             feature_vector=True)
    fd = fd.reshape(1, -1)
    pred = model.predict(fd)[0]
    probs = None
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(fd)[0]
    return pred, probs


def find_default_paths():
    """Return sensible default paths for train/test CSVs based on common layout."""
    candidates = [
        ('dataset/sign_mnist_train.csv', 'dataset/sign_mnist_test.csv'),
        ('sign_mnist_train.csv', 'sign_mnist_test.csv'),
        ('sign_mnist_train/sign_mnist_train.csv', 'sign_mnist_test/sign_mnist_test.csv')
    ]
    for t, s in candidates:
        if os.path.exists(t) and os.path.exists(s):
            return t, s
    # fallback to first candidate (may raise later)
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description='HOG + SVM Hand Gesture Recognition (Sign Language MNIST)')
    parser.add_argument('--train', action='store_true', help='Train model using training CSV and evaluate on test CSV')
    parser.add_argument('--train_path', type=str, default=None, help='Path to sign_mnist_train.csv')
    parser.add_argument('--test_path', type=str, default=None, help='Path to sign_mnist_test.csv')
    parser.add_argument('--model_out', type=str, default='gesture_svm.pkl', help='Where to save trained model')
    parser.add_argument('--model_in', type=str, default=None, help='Path to trained model for prediction')
    parser.add_argument('--predict_index', type=int, default=-1, help='Index in test set to run a single prediction')
    args = parser.parse_args()

    # determine paths
    if args.train_path and args.test_path:
        train_path, test_path = args.train_path, args.test_path
    else:
        train_path, test_path = find_default_paths()

    model = None

    if args.train:
        print(f"Loading data from:\n  train: {train_path}\n  test:  {test_path}")
        X_train_raw, y_train, X_test_raw, y_test = load_data(train_path, test_path)

        print("Preprocessing images...")
        X_train_imgs = preprocess(X_train_raw)
        X_test_imgs = preprocess(X_test_raw)

        print("Extracting HOG features...")
        X_train_hog = extract_hog_features(X_train_imgs)
        X_test_hog = extract_hog_features(X_test_imgs)

        print("Training SVM model...")
        model = train_model(X_train_hog, y_train)

        print("Evaluating on test set...")
        evaluate_model(model, X_test_hog, y_test)

        print(f"Saving trained model to {args.model_out}")
        joblib.dump(model, args.model_out)

    # prediction: either from just-trained model or load provided model
    if args.predict_index >= 0:
        # ensure test set loaded
        if 'X_test_raw' not in locals():
            # load test only
            X_train_raw, y_train, X_test_raw, y_test = load_data(train_path, test_path)
            X_test_imgs = preprocess(X_test_raw)

        if model is None:
            if args.model_in and os.path.exists(args.model_in):
                model = joblib.load(args.model_in)
            elif os.path.exists(args.model_out):
                model = joblib.load(args.model_out)
            else:
                raise FileNotFoundError('No trained model found. Train first or supply --model_in')

        idx = args.predict_index
        if idx < 0 or idx >= len(X_test_imgs):
            raise IndexError('predict_index out of range for test set')

        sample = X_test_imgs[idx]
        pred, probs = predict_single(model, sample)
        print(f"Predicted label for test index {idx}: {pred}")
        if probs is not None:
            top_k = np.argsort(probs)[::-1][:5]
            print("Top probabilities:")
            for k in top_k:
                print(f"  class {k}: {probs[k]:.4f}")


if __name__ == '__main__':
    main()
