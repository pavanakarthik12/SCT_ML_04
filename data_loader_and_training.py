"""
ASL Fingerspelling Data Loader and Training Script
Loads ASL MediaPipe Converted Dataset (21 hand landmarks x,y,z), trains SVM classifier.
Saves trained model and scaler for real-time recognition.

Usage: python data_loader_and_training.py --dataset_path path/to/dataset.csv
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(dataset_path):
    """Load ASL MediaPipe dataset and extract 63 landmark features (21 landmarks * x,y,z)"""
    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Extract feature columns (should be 63 landmark coordinates)
    feature_cols = [col for col in df.columns if col != 'label' and col != 'class']
    if len(feature_cols) != 63:
        print(f"Warning: Expected 63 features (21 landmarks * 3), found {len(feature_cols)}")
    
    X = df[feature_cols].values
    y = df['label'].values if 'label' in df.columns else df['class'].values
    
    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {sorted(np.unique(y))}")
    print(f"Class distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} samples")
    
    return X, y


def balance_classes(X, y):
    """Handle class imbalance by computing class weights"""
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    print(f"Class weights computed: {weight_dict}")
    return weight_dict


def train_svm_model(X_train, y_train, class_weights):
    """Train SVM with RBF kernel"""
    print("Training SVM with RBF kernel...")
    model = SVC(kernel='rbf', C=10, gamma='scale', class_weight=class_weights, 
                probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, class_weights):
    """Train Random Forest classifier"""
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=200, max_depth=20, 
                                   class_weight=class_weights, random_state=42, n_jobs=1)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    labels = sorted(np.unique(y_test))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'{model_name.lower()}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved as {model_name.lower()}_confusion_matrix.png")
    
    return accuracy


def main():
    parser = argparse.ArgumentParser(description='Train ASL fingerspelling classifier')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to ASL MediaPipe dataset CSV file')
    parser.add_argument('--model_type', type=str, choices=['svm', 'rf', 'both'], default='both',
                        help='Model type to train: svm, rf (random forest), or both')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size (default: 0.2)')
    args = parser.parse_args()
    
    # Load and preprocess data
    X, y = load_and_preprocess_data(args.dataset_path)
    
    # Split data
    print(f"\nSplitting data: {1-args.test_size:.0%} train, {args.test_size:.0%} test")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compute class weights for imbalanced data
    class_weights = balance_classes(X_train, y_train)
    
    best_model = None
    best_accuracy = 0
    best_name = ""
    
    # Train SVM
    if args.model_type in ['svm', 'both']:
        svm_model = train_svm_model(X_train_scaled, y_train, class_weights)
        svm_accuracy = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
        
        if svm_accuracy > best_accuracy:
            best_model = svm_model
            best_accuracy = svm_accuracy
            best_name = "SVM"
    
    # Train Random Forest
    if args.model_type in ['rf', 'both']:
        rf_model = train_random_forest(X_train_scaled, y_train, class_weights)
        rf_accuracy = evaluate_model(rf_model, X_test_scaled, y_test, "RandomForest")
        
        if rf_accuracy > best_accuracy:
            best_model = rf_model
            best_accuracy = rf_accuracy
            best_name = "RandomForest"
    
    # Save best model and scaler
    print(f"\nBest model: {best_name} with accuracy {best_accuracy:.4f}")
    
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save model metadata
    model_info = {
        'model_type': best_name,
        'accuracy': best_accuracy,
        'feature_count': X.shape[1],
        'classes': sorted(np.unique(y).tolist()) if hasattr(np.unique(y), 'tolist') else list(sorted(np.unique(y))),
        'scaler_type': 'StandardScaler'
    }
    
    import json
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\nModel training complete!")
    print("Saved files:")
    print("  - model.pkl (trained classifier)")
    print("  - scaler.pkl (feature scaler)")
    print("  - model_info.json (model metadata)")
    print("  - confusion_matrix.png (evaluation plot)")
    
    print(f"\nFeature vector format for inference:")
    print("63 values: [x0,y0,z0, x1,y1,z1, ..., x20,y20,z20]")
    print("where (xi,yi,zi) are normalized coordinates of landmark i")


if __name__ == "__main__":
    main()