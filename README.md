# ASL Fingerspelling Recognition System (Classical ML)

A complete real-time ASL fingerspelling recognition system using **MediaPipe hand landmarks** and **classical machine learning** (SVM/Random Forest). No deep learning frameworks required.

## Features

✅ **Classical ML Only**: Uses SVM/Random Forest (no TensorFlow/PyTorch)  
✅ **MediaPipe Integration**: 21 hand landmarks → 63 features (x,y,z)  
✅ **Real-time Recognition**: Live webcam with smoothing & FPS display  
✅ **Class Balancing**: Handles imbalanced datasets automatically  
✅ **CPU Optimized**: Runs in real-time on standard hardware  

## Project Structure

```
ASL_Fingerspelling_ML/
├── data_loader_and_training.py    # Train SVM/RF model on ASL dataset
├── live_recognition.py            # Real-time webcam recognition
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── model.pkl                      # Trained classifier (generated)
├── scaler.pkl                     # Feature scaler (generated)
├── model_info.json               # Model metadata (generated)
└── confusion_matrix.png           # Evaluation plot (generated)
```

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download ASL MediaPipe Dataset**:
   - Get "ASL Mediapipe Converted Dataset" from Kaggle
   - Should contain 63 columns (21 landmarks × 3 coordinates) + label column
   - Save as CSV file

## Usage

### Step 1: Train the Model

```bash
python data_loader_and_training.py --dataset_path path/to/your/asl_dataset.csv
```

**Options**:
- `--model_type svm|rf|both` (default: both) - Train SVM, Random Forest, or both
- `--test_size 0.2` (default: 0.2) - Test set percentage

**Output**:
- `model.pkl` - Best trained classifier
- `scaler.pkl` - Feature normalization scaler  
- `model_info.json` - Model metadata
- `confusion_matrix.png` - Evaluation plots

### Step 2: Run Live Recognition

```bash
python live_recognition.py
```

**Controls**:
- Show your hand to webcam for ASL letter recognition
- Press `q` to quit
- Press `s` to save current frame

## Dataset Format

The CSV dataset should contain:
- **63 feature columns**: `x0,y0,z0,x1,y1,z1,...,x20,y20,z20`
- **1 label column**: `label` or `class` (A-Z letters)

Each row represents one hand pose with normalized landmark coordinates.

## Model Performance

**Expected Results**:
- **SVM (RBF)**: 85-95% accuracy on test set
- **Random Forest**: 80-90% accuracy on test set
- **Real-time FPS**: 15-30 FPS (depending on hardware)

**Features**:
- Automatic class balancing for imbalanced data
- Feature standardization (zero mean, unit variance)  
- Temporal smoothing (10-frame buffer) reduces flickering
- Confidence scores for predictions

## Algorithm Details

### Training Pipeline:
1. **Load Dataset** → 63 landmark features + labels
2. **Train/Test Split** → 80/20 stratified split
3. **Feature Scaling** → StandardScaler normalization
4. **Class Balancing** → Compute balanced class weights
5. **Model Training** → SVM (RBF) + Random Forest
6. **Evaluation** → Accuracy, confusion matrix, classification report
7. **Save Models** → Best model + scaler as .pkl files

### Live Recognition Pipeline:
1. **Webcam Capture** → OpenCV video stream
2. **Hand Detection** → MediaPipe Hands (21 landmarks)
3. **Feature Extraction** → Convert to 63-value vector
4. **Preprocessing** → Apply saved scaler transform
5. **Prediction** → Trained model inference
6. **Smoothing** → 10-frame majority vote buffer
7. **Display** → Letter prediction + confidence + FPS

## Troubleshooting

### No hands detected:
- Ensure good lighting
- Keep hand clearly visible in frame
- Check MediaPipe detection confidence (adjust in code)

### Low accuracy:
- Verify dataset format (63 features expected)
- Check class distribution (use `--model_type both` for comparison)
- Increase training data size

### Performance issues:
- Reduce MediaPipe detection confidence
- Lower webcam resolution
- Use fewer smoothing frames

## Technical Specifications

**Dependencies**: scikit-learn, mediapipe, opencv, pandas, numpy  
**Models**: SVM (RBF kernel) or Random Forest  
**Features**: 63D vector (21 hand landmarks × x,y,z)  
**Classes**: 26 letters (A-Z)  
**Platform**: CPU-only, cross-platform  

## Example Output

```
Loading dataset from asl_dataset.csv
Dataset shape: (50000, 63)
Classes: ['A', 'B', 'C', ..., 'Z']

Training SVM with RBF kernel...
Training Random Forest...

SVM Results:
Test Accuracy: 0.9234

Best model: SVM with accuracy 0.9234
Saved files:
  - model.pkl (trained classifier)  
  - scaler.pkl (feature scaler)
  - model_info.json (model metadata)
```

**Live Recognition**:
```
Starting live ASL recognition...
Model: SVM, Accuracy: 0.923

[Webcam opens showing:]
Prediction: A
Confidence: 89.2%
FPS: 24.1
```

## License

MIT License - Feel free to use and modify for your projects.