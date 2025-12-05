"""
Demo script to test the ASL recognition system with synthetic data
Run this to verify the system works before using real ASL dataset
"""

import numpy as np
import pandas as pd
import os
from data_loader_and_training import main as train_main
import sys

def create_synthetic_asl_data(n_samples=1000):
    """Create synthetic ASL dataset for testing"""
    print("Creating synthetic ASL dataset for testing...")
    
    # 26 letters A-Z
    letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    # Generate synthetic 63-feature vectors (21 landmarks * 3 coords)
    np.random.seed(42)
    data = []
    
    for i, letter in enumerate(letters):
        # Generate samples for each letter with different patterns
        n_per_class = n_samples // len(letters)
        
        for _ in range(n_per_class):
            # Create synthetic landmark pattern for this letter
            # Each letter gets a slightly different base pattern + noise
            base_pattern = np.sin(np.arange(63) * (i + 1) * 0.1) * 0.3
            noise = np.random.normal(0, 0.1, 63)
            features = base_pattern + noise
            
            # Normalize to typical MediaPipe range [0, 1]
            features = (features - features.min()) / (features.max() - features.min())
            
            # Add this sample
            row = list(features) + [letter]
            data.append(row)
    
    # Create DataFrame
    columns = [f'landmark_{i//3}_{["x","y","z"][i%3]}' for i in range(63)] + ['label']
    df = pd.DataFrame(data, columns=columns)
    
    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Save synthetic dataset
    df.to_csv('synthetic_asl_dataset.csv', index=False)
    print(f"Synthetic dataset created: {len(df)} samples, {len(letters)} classes")
    return 'synthetic_asl_dataset.csv'

def demo_training():
    """Demo the training process with synthetic data"""
    print("\n" + "="*50)
    print("DEMO: ASL Fingerspelling Recognition System")
    print("="*50)
    
    # Create synthetic dataset
    dataset_path = create_synthetic_asl_data(2600)  # 100 samples per letter
    
    # Override sys.argv to pass arguments to training script
    original_argv = sys.argv
    sys.argv = [
        'data_loader_and_training.py',
        '--dataset_path', dataset_path,
        '--model_type', 'both',
        '--test_size', '0.2'
    ]
    
    try:
        # Run training
        print("\nStarting training process...")
        train_main()
        
        # Check if files were created
        expected_files = ['model.pkl', 'scaler.pkl', 'model_info.json']
        created_files = [f for f in expected_files if os.path.exists(f)]
        
        print(f"\nDemo completed successfully!")
        print(f"Created files: {created_files}")
        
        if len(created_files) == len(expected_files):
            print("\n✅ Training demo successful!")
            print("Now you can run 'python live_recognition.py' to test live recognition")
            print("(Note: This demo uses synthetic data, so predictions may not be meaningful)")
        else:
            print("⚠️  Some files missing. Check error messages above.")
            
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        # Restore original argv
        sys.argv = original_argv

if __name__ == "__main__":
    demo_training()