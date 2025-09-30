# TabPFN Reproduction Code
# This reproduces the original paper results on benchmark datasets

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tabpfn import TabPFNClassifier
import time
import os

# Create directories
os.makedirs('results', exist_ok=True)

print("✓ Setup complete")

# Helper Functions
def load_openml_dataset(dataset_id, max_samples=1000):
    """Load and preprocess OpenML dataset"""
    from sklearn.datasets import fetch_openml
    
    print(f"Loading OpenML dataset {dataset_id}...")
    data = fetch_openml(data_id=dataset_id, as_frame=True, parser='auto')
    X = data.data
    y = data.target
    
    # Handle categorical features
    for col in X.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Limit samples
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X = X.iloc[indices]
        y = y.iloc[indices]
    
    print(f"✓ Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def check_tabpfn_constraints(X, y):
    """Check if dataset meets TabPFN constraints"""
    n_samples, n_features = X.shape
    n_classes = len(np.unique(y))
    
    checks = {
        'samples': n_samples <= 1000,
        'features': n_features <= 100,
        'classes': n_classes <= 10
    }
    
    print(f"Constraint checks:")
    print(f"  Samples: {n_samples}/1000 {'✓' if checks['samples'] else '✗'}")
    print(f"  Features: {n_features}/100 {'✓' if checks['features'] else '✗'}")
    print(f"  Classes: {n_classes}/10 {'✓' if checks['classes'] else '✗'}")
    
    return all(checks.values())

# Main Reproduction Code
print("="*60)
print("PHASE 1: REPRODUCING ORIGINAL PAPER RESULTS")
print("="*60)

datasets_from_paper = {
    'credit-g': 31,
    'diabetes': 37,
    'vehicle': 54,
    'breast-w': 15
}

original_results = {}

for name, openml_id in datasets_from_paper.items():
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"{'='*60}")
    
    try:
        X, y = load_openml_dataset(openml_id, max_samples=1000)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        if not check_tabpfn_constraints(X, y_encoded):
            print("⚠️ Dataset violates constraints, skipping...")
            continue
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.5, random_state=42, stratify=y_encoded
        )
        
        print("\nRunning TabPFN...")
        start = time.time()
        clf = TabPFNClassifier(device='cpu')
        clf.fit(X_train.values, y_train)
        y_pred = clf.predict(X_test.values)
        tabpfn_time = time.time() - start
        
        acc = accuracy_score(y_test, y_pred)
        
        original_results[name] = {
            'Accuracy': acc,
            'Time(s)': tabpfn_time,
            'N_samples': len(X_train),
            'N_features': X_train.shape[1],
            'N_classes': len(np.unique(y_encoded))
        }
        
        print(f"✓ Accuracy: {acc:.4f}")
        print(f"✓ Time: {tabpfn_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        continue

# Save results
print("\n" + "="*60)
print("REPRODUCTION RESULTS")
print("="*60)
original_df = pd.DataFrame(original_results).T
print(original_df)
original_df.to_csv('results/original_reproduction.csv')
print("\n✓ Results saved to results/original_reproduction.csv")
