# Student Performance Prediction using TabPFN and Baseline Models

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time
import os

print("="*60)
print("PHASE 2: STUDENT PERFORMANCE PREDICTION")
print("="*60)

# Load dataset (make sure student-por.csv is in the same directory)
filename = 'student-por.csv'
df_new = pd.read_csv(filename)

print(f"\nDataset shape: {df_new.shape}")

# Convert G3 to binary classification
print(f"\nOriginal target (G3) distribution:")
print(df_new['G3'].value_counts().sort_index())

df_new['pass'] = (df_new['G3'] >= 10).astype(int)

print(f"\nConverted to binary classification:")
print(f"  Pass (G3 >= 10): {(df_new['pass'] == 1).sum()} students")
print(f"  Fail (G3 < 10): {(df_new['pass'] == 0).sum()} students")

TARGET_COLUMN = 'pass'

# Drop G1, G2, G3 to avoid data leakage
X_new = df_new.drop(columns=['G1', 'G2', 'G3', TARGET_COLUMN])
y_new = df_new[TARGET_COLUMN]

# Encode categorical features
print("\nConverting categorical features...")
for col in X_new.select_dtypes(include=['object', 'category']).columns:
    le = LabelEncoder()
    X_new[col] = le.fit_transform(X_new[col].astype(str))
    print(f"  Encoded: {col}")

X_new = X_new.fillna(X_new.mean())

le_target = LabelEncoder()
y_new_encoded = le_target.fit_transform(y_new)

print(f"\n✓ Final shape: {X_new.shape}")
print(f"✓ Number of classes: {len(np.unique(y_new_encoded))}")

# Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X_new, y_new_encoded, test_size=0.3, random_state=42, stratify=y_new_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'TabPFN': (TabPFNClassifier(device='cpu'), False),
    'XGBoost': (XGBClassifier(random_state=42, eval_metric='logloss'), True),
    'LightGBM': (LGBMClassifier(random_state=42, verbose=-1), True),
    'RandomForest': (RandomForestClassifier(n_estimators=100, random_state=42), True),
    'LogisticReg': (LogisticRegression(max_iter=1000, random_state=42), True)
}

results = {}

for name, (model, use_scaled) in models.items():
    print(f"\nTraining: {name}")
    
    try:
        start = time.time()
        
        if use_scaled:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)
        else:
            model.fit(X_train.values, y_train)
            y_pred = model.predict(X_test.values)
            y_proba = model.predict_proba(X_test.values)
        
        train_time = time.time() - start
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba[:, 1])
        
        results[name] = {
            'Accuracy': acc,
            'ROC-AUC': roc,
            'Time(s)': train_time
        }
        
        print(f"✓ Accuracy: {acc:.4f}, ROC-AUC: {roc:.4f}, Time: {train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

os.makedirs('results', exist_ok=True)
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(results_df)
results_df.to_csv('results/new_dataset_results.csv')
print("\n✓ Saved to results/new_dataset_results.csv")
