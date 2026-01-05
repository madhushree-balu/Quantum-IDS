import numpy as np
import os

PROCESSED_DIR = 'data/processed'

# Load all data
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train_standard.npy'))
X_val = np.load(os.path.join(PROCESSED_DIR, 'X_val_standard.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test_standard.npy'))

y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'))
y_val = np.load(os.path.join(PROCESSED_DIR, 'y_val.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))

print("ACTUAL PREPROCESSED DATA SIZES:")
print(f"  X_train: {len(X_train):,} samples")
print(f"  X_val:   {len(X_val):,} samples")
print(f"  X_test:  {len(X_test):,} samples")
print(f"  TOTAL:   {len(X_train) + len(X_val) + len(X_test):,} samples")

print("\nPER-CLASS DISTRIBUTION (Training):")
for i in range(len(np.unique(y_train))):
    count = np.sum(y_train == i)
    print(f"  Class {i}: {count:,} samples ({count/len(y_train)*100:.1f}%)")

print("\nMAXIMUM SAFE SAMPLING:")
print(f"  Max train samples: {len(X_train):,}")
print(f"  Safe range: up to {int(len(X_train) * 0.9):,} (90% of available)")