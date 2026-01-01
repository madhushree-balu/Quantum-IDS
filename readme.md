# Hybrid Quantum-Classical Intrusion Detection System
#### Recommended: Python 3.10

## Install Required Libraries
```
# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Install Qiskit (quantum computing)
pip install qiskit
pip install qiskit-aer
pip install qiskit-machine-learning

# Install GPU-accelerated libraries (optional but recommended)
pip install tensorflow-gpu  # or just tensorflow for latest
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Jupyter for notebooks (optional)
pip install jupyter notebook
pip install ipykernel

# Install GPU-accelerated Qiskit backend
pip install qiskit-aer-gpu  # Uses your RTX 3050!
```

## **Project Structure (Local)**
```
quantum_ids_project/
├── data/
│   ├── raw/
│   │   └── kddcup.data_10_percent
│   └── processed/
│       ├── X_train.npy
│       └── y_train.npy
├── src/
│   ├── 01_preprocessing.py
│   ├── 02_classical_baseline.py
│   ├── 03_quantum_kernel.py
│   └── 04_analysis.py
├── results/
│   ├── classical_results.csv
│   ├── quantum_results.csv
│   └── figures/
├── models/
│   └── saved_models/
├── paper/
│   └── manuscript.tex
└── requirements.txt
```
