# Hybrid Quantum-Classical Intrusion Detection System
#### Recommended: Python 3.10

## Install Required Libraries
```
# Upgrade pip
pip install --upgrade pip

# Install core libraries
pip install numpy pandas matplotlib seaborn scikit-learn

# Install Qiskit (quantum computing)
pip install qiskit==0.45.0
pip install qiskit-aer==0.13.0
pip install qiskit-machine-learning==0.7.0

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
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_classical_baseline.ipynb
│   ├── 03_quantum_kernel.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── classical_models.py
│   ├── quantum_models.py
│   └── utils.py
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