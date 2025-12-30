# Examples for statistical ML: Bayesian and Quantum Neural Networks 

This repository contains minimal implementations of:
1. **Bayesian Neural Network** (`simple_bnn_example.py`) - Uncertainty quantification using MCMC
2. **Hybrid Quantum-Classical Neural Network** (`minimal_hybrid.py`) - Quantum computing + classical ML

Both implementations are built from scratch using only `numpy` and `qiskit` to demystify the core concepts without framework abstractions.

---

## Table of Contents

1. [Bayesian Neural Network](#bayesian-neural-network-from-scratch)
2. [Hybrid Quantum-Classical Neural Network](#hybrid-quantum-classical-neural-network-minimal_hybridpy)
3. [Citation](#citation)

---

## Bayesian Neural Network 

This repository contains a "bare minimum" implementation of a Bayesian Neural Network (BNN) using only `numpy`. It demonstrates how to perform Bayesian inference on neural network weights without relying on deep learning frameworks.

### Key Concepts

*   **Standard NN (Optimization)**: Uses Gradient Descent to find the single "best" set of weights.
*   **Bayesian NN (Sampling)**: Uses MCMC to explore the probability landscape and learn the **Posterior Distribution** $P(w|D)$.

### Architecture
*   **Input**: 1 dimension
*   **Hidden**: 5 neurons (`tanh` activation)
*   **Output**: 1 dimension
*   **Total Parameters**: 17 (16 weights + 1 noise parameter)

### Usage

```bash
python simple_bnn_example.py
```

---

## Hybrid Quantum-Classical Neural Network (`minimal_hybrid.py`)

### Overview

A minimal implementation demonstrating how quantum computing can be integrated with classical neural networks for regression tasks. This hybrid approach combines:
- **Quantum path**: Creates complex, bounded feature representations using parametrized quantum circuits
- **Classical path**: Learns residual linear trends using traditional neurons
- **Joint training**: Both paths trained together to find optimal division of labor

### Architecture

The model uses a parallel hybrid architecture where the final prediction is the sum of a quantum circuit output and a classical neural network output.

```
Input: x (normalized scalar)

┌─ QUANTUM PATH (2 Layers) ──────────────────────┐
│ Layer 1: Ry(ax+b) → [CNOT] → Rx(cx+d) → Ry(ex+f)
│ Layer 2: Ry(gx+h) → [CNOT] → Rx(ix+j) → Ry(kx+l)
│            ↓                                    
│     Measure in Z-basis → P(|00⟩)...P(|11⟩)      
│            ↓                                    
│     Output: Σ w_qi · P(|i⟩)                   
│     (Weighted sum of basis state probabilities)
└─────────────────────────────────────────────────┘

┌─ CLASSICAL PATH ───────────────────────────────┐
│ x → tanh(w_c·x + b_c)                          │
└─────────────────────────────────────────────────┘

Final Output: quantum_output + classical_output
Total Parameters: 18
```

### Key Components

#### 1. Data Re-uploading
The input $x$ is encoded into the quantum circuit multiple times (in two layers). This allows the quantum circuit to approximate more complex functions (like cubic polynomials) by increasing the available frequency spectrum of the output.

#### 2. Phase Kickback
The architecture uses a 2-qubit system where the second qubit acts as an ancilla in the $|-\rangle$ state. Applying CNOT gates during the encoding process creates "phase kickback," introducing conditional dynamics that enhance the expressivity of the circuit beyond simple single-qubit rotations.

#### 3. Classical Residual Connection
A parallel classical path (a single neuron with `tanh` activation) is added to handle linear trends and offsets. This allows the quantum circuit to focus on learning the complex, non-linear parts of the data (the "wiggles"), while the classical path handles the broad strokes.

#### 4. Parameter Shift Rule
Gradients for the quantum parameters are computed exactly using the parameter shift rule:
$$ \frac{\partial L}{\partial \theta} = \frac{L(\theta + \pi/2) - L(\theta - \pi/2)}{2} $$

### Usage

```bash
python minimal_hybrid.py
```

---

## Citation

If you use this code or these concepts in your research, please cite this repository:

```bibtex
@software{sawant2025_quantum_classical_nn,
  title = {Minimal Hybrid Quantum-Classical Neural Network},
  author = {Sawant, Nilesh}, 
  year = {2025},
  url = {https://github.com/nileshsawant/bayesianQC},
  version={0.1},
  month={12}
}
```

## Requirements

```bash
pip install numpy matplotlib qiskit qiskit-aer
```
