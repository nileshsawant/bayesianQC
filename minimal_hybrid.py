"""
Minimal Hybrid Quantum-Classical Neural Network
================================================

A minimal implementation demonstrating hybrid quantum-classical machine learning
for regression tasks. This architecture combines quantum computing's ability to
create complex feature spaces with classical neural networks' capacity for learning
linear trends.

Architecture Design:
-------------------
  Input: x (single scalar, normalized)
  
  ┌─ QUANTUM PATH ─────────────────────────────────┐
  │ x → Ry(ax+b) → Rx(cx+d) → Ry(ex+f)            │  6 encoding params
  │            ↓                                    │
  │     Measure in Z-basis → P(|0⟩), P(|1⟩)       │
  │            ↓                                    │
  │     Output: w_q0·P(|0⟩) + w_q1·P(|1⟩)         │  2 output weights
  └─────────────────────────────────────────────────┘
  
  ┌─ CLASSICAL PATH ───────────────────────────────┐
  │ x → tanh(w_c·x + b_c)                          │  2 params
  └─────────────────────────────────────────────────┘
  
  Final Output: quantum_output + classical_output

Total Parameters: 10
  - 6 quantum encoding (a, b, c, d, e, f)
  - 2 quantum output weights (w_q0, w_q1)
  - 2 classical neuron (w_c, b_c)

Key Design Choices:
------------------
1. **Ry-Rx-Ry gate sequence**: All three rotations affect Z-basis measurement
   (Note: Rz doesn't affect computational basis measurements!)
   
2. **Normalized inputs**: Both paths operate on standardized data (mean=0, std=1)
   ensuring quantum and classical contributions are on similar scales
   
3. **Bounded activations**: 
   - Quantum: Probabilities bounded to [0,1], weighted output bounded
   - Classical: tanh bounds output to [-1,1]
   
4. **Complementary learning**:
   - Quantum learns complex, bounded, periodic patterns via rotations
   - Classical learns residual linear/smooth trends
   
5. **Parameter shift rule**: Used for quantum gradients (exact, not finite difference)

Mathematical Background:
-----------------------
For a single qubit with Ry-Rx-Ry rotations:
  |ψ⟩ = Ry(e·x+f) Rx(c·x+d) Ry(a·x+b) |0⟩
  
Measurement probabilities:
  P(|0⟩) = |⟨0|ψ⟩|²
  P(|1⟩) = |⟨1|ψ⟩|²
  
Quantum output:
  y_quantum = w_q0·P(|0⟩) + w_q1·P(|1⟩)
  
Classical output:
  y_classical = tanh(w_c·x + b_c)
  
Total output:
  y_pred = y_quantum + y_classical

Training:
--------
- Loss: Mean Squared Error (MSE)
- Optimizer: Gradient Descent with Momentum (β=0.9)
- Quantum gradients: Parameter shift rule (shift = π/2)
- Classical gradients: Analytical derivatives
- Gradient clipping: Norm clipped to 1.0 for stability

Performance:
-----------
On y = x² + 1 + noise(σ=0.1):
  - R² ≈ 0.35 with 10 parameters
  - Quantum contributes structured patterns (std ~ 0.125)
  - Classical contributes linear trends (std ~ 0.60)
  - Training time: ~10 min (50 epochs on GPU)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
import time


class MinimalHybrid:
    """
    Minimal Hybrid Quantum-Classical Neural Network.
    
    This class implements the simplest possible hybrid architecture that
    demonstrates the key principles of quantum-classical machine learning:
    
    1. Quantum path uses parametrized quantum circuit (PQC) for feature mapping
    2. Classical path uses traditional neural network for residual learning
    3. Both paths trained jointly with gradient-based optimization
    
    The quantum circuit uses Ry-Rx-Ry pattern because:
    - All three gates affect Z-basis measurement (unlike Rz which only adds phase)
    - Provides sufficient expressiveness for non-linear function approximation
    - Still simple enough to train efficiently
    
    Parameters are initialized small (std=0.1) to avoid getting stuck in
    local minima and to prevent gradient explosion in early training.
    """
    
    def __init__(self, seed=42):
        """
        Initialize minimal hybrid quantum-classical network.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Quantum encoding: 2 qubits with phase kickback
        # Qubit 0: Ry(a*x+b) → [CNOT control] → Rx(c*x+d) → Ry(e*x+f)
        # Qubit 1: X → H → [CNOT target] (ancilla in |-⟩ state)
        # Phase kickback: qubit 1's state creates phase on qubit 0
        # Total: 6 encoding parameters (only qubit 0 is parameterized)
        self.quantum_params = np.random.randn(6) * 0.1  # [a,b,c,d,e,f]
        
        # Quantum output weights: map 4 basis states to scalar
        # |00⟩, |01⟩, |10⟩, |11⟩ → weighted combination
        # Equal initialization (0.25 each) lets network learn the mapping
        self.w_q = np.array([0.25, 0.25, 0.25, 0.25])  # weights for |00⟩, |01⟩, |10⟩, |11⟩
        
        # Classical neuron parameters
        # Initialized at zero → starts as pure quantum model
        # Network learns to add classical component only if needed
        self.w_c = 0.0  # classical weight (slope)
        self.b_c = 0.0  # classical bias (offset)
        
        # Momentum buffers for all 12 parameters
        # Momentum helps smooth out noisy gradients and accelerate convergence
        # Format: [quantum_params(6), w_q(4), w_c, b_c]
        self.momentum = np.zeros(12)
        
        # GPU simulator setup
        # Tries GPU first for speed, falls back to CPU if unavailable
        try:
            self.simulator = AerSimulator(method='statevector', device='GPU')
            print("✓ GPU acceleration enabled")
        except:
            self.simulator = AerSimulator(method='statevector')
            print("✓ CPU simulator")
    
    def quantum_path(self, x):
        """
        Execute quantum path: 2-qubit entangled circuit.
        
        Architecture:
        - Qubit 0: Ry-Rx-Ry encoding of input x
        - Qubit 1: Ry-Rx-Ry encoding of input x (different parameters)
        - CNOT(0,1): Entangles the two qubits
        
        The CNOT creates quantum correlations between qubits:
        - If q0 is |1⟩, flips q1
        - Creates superposition over 4 basis states |00⟩, |01⟩, |10⟩, |11⟩
        - Much richer feature space than single qubit
        
        All rotation gates affect Z-basis measurement probabilities,
        unlike Rz which only adds global phase (invisible in measurements).
        
        Args:
            x (float): Normalized input value
            
        Returns:
            tuple: (weighted_output, probabilities_dict)
                - weighted_output: Σ w_i * P(|i⟩)
                - probabilities_dict: {0: P(|00⟩), 1: P(|01⟩), 2: P(|10⟩), 3: P(|11⟩)}
        """
        # Compute rotation angles for qubit 0
        angle_q0_y1 = self.quantum_params[0] * x + self.quantum_params[1]
        angle_q0_x = self.quantum_params[2] * x + self.quantum_params[3]
        angle_q0_y2 = self.quantum_params[4] * x + self.quantum_params[5]
        
        # Build 2-qubit circuit with phase kickback
        qc = QuantumCircuit(2)
        
        # Qubit 0: First rotation
        qc.ry(angle_q0_y1, 0)
        
        # Qubit 1: Prepare ancilla in |-⟩ state for phase kickback
        qc.x(1)  # |0⟩ → |1⟩
        qc.h(1)  # |1⟩ → |-⟩ = (|0⟩ - |1⟩)/√2
        
        # CNOT with qubit 0 as control, qubit 1 as target
        # This creates phase kickback: the state of qubit 1 affects phase on qubit 0
        qc.cx(0, 1)
        
        # Qubit 0: Continue encoding (affected by phase kickback)
        qc.rx(angle_q0_x, 0)
        qc.ry(angle_q0_y2, 0)
        
        qc.save_probabilities()
        
        # Execute circuit
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.simulator)
        transpiled = pm.run(qc)
        result = self.simulator.run(transpiled, shots=1000).result()
        probs = result.data()['probabilities']
        
        # Map 4 basis states to output using learned weights
        # |00⟩=0, |01⟩=1, |10⟩=2, |11⟩=3
        # probs is a numpy array of length 4 (for 2 qubits)
        output = sum(self.w_q[i] * probs[i] for i in range(4))
        return output, probs
    
    def classical_path(self, x):
        """
        Execute classical path: linear transformation + tanh activation.
        
        The tanh activation provides:
        1. Non-linearity for learning complex patterns
        2. Bounded output [-1, 1] matching quantum output scale
        3. Smooth gradients (no vanishing gradient at extremes like sigmoid)
        
        Args:
            x (float): Normalized input value
            
        Returns:
            float: tanh(w_c * x + b_c)
        """
        return np.tanh(self.w_c * x + self.b_c)
    
    def forward(self, x):
        """
        Forward pass: combine quantum and classical paths.
        
        This implements the core hybrid idea:
        - Quantum circuit creates complex feature representation
        - Classical neuron learns residual/trend component
        - Sum gives final prediction
        
        The additive combination allows:
        1. Quantum to focus on bounded, periodic patterns
        2. Classical to handle linear trends and offsets
        3. Joint training to find optimal division of labor
        
        Args:
            x (float): Normalized input value
            
        Returns:
            float: y_pred = quantum_output + classical_output
        """
        q_out, _ = self.quantum_path(x)
        c_out = self.classical_path(x)
        return q_out + c_out
    
    def compute_gradient(self, x, y_true, shift=np.pi/2):
        """
        Compute gradients for all 10 parameters using hybrid approach.
        
        Gradient Computation Strategy:
        ------------------------------
        1. **Quantum parameters (12)**: Parameter shift rule
           - Exact gradient, no approximation
           - For parameter θ: ∂L/∂θ = [L(θ+π/2) - L(θ-π/2)] / 2
           - Requires 2 circuit evaluations per parameter
           
        2. **Quantum output weights (4)**: Analytical
           - ∂L/∂w_i = 2(y_pred - y_true) * P(|i⟩) for i ∈ {00,01,10,11}
           
        3. **Classical parameters (2)**: Analytical
           - Standard backpropagation through tanh
           - ∂L/∂w_c = 2(y_pred - y_true) * (1 - tanh²) * x
           - ∂L/∂b_c = 2(y_pred - y_true) * (1 - tanh²)
        
        The parameter shift rule is crucial for quantum ML:
        - Works for any quantum gate with eigenvalues ±1
        - Gives exact gradient (not numerical approximation)
        - Shift of π/2 is optimal for single-parameter gates
        
        Args:
            x (float): Input value
            y_true (float): Target value
            shift (float): Parameter shift amount (default π/2)
            
        Returns:
            tuple: (gradient_vector, loss_value)
                - gradient_vector: np.array of shape (12,)
                - loss_value: scalar MSE loss
        """
        y_pred = self.forward(x)
        loss = (y_pred - y_true)**2
        gradient = np.zeros(12)
        
        dloss_dpred = 2 * (y_pred - y_true)  # Derivative of MSE
        
        # Get current quantum output and probabilities
        q_out, probs = self.quantum_path(x)
        c_out = self.classical_path(x)
        
        # ============================================================
        # QUANTUM PARAMETER GRADIENTS (Parameter Shift Rule)
        # ============================================================
        # For each quantum encoding parameter (6 total), evaluate circuit at θ±π/2
        for i in range(6):
            original = self.quantum_params[i]
            
            # Forward pass with θ + π/2
            self.quantum_params[i] = original + shift
            y_plus = self.forward(x)
            
            # Forward pass with θ - π/2
            self.quantum_params[i] = original - shift
            y_minus = self.forward(x)
            
            # Restore original value
            self.quantum_params[i] = original
            
            # Compute gradient using parameter shift formula
            # Note: Division by 2*sin(shift) = 2*sin(π/2) = 2
            gradient[i] = ((y_plus - y_true)**2 - (y_minus - y_true)**2) / (2 * np.sin(shift))
        
        # ============================================================
        # QUANTUM OUTPUT WEIGHT GRADIENTS (Analytical)
        # ============================================================
        # These are classical parameters acting on quantum probabilities
        # 4 basis states: |00⟩, |01⟩, |10⟩, |11⟩
        for i in range(4):
            gradient[6 + i] = dloss_dpred * probs[i]  # ∂L/∂w_q[i]
        
        # ============================================================
        # CLASSICAL PARAMETER GRADIENTS (Analytical)
        # ============================================================
        # Standard backpropagation through tanh activation
        dtanh = 1 - c_out**2  # Derivative of tanh
        gradient[10] = dloss_dpred * dtanh * x  # ∂L/∂w_c
        gradient[11] = dloss_dpred * dtanh       # ∂L/∂b_c
        
        return gradient, loss


def train(model, X_train, y_train, epochs=50, lr=0.05, momentum=0.9):
    """
    Train hybrid model using gradient descent with momentum.
    
    Training Strategy:
    -----------------
    1. **Stochastic updates**: Update after each sample (online learning)
    2. **Momentum**: Smooths gradient descent, accelerates convergence
    3. **Gradient clipping**: Prevents explosion, max norm = 1.0
    4. **Learning rate**: Fixed at 0.05 (could use scheduling for better results)
    
    The momentum update rule:
        v_t = β*v_{t-1} + (1-β)*g_t
        θ_t = θ_{t-1} - α*v_t
    
    where:
        - v_t: momentum velocity
        - g_t: gradient at time t
        - β: momentum coefficient (0.9)
        - α: learning rate (0.05)
    
    Why momentum helps:
    - Quantum gradients can be noisy (1000 shots, not infinite)
    - Momentum averages out noise over multiple steps
    - Accelerates convergence in consistent gradient directions
    
    Args:
        model: MinimalHybrid instance
        X_train: Training inputs (normalized)
        y_train: Training targets (normalized)
        epochs: Number of passes through dataset
        lr: Learning rate (step size)
        momentum: Momentum coefficient (β)
        
    Returns:
        list: Training losses per epoch
    """
    losses = []
    
    print("\nTraining:")
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        total_loss = 0
        
        # Loop over all training samples
        for x, y in zip(X_train, y_train):
            # Compute gradients using parameter shift + backprop
            grad, loss = model.compute_gradient(x, y)
            total_loss += loss
            
            # Gradient clipping for stability
            # Prevents exploding gradients in early training
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1.0:
                grad = grad / grad_norm
            
            # Momentum update
            # Accumulate gradient direction over time
            model.momentum = momentum * model.momentum + (1 - momentum) * grad
            
            # Update all parameters
            # Quantum encoding parameters (6)
            for i in range(6):
                model.quantum_params[i] -= lr * model.momentum[i]
            
            # Quantum output weights (4)
            for i in range(4):
                model.w_q[i] -= lr * model.momentum[6 + i]
            
            # Classical parameters (2)
            model.w_c -= lr * model.momentum[10]
            model.b_c -= lr * model.momentum[11]
        
        # Track average loss per epoch
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        
        # Print progress every 5 epochs
        epoch_time = time.time() - epoch_start
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f} ({epoch_time:.1f}s)")
    
    return losses


def main():
    print("="*70)
    print("MINIMAL HYBRID: 1 QUBIT + 1 NEURON")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    n_samples = 20
    X = np.linspace(-2, 2, n_samples)
    y = X**2 + 1 + np.random.normal(0, 0.1, n_samples)
    
    print(f"\nData: y = x² + 1 + noise(σ=0.1)")
    print(f"  Samples: {n_samples}")
    print(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"  y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Normalize data
    X_mean, X_std = X.mean(), X.std()
    X_scaled = (X - X_mean) / X_std
    
    y_mean, y_std = y.mean(), y.std()
    y_scaled = (y - y_mean) / y_std
    
    print(f"\nScaled ranges:")
    print(f"  X: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"  y: [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")
    
    # Create model
    print("\nArchitecture:")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ Quantum Path:  2 qubits with phase kickback    │ 10 params")
    print("  │   Qubit 0: Ry(ax+b)→[CNOT]→Rx(cx+d)→Ry(ex+f)  │")
    print("  │   Qubit 1: X→H→[CNOT target] (ancilla in |-⟩) │")
    print("  │      ↓                                          │")
    print("  │   Output: Σ wᵢ·P(|i⟩) for |00⟩,|01⟩,|10⟩,|11⟩ │")
    print("  └─────────────────────────────────────────────────┘")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ Classical Path: x → tanh(w·x + b)              │  2 params")
    print("  └─────────────────────────────────────────────────┘")
    print("  Output: quantum + classical")
    print(f"  Total: 12 parameters (6 encoding + 4 output + 2 classical)")
    
    model = MinimalHybrid(seed=42)
    
    print(f"\nInitial parameters:")
    print(f"  Quantum params: {model.quantum_params}")
    print(f"  Quantum output: {model.w_q}")
    print(f"  Classical: w_c={model.w_c:.4f}, b_c={model.b_c:.4f}")
    
    # Train
    start_time = time.time()
    losses = train(model, X_scaled, y_scaled, epochs=50, lr=0.05, momentum=0.9)
    train_time = time.time() - start_time
    
    # Evaluate
    predictions_scaled = np.array([model.forward(x) for x in X_scaled])
    predictions = predictions_scaled * y_std + y_mean
    
    mse = np.mean((predictions - y)**2)
    r2 = 1 - np.sum((y - predictions)**2) / np.sum((y - np.mean(y))**2)
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Training time: {train_time:.1f}s")
    print(f"Final loss (scaled): {losses[-1]:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    print(f"\nLearned parameters:")
    print(f"  Quantum encoding: {model.quantum_params}")
    print(f"  Quantum output weights: {model.w_q}")
    print(f"  Classical: w_c={model.w_c:.4f}, b_c={model.b_c:.4f}")
    
    # Analyze contributions
    quantum_outputs = []
    classical_outputs = []
    for x in X_scaled:
        q_out, _ = model.quantum_path(x)
        c_out = model.classical_path(x)
        quantum_outputs.append(q_out)
        classical_outputs.append(c_out)
    
    quantum_outputs = np.array(quantum_outputs)
    classical_outputs = np.array(classical_outputs)
    
    print(f"\nPath Analysis:")
    print(f"  Quantum contribution:  mean={quantum_outputs.mean():.4f}, std={quantum_outputs.std():.4f}")
    print(f"  Classical contribution: mean={classical_outputs.mean():.4f}, std={classical_outputs.std():.4f}")
    
    baseline_mse = np.var(y)
    print(f"\nBaseline MSE: {baseline_mse:.4f}")
    print(f"Improvement: {(1 - mse/baseline_mse)*100:.1f}%")
    
    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss
    axes[0,0].plot(range(1, len(losses)+1), losses, 'b-', linewidth=2)
    axes[0,0].set_xlabel('Epoch', fontsize=12)
    axes[0,0].set_ylabel('Loss (scaled)', fontsize=12)
    axes[0,0].set_title('Training Progress', fontsize=14, fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Predictions
    axes[0,1].scatter(X, y, color='blue', s=80, alpha=0.6, label='Data', zorder=3)
    axes[0,1].plot(X, predictions, 'r-', linewidth=2, label='Hybrid', zorder=2)
    
    X_dense = np.linspace(-2, 2, 100)
    y_true = X_dense**2 + 1
    axes[0,1].plot(X_dense, y_true, 'g--', linewidth=1.5, alpha=0.5, 
                   label='True', zorder=1)
    
    axes[0,1].set_xlabel('x', fontsize=12)
    axes[0,1].set_ylabel('y', fontsize=12)
    axes[0,1].set_title(f'Predictions (R² = {r2:.3f})', fontsize=14, fontweight='bold')
    axes[0,1].legend(fontsize=10)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Quantum vs Classical contributions
    axes[1,0].plot(X, quantum_outputs * y_std, 'purple', linewidth=2, 
                   label='Quantum path', marker='o')
    axes[1,0].plot(X, classical_outputs * y_std, 'orange', linewidth=2, 
                   label='Classical path', marker='s')
    axes[1,0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1,0].set_xlabel('x', fontsize=12)
    axes[1,0].set_ylabel('Contribution', fontsize=12)
    axes[1,0].set_title('Path Decomposition', fontsize=14, fontweight='bold')
    axes[1,0].legend(fontsize=10)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Residuals
    residuals = y - predictions
    axes[1,1].scatter(X, residuals, color='red', s=80, alpha=0.6)
    axes[1,1].axhline(y=0, color='k', linestyle='--', linewidth=2)
    axes[1,1].set_xlabel('x', fontsize=12)
    axes[1,1].set_ylabel('Residual', fontsize=12)
    axes[1,1].set_title('Residuals', fontsize=14, fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_file = 'minimal_hybrid_result.png'
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_file}")
    
    print("\n" + "="*70)
    print("Summary:")
    print(f"  • 2-qubit phase kickback architecture: 12 params")
    print(f"  • Quantum: Qubit 0 (Ry-Rx-Ry) + CNOT + Qubit 1 (X-H ancilla)")
    print(f"  • Phase kickback: ancilla in |-⟩ creates phase on qubit 0")
    print(f"  • Output: 4 basis states |00⟩, |01⟩, |10⟩, |11⟩")
    print(f"  • Classical: tanh learns residual linear trends")
    print("="*70)


if __name__ == "__main__":
    main()
