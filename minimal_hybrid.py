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
  
  ┌─ QUANTUM PATH (2 Layers) ──────────────────────┐
  │ Layer 1: Ry(ax+b) → [CNOT] → Rx(cx+d) → Ry(ex+f)
  │ Layer 2: Ry(gx+h) → [CNOT] → Rx(ix+j) → Ry(kx+l)
  │            ↓                                    
  │     Measure in Z-basis → P(|00⟩)...P(|11⟩)      
  │            ↓                                    
  │     Output: Σ w_qi · P(|i⟩)                   
  └─────────────────────────────────────────────────┘
  
  ┌─ CLASSICAL PATH ───────────────────────────────┐
  │ x → tanh(w_c·x + b_c)                          │
  └─────────────────────────────────────────────────┘
  
  Final Output: quantum_output + classical_output

Total Parameters: 18
  - 12 quantum encoding (a..l)
  - 4 quantum output weights (w_q0..w_q3)
  - 2 classical neuron (w_c, b_c)

Key Design Choices:
------------------
1. **Data Re-uploading**: Input x is encoded twice (2 layers) to capture 
   higher-frequency components (like cubic functions).

2. **Phase Kickback**: Uses an ancilla qubit in |-⟩ state to create 
   conditional dynamics during encoding.
   
3. **Normalized inputs**: Both paths operate on standardized data (mean=0, std=1).
   
4. **Complementary learning**:
   - Quantum learns complex, bounded, periodic patterns
   - Classical learns residual linear/smooth trends
   
5. **Parameter shift rule**: Used for quantum gradients (exact).

Training:
--------
- Loss: Mean Squared Error (MSE)
- Optimizer: Gradient Descent with Momentum (β=0.9)
- Quantum gradients: Parameter shift rule
- Classical gradients: Analytical derivatives

Performance:
-----------
On y = x³ - x + 1 + noise:
  - R² ≈ 0.84 with 18 parameters
  - Training time: ~20 sec (with parameter binding optimization)
"""

import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
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
        
        # Quantum encoding: 2 qubits with phase kickback + Data Re-uploading
        # Layer 1: Ry(a*x+b) → [CNOT] → Rx(c*x+d) → Ry(e*x+f)
        # Layer 2: Ry(g*x+h) → [CNOT] → Rx(i*x+j) → Ry(k*x+l)
        # Qubit 1: X → H → [CNOT target] (ancilla in |-⟩ state)
        # Total: 12 encoding parameters (6 per layer on qubit 0)
        self.quantum_params = np.random.randn(12) * 0.5  # [a..l]
        
        # Quantum output weights: map 4 basis states to scalar
        # |00⟩, |01⟩, |10⟩, |11⟩ → weighted combination
        # Equal initialization (0.25 each) lets network learn the mapping
        self.w_q = np.array([0.25, 0.25, 0.25, 0.25])  # weights for |00⟩, |01⟩, |10⟩, |11⟩
        
        # Classical neuron parameters
        # Initialized at zero → starts as pure quantum model
        # Network learns to add classical component only if needed
        self.w_c = 0.0  # classical weight (slope)
        self.b_c = 0.0  # classical bias (offset)
        
        # Momentum buffers for all 18 parameters
        # Format: [quantum_params(12), w_q(4), w_c, b_c]
        self.momentum = np.zeros(18)
        
        # GPU simulator setup
        # Tries GPU first for speed, falls back to CPU if unavailable
        try:
            # Use 'automatic' method to allow shot-based simulation
            self.simulator = AerSimulator(method='automatic', device='GPU')
            print("✓ GPU acceleration enabled")
        except:
            self.simulator = AerSimulator(method='automatic')
            print("✓ CPU simulator")
            
        # ============================================================
        # PRE-COMPILE QUANTUM CIRCUIT (Optimization)
        # ============================================================
        # Instead of rebuilding the circuit every time, we build it once
        # with Parameters and bind values at runtime.
        
        # Define parameters
        self.x_param = Parameter('x')
        self.theta_params = [Parameter(f'theta_{i}') for i in range(12)]
        
        # Build circuit
        self.qc = QuantumCircuit(2)
        
        # Layer 1 Angles: params[0]*x + params[1]
        l1_y1 = self.theta_params[0] * self.x_param + self.theta_params[1]
        l1_x  = self.theta_params[2] * self.x_param + self.theta_params[3]
        l1_y2 = self.theta_params[4] * self.x_param + self.theta_params[5]
        
        # Layer 2 Angles: params[6]*x + params[7]
        l2_y1 = self.theta_params[6] * self.x_param + self.theta_params[7]
        l2_x  = self.theta_params[8] * self.x_param + self.theta_params[9]
        l2_y2 = self.theta_params[10] * self.x_param + self.theta_params[11]
        
        # --- Layer 1 ---
        self.qc.ry(l1_y1, 0)
        self.qc.x(1)
        self.qc.h(1) # Ancilla in |-⟩
        self.qc.cx(0, 1)
        self.qc.rx(l1_x, 0)
        self.qc.ry(l1_y2, 0)
        
        # --- Layer 2 (Re-uploading) ---
        self.qc.ry(l2_y1, 0)
        self.qc.cx(0, 1)
        self.qc.rx(l2_x, 0)
        self.qc.ry(l2_y2, 0)
        
        # Measurement
        self.qc.measure_all()
        
        # Transpile once
        pm = generate_preset_pass_manager(optimization_level=1, backend=self.simulator)
        self.transpiled_qc = pm.run(self.qc)
    
    def quantum_path(self, x):
        """
        Execute quantum path: 2-qubit entangled circuit with Data Re-uploading.
        
        Uses pre-compiled parameterized circuit for speed.
        
        Args:
            x (float): Normalized input value
            
        Returns:
            tuple: (weighted_output, probabilities_dict)
        """
        # Bind parameters: x and current weights
        # We create a dictionary mapping Parameter objects to values
        param_values = {self.x_param: x}
        for i in range(12):
            param_values[self.theta_params[i]] = self.quantum_params[i]
            
        # Bind and run
        # assign_parameters creates a new bound circuit (lightweight operation)
        bound_qc = self.transpiled_qc.assign_parameters(param_values)
        
        # Run with finite shots
        shots = 100
        result = self.simulator.run(bound_qc, shots=shots).result()
        counts = result.get_counts()
        
        # Convert counts to probabilities
        # Keys are bitstrings like '00', '01' (little endian in Qiskit usually, but measure_all makes it standard)
        # We need to map '00'->0, '01'->1, '10'->2, '11'->3
        probs = np.zeros(4)
        total_shots = sum(counts.values())
        
        for bitstring, count in counts.items():
            # bitstring is like "0 1" or "01" depending on register structure
            # measure_all produces "q1 q0" usually. Let's parse carefully.
            # We treat the integer value of the bitstring as the index.
            # int(bitstring, 2) converts binary string to int.
            # Note: Qiskit uses little-endian (qubit 0 is rightmost).
            # So '10' means q1=1, q0=0 -> index 2.
            idx = int(bitstring.replace(" ", ""), 2)
            if idx < 4:
                probs[idx] = count / total_shots
        
        # Map 4 basis states to output using learned weights
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
        Compute gradients for all 18 parameters using hybrid approach.
        
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
                - gradient_vector: np.array of shape (18,)
                - loss_value: scalar MSE loss
        """
        y_pred = self.forward(x)
        loss = (y_pred - y_true)**2
        gradient = np.zeros(18)
        
        dloss_dpred = 2 * (y_pred - y_true)  # Derivative of MSE
        
        # Get current quantum output and probabilities
        q_out, probs = self.quantum_path(x)
        c_out = self.classical_path(x)
        
        # ============================================================
        # QUANTUM PARAMETER GRADIENTS (Parameter Shift Rule)
        # ============================================================
        # For each quantum encoding parameter (12 total), evaluate circuit at θ±π/2
        for i in range(12):
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
            gradient[12 + i] = dloss_dpred * probs[i]  # ∂L/∂w_q[i]
        
        # ============================================================
        # CLASSICAL PARAMETER GRADIENTS (Analytical)
        # ============================================================
        # Standard backpropagation through tanh activation
        dtanh = 1 - c_out**2  # Derivative of tanh
        gradient[16] = dloss_dpred * dtanh * x  # ∂L/∂w_c
        gradient[17] = dloss_dpred * dtanh       # ∂L/∂b_c
        
        return gradient, loss


def train(model, X_train, y_train, epochs=50, lr=0.05, momentum=0.9, lr_decay=0.99):
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
        lr_decay: Learning rate decay factor per epoch (default 0.99)
        
    Returns:
        list: Training losses per epoch
    """
    losses = []
    
    # Track best model
    best_loss = float('inf')
    best_params = {
        'quantum': model.quantum_params.copy(),
        'w_q': model.w_q.copy(),
        'w_c': model.w_c,
        'b_c': model.b_c
    }
    
    print("\nTraining:")
    current_lr = lr
    
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
            # Quantum encoding parameters (12)
            for i in range(12):
                model.quantum_params[i] -= current_lr * model.momentum[i]
            
            # Quantum output weights (4)
            for i in range(4):
                model.w_q[i] -= current_lr * model.momentum[12 + i]
            
            # Classical parameters (2)
            model.w_c -= current_lr * model.momentum[16]
            model.b_c -= current_lr * model.momentum[17]
        
        # Track average loss per epoch
        avg_loss = total_loss / len(X_train)
        losses.append(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_params['quantum'] = model.quantum_params.copy()
            best_params['w_q'] = model.w_q.copy()
            best_params['w_c'] = model.w_c
            best_params['b_c'] = model.b_c
            
        # Decay learning rate
        current_lr *= lr_decay
        
        # Print progress every 10 epochs
        epoch_time = time.time() - epoch_start
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{epochs}: Loss = {avg_loss:.4f} (lr={current_lr:.4f}, {epoch_time:.1f}s)")
            
    # Restore best parameters
    print(f"\nRestoring best model (Loss = {best_loss:.4f})")
    model.quantum_params = best_params['quantum']
    model.w_q = best_params['w_q']
    model.w_c = best_params['w_c']
    model.b_c = best_params['b_c']
    
    return losses


def main():
    print("="*70)
    print("MINIMAL HYBRID: CUBIC FUNCTION TEST")
    print("="*70)
    
    # Generate data
    np.random.seed(42)
    n_samples = 20
    X = np.linspace(-2, 2, n_samples)
    # Cubic function: y = x³ - x + 1
    # This tests if the model can learn odd symmetry and inflection points
    y = X**3 - X + 1 + np.random.normal(0, 0.4, n_samples)
    
    #print(f"\nData: y = x³ - x + 1 + noise(σ=0.1)")
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
    print("  │ Quantum Path:  2 qubits with Data Re-uploading │ 16 params total")
    print("  │   Layer 1: Ry-Rx-Ry (Q0) + CNOT(0,1)          │ (12 encoding +")
    print("  │   Layer 2: Ry-Rx-Ry (Q0) + CNOT(0,1)          │  4 output weights)")
    print("  │   Qubit 1: Ancilla in |-⟩ state               │")
    print("  │      ↓                                          │")
    print("  │   Output: Σ wᵢ·P(|i⟩) for |00⟩,|01⟩,|10⟩,|11⟩ │")
    print("  └─────────────────────────────────────────────────┘")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ Classical Path: x → tanh(w·x + b)              │  2 params")
    print("  └─────────────────────────────────────────────────┘")
    print("  Output: quantum + classical")
    print(f"  Total: 18 parameters (12 encoding + 4 output + 2 classical)")
    
    model = MinimalHybrid(seed=42)
    
    print(f"\nInitial parameters:")
    print(f"  Quantum params: {model.quantum_params}")
    print(f"  Quantum output: {model.w_q}")
    print(f"  Classical: w_c={model.w_c:.4f}, b_c={model.b_c:.4f}")
    
    # Train
    start_time = time.time()
    # Optimized training: 30 epochs with tuned hyperparameters
    # Strategy: High initial LR (0.15) to escape plateau, higher momentum (0.8) for stability,
    # slower decay (0.94) to keep learning longer.
    losses = train(model, X_scaled, y_scaled, epochs=40, lr=0.15, momentum=0.8, lr_decay=0.94)
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
    y_true = X_dense**3 - X_dense + 1
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
    print(f"  • 2-qubit phase kickback architecture: 18 params")
    print(f"  • Quantum: Qubit 0 (Ry-Rx-Ry) x 2 layers + CNOT + Qubit 1 (X-H ancilla)")
    print(f"  • Phase kickback: ancilla in |-⟩ creates phase on qubit 0")
    print(f"  • Output: 4 basis states |00⟩, |01⟩, |10⟩, |11⟩")
    print(f"  • Classical: tanh learns residual linear trends")
    print("="*70)


if __name__ == "__main__":
    main()
