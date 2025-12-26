# Bayesian Neural Network (from Scratch)

This repository contains a "bare minimum" implementation of a Bayesian Neural Network (BNN) using only `numpy`. It demonstrates how to perform Bayesian inference on neural network weights without relying on deep learning frameworks like PyTorch or TensorFlow.

## Overview

The goal of this project is to demystify Bayesian Deep Learning by building the core algorithms from scratch. It highlights the fundamental difference between **Optimization** (Standard NN) and **Integration/Sampling** (Bayesian NN).

## Key Concepts

### 1. Standard vs. Bayesian Neural Networks
*   **Standard NN (Optimization)**: Uses Gradient Descent to slide downhill and find the single "best" set of weights (Maximum Likelihood). It asks: *"What is the absolute lowest error?"*
*   **Bayesian NN (Sampling)**: Uses MCMC to explore the probability landscape. It learns the **Posterior Distribution** $P(w|D)$ of all plausible weights given the data. It asks: *"What is the volume of good solutions?"*

**Benefits:**
*   **Uncertainty Quantification**: The model tells you when it is "guessing" (high variance in predictions).
*   **Robustness**: Averaging over thousands of models (ensemble) prevents overfitting.
*   **Safety**: Critical for high-stakes applications (medical, automotive) where knowing *what you don't know* is essential.

### 2. The Algorithm: Metropolis-Hastings MCMC
Since we cannot calculate the posterior distribution analytically (the denominator $P(D)$ is intractable), we use **Markov Chain Monte Carlo (MCMC)**.
1.  **Propose**: Make a small random jump in parameter space.
2.  **Evaluate**: Check if the new position is more probable using Bayes' Rule (Likelihood $\times$ Prior).
3.  **Accept/Reject**: If more probable, move there. If less probable, move there with probability $p$.
4.  **Repeat**: Over time, the samples map out the true distribution.

### 3. Hierarchical Bayes (Automatic Noise Inference)
In real-world problems, we don't know the true noise level of the data. This implementation uses **Hierarchical Bayes**:
*   We treat the noise standard deviation ($\sigma$) as an unknown parameter, just like the weights.
*   The state vector is $\theta = \{w_1, w_2, ..., w_{16}, \log \sigma\}$.
*   The model automatically infers the noise level that best balances fitting the data vs. satisfying the prior.

### 4. Mathematical Details

We use **Log-Probabilities** for numerical stability. The MCMC acceptance ratio relies on the ratio of probabilities, so constant normalizing factors cancel out.

#### Log Prior
We assume a Gaussian Prior on the parameters $\theta$:
$$ P(\theta) = \frac{1}{\sqrt{2\pi}} e^{-\frac{1}{2}\theta^2} $$

Taking the log:
$$ \log P(\theta) = \log\left(\frac{1}{\sqrt{2\pi}}\right) - \frac{1}{2}\theta^2 $$

In the code (`log_prior`), we drop the constant term $\log(1/\sqrt{2\pi})$ because it is the same for all $\theta$ and cancels out in the MCMC ratio.
$$ \text{Code: } \texttt{lp} = -0.5 * \sum \theta^2 $$

#### Log Likelihood
We assume the data $y$ comes from a Gaussian distribution centered at the network prediction $\hat{y}$ with noise $\sigma$:
$$ P(D|\theta) = \prod_{i=1}^{N} \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}} $$

Taking the log:
$$ \log P(D|\theta) = \sum_{i=1}^{N} \left[ -\log(\sigma) - \log(\sqrt{2\pi}) - \frac{(y_i - \hat{y}_i)^2}{2\sigma^2} \right] $$
$$ \log P(D|\theta) = -N\log(\sigma) - \frac{N}{2}\log(2\pi) - \frac{1}{2\sigma^2}\sum (y_i - \hat{y}_i)^2 $$

In the code (`log_likelihood`):
*   We **drop** $-\frac{N}{2}\log(2\pi)$ (Constant).
*   We **KEEP** $-N\log(\sigma)$ because $\sigma$ is a parameter we are sampling (it changes!).
$$ \text{Code: } \texttt{ll} = -N \log(\sigma) - 0.5 \frac{\text{SSE}}{\sigma^2} $$

## The Code (`simple_bnn_example.py`)

### Architecture
*   **Input**: 1 dimension
*   **Hidden**: 5 neurons (`tanh` activation)
*   **Output**: 1 dimension
*   **Total Parameters**: 17 (16 weights + 1 noise parameter)

### Implementation Details
*   **Log-Likelihood**: Gaussian likelihood. Includes the $-N \log(\sigma)$ term to penalize "explaining away" error with infinite noise.
*   **Log-Prior**: 
    *   Gaussian prior on weights (equivalent to L2 regularization).
    *   Gaussian prior on $\log \sigma$ (weak belief that noise is around 1.0).
*   **Sampling**: 
    *   50,000 iterations.
    *   **Burn-in**: First 10,000 samples discarded to allow convergence.
    *   **Thinning**: Every 10th sample kept to reduce correlation.

## Usage

Run the script directly with Python:

```bash
python simple_bnn_example.py
```

## Results

The script outputs the inferred statistics and generates a plot `simple_bnn_example.png`.

**Example Output:**
```text
Posterior Statistics:
---------------------
Inferred Noise Std: 1.037 +/- 0.129
(True Noise Std used for generation: 1.0)
```

**Visualization:**
*   **Blue Dots**: Noisy training data ($y = x^3 + \epsilon$).
*   **Red Line**: The mean prediction of the Bayesian ensemble.
*   **Red Shading**: The uncertainty ($\pm 2\sigma$). Notice how the model becomes uncertain (shading widens) in regions where there is no data.

![BNN Result Plot](simple_bnn_example.png)
