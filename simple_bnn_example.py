#!/projects/hpcapps/nsawant/qcBac/lbmQC/envs/q_gpu/bin/python

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Bare Minimum Bayesian Neural Network (BNN) Example
# Using Numpy and Metropolis-Hastings Sampling
# -----------------------------------------------------------------------------
#
# WHY BAYESIAN NEURAL NETWORKS?
# -----------------------------
# Standard Neural Networks (NN) give you a single answer: "I predict y=5.2".
# But they don't tell you how CONFIDENT they are. They might be "guessing"
# because they haven't seen data like this before.
#
# Bayesian Neural Networks (BNN) give you a DISTRIBUTION of answers.
# Instead of learning one "best" set of weights, we learn a probability
# distribution over all possible weights.
#
# BENEFITS:
# 1. Uncertainty Quantification: The model tells you when it "doesn't know".
#    (e.g., "I predict y=5.2, but the standard deviation is huge (±10.0)")
# 2. Robustness: Less likely to overfit because we average over many models.
# 3. Safety: Critical for self-driving cars, medical diagnosis, etc.
#
# HOW IT WORKS (SIMPLIFIED):
# 1. We define a "Prior" belief about weights (e.g., they should be small).
# 2. We see Data and calculate "Likelihood" (how well weights fit data).
# 3. We combine them to get the "Posterior" (updated belief about weights).
# 4. Since we can't calculate the Posterior exactly, we use MCMC sampling
#    to "explore" the landscape of good weights.
# -----------------------------------------------------------------------------

def simple_bnn_example():
    print("Bare Minimum BNN Example (Numpy + Metropolis-Hastings)")
    print("======================================================")

    # 1. Generate Non-Linear Data (y = x^3 + noise)
    np.random.seed(42)
    # IMPROVEMENT 1: More Data
    # Increasing samples from 20 -> 50 gives the model more evidence.
    # This sharpens the posterior (reduces uncertainty) and improves accuracy.
    N_SAMPLES = 50
    
    # TRUE_NOISE_STD is used ONLY to generate the synthetic data.
    # The model does NOT know this value. It must infer it.
    TRUE_NOISE_STD = 1.0 
    
    X = np.random.uniform(-2, 2, size=(N_SAMPLES, 1))
    y = X**3 + np.random.normal(0, TRUE_NOISE_STD, size=(N_SAMPLES, 1))
    
    # 2. Define Network Architecture
    # Structure: 1 input -> 5 hidden (tanh) -> 1 output
    #
    # QUESTION: Why 5 neurons (17 params) for a cubic function (4 params)?
    # A polynomial model (ax^3 + bx^2 + cx + d) fits this perfectly with 4 params.
    # However, a Neural Network does not know about polynomials!
    # It must build the curve x^3 using "tanh" functions (S-shaped curves).
    #
    # This is the "Universal Approximation" cost:
    # To approximate a simple polynomial using generic S-curves, we often need
    # MORE parameters than the polynomial itself.
    # 3 neurons was too stiff; 5 neurons gives enough flexibility to bend.
    n_input, n_hidden, n_output = 1, 5, 1
    n_params = (n_input * n_hidden) + n_hidden + (n_hidden * n_output) + n_output + 1
    
    def unpack_params(params_flat):
        # Last parameter is log_noise_std (we sample in log space to keep it positive)
        log_noise_std = params_flat[-1]
        noise_std = np.exp(log_noise_std)
        
        # The rest are weights
        w_flat = params_flat[:-1]
        
        # Unpack flat weight vector into matrices
        idx = 0
        w1 = w_flat[idx : idx + n_input*n_hidden].reshape(n_input, n_hidden)
        idx += n_input*n_hidden
        b1 = w_flat[idx : idx + n_hidden]
        idx += n_hidden
        w2 = w_flat[idx : idx + n_hidden*n_output].reshape(n_hidden, n_output)
        idx += n_hidden*n_output
        b2 = w_flat[idx : idx + n_output]
        return w1, b1, w2, b2, noise_std

    def forward(x, params_flat):
        # This is a standard neural network forward pass.
        # The only difference is that 'params_flat' is not a fixed, optimized vector.
        # It is a RANDOM SAMPLE from our posterior distribution.
        w1, b1, w2, b2, _ = unpack_params(params_flat) # We don't need noise_std for prediction mean
        # Layer 1
        z1 = x @ w1 + b1
        a1 = np.tanh(z1)
        # Layer 2
        out = a1 @ w2 + b2
        return out

    # 3. Define Probabilistic Model
    # -------------------------------------------------------------------------
    # In a BNN, we treat weights AND noise parameters as random variables.
    # Let theta = {weights, log_sigma} be the set of all parameters we sample.
    # (We sample log_sigma to ensure sigma is always positive).
    #
    # Posterior P(theta|D) = Likelihood P(D|theta) * Prior P(theta) / P(D)
    #
    # We work with logs:
    # log P(theta|D) = log P(D|theta) + log P(theta) + const
    # -------------------------------------------------------------------------
    
    # Log Prior: 
    # 1. Gaussian N(0, 1) on all weights (L2 regularization)
    # 2. Gaussian N(0, 1) on log_noise_std (Weak belief that noise is around 1.0)
    #
    # MATH:
    # Prior P(theta) ~ N(0, 1) = (1/sqrt(2pi)) * exp(-0.5 * theta^2)
    # Log Prior      = log(1/sqrt(2pi)) - 0.5 * theta^2
    #                = Constant         - 0.5 * theta^2
    #
    # In MCMC, we ignore the constant because it cancels out in the ratio P(new)/P(old).
    def log_prior(params_flat):
        w_flat = params_flat[:-1]
        log_noise_std = params_flat[-1]
        
        lp_weights = -0.5 * np.sum(w_flat**2)
        lp_noise = -0.5 * (log_noise_std**2)
        return lp_weights + lp_noise

    # Log Likelihood: Gaussian N(y_pred, noise_std)
    # This measures how well the network fits the data.
    # It assumes the data y is generated by the network plus Gaussian noise.
    #
    # MATH:
    # Likelihood P(D|theta) = Product_i [ (1 / (sigma * sqrt(2pi))) * exp( -0.5 * (y_i - y_pred_i)^2 / sigma^2 ) ]
    #
    # Log Likelihood = Sum_i [ -log(sigma) - 0.5*log(2pi) - 0.5 * (y_i - y_pred_i)^2 / sigma^2 ]
    #                = -N * log(sigma) - (N/2)*log(2pi) - 0.5 * SSE / sigma^2
    #
    # We drop -(N/2)*log(2pi) because it's constant.
    # We KEEP -N*log(sigma) because sigma is a parameter we are sampling!
    def log_likelihood(params_flat, x_data, y_data):
        # We unpack the noise_std from the parameters!
        # The model can now "tune" the noise level to balance data vs prior.
        w1, b1, w2, b2, noise_std = unpack_params(params_flat)
        
        # Re-implement forward locally or call forward (which ignores noise)
        # Let's just call forward since it returns y_pred
        y_pred = forward(x_data, params_flat)
        
        N = len(y_data)
        
        term1 = -0.5 * np.sum((y_data - y_pred)**2) / (noise_std**2)
        term2 = -N * np.log(noise_std)
        
        return term1 + term2

    # Log Posterior = Log Likelihood + Log Prior
    # This is the function we want to sample from.
    def log_posterior(params_flat, x_data, y_data):
        return log_likelihood(params_flat, x_data, y_data) + log_prior(params_flat)

    # 4. Metropolis-Hastings Sampler
    # -------------------------------------------------------------------------
    # GOAL: To calculate the Posterior Distribution P(w|D).
    #
    # The posterior tells us which neural network weights are likely given the data.
    # Unlike standard training which finds the single "best" weight vector (Maximum Likelihood),
    # we want to find a *distribution* of plausible weight vectors.
    #
    # STANDARD NN vs. BAYESIAN NN:
    # 1. The Goal (Optimization vs. Integration):
    #    - Standard NN: Wants to find the single BEST point (Maximum Likelihood).
    #      It asks: "What is the absolute lowest error?"
    #    - Bayesian NN: Wants to find the whole SHAPE (Posterior Distribution).
    #      It asks: "What is the volume of good solutions?"
    #
    # 2. The Algorithm (Gradient Descent vs. Bayes Rule):
    #    - Standard NN uses GRADIENT DESCENT. It calculates the slope (derivative)
    #      of the loss and slides downhill. It ignores probability, it just wants
    #      to go down.
    #    - Bayesian NN (this example) uses BAYES THEOREM + SAMPLING.
    #      We don't calculate any derivatives! We just propose a random jump
    #      and check Bayes Rule: "Is P(New) / P(Old) good enough?"
    #      If yes, we move. If no, we stay.
    #
    #      (Note: Advanced BNNs like HMC *do* use gradients to sample faster,
    #       but the goal remains exploration, not just minimization.)
    #
    # PROBLEM: The posterior P(w|D) is proportional to P(D|w) * P(w).
    # We can calculate the numerator (Likelihood * Prior), but the denominator P(D)
    # (the evidence) is an intractable integral over all possible weights.
    #
    # SOLUTION: Markov Chain Monte Carlo (MCMC).
    # We construct a chain of samples that converges to the posterior distribution
    # without needing to know the normalization constant P(D).
    # -------------------------------------------------------------------------
    print(f"Sampling {n_params} parameters using Metropolis-Hastings...")
    
    # IMPROVEMENT 3: Longer Sampling
    # Since we have more parameters (17 vs 11), we need more steps to explore.
    #
    # QUESTION: Why not just increase sampling?
    # Sampling only helps us "see" the posterior distribution clearly.
    # It does NOT change the distribution itself!
    # - If the model is too simple (3 hidden neurons), the posterior will
    #   be centered around "bad" solutions that don't fit x^3 well.
    # - If the data is too sparse (20 samples), the posterior will be very
    #   wide/uncertain, no matter how much we sample.
    #
    # To improve ACCURACY, we needed:
    # 1. Better Model (5 neurons) -> Allows the posterior to contain "good" solutions.
    # 2. More Data (50 samples) -> Narrows the posterior around the truth.
    # 3. More Sampling -> Ensures we actually find those good solutions.
    n_iterations = 50000
    
    # proposal_std controls the "step size" of our exploration.
    # - If too small: We explore very slowly (high correlation).
    # - If too large: We jump to bad regions and get rejected often.
    # 0.1 is a heuristic choice for this specific problem scale.
    proposal_std = 0.1
    
    # Initialize parameters (theta)
    # We start with small random values to avoid saturation.
    # Using proposal_std as the scale is a reasonable heuristic here.
    current_params = np.random.randn(n_params) * proposal_std
    current_log_prob = log_posterior(current_params, X, y)
    
    samples = []
    accepted = 0
    
    for i in range(n_iterations):
        # Propose new parameters by adding small random noise to current parameters
        proposed_params = current_params + np.random.normal(0, proposal_std, size=n_params)
        proposed_log_prob = log_posterior(proposed_params, X, y)
        
        # Acceptance probability (Metropolis Step)
        # We accept the new state if it's more probable (log_alpha > 0)
        # OR with probability exp(log_alpha) if it's less probable.
        # log_alpha = log( P(new)/P(old) ) = log(P(new)) - log(P(old))
        log_alpha = proposed_log_prob - current_log_prob
        
        # Accept/Reject
        if np.log(np.random.rand()) < log_alpha:
            current_params = proposed_params
            current_log_prob = proposed_log_prob
            accepted += 1
            
        # Save sample (burn-in and thinning)
        # Burn-in: Discard first 10000 steps (increased from 5000)
        # Thinning: Keep every 10th step to reduce correlation between samples.
        if i > 10000 and i % 10 == 0:
            # CRITICAL CONCEPT:
            # We are saving 'current_params' into our list of samples.
            # Each sample represents a FULLY FUNCTIONAL NEURAL NETWORK + NOISE ESTIMATE.
            # By collecting 1000 samples, we effectively have 1000 different
            # neural networks, each slightly different but all fitting the data well.
            samples.append(current_params)
            
    print(f"Acceptance rate: {accepted / n_iterations:.2f}")
    print(f"Collected {len(samples)} samples.")

    # -------------------------------------------------------------------------
    # Analyze Posterior Distribution of Weights
    # Since we have samples from P(w|D), we can calculate statistics like
    # the mean and standard deviation for each weight.
    # -------------------------------------------------------------------------
    samples_array = np.array(samples)
    
    # Separate weights and noise parameters
    weights_samples = samples_array[:, :-1]
    log_noise_samples = samples_array[:, -1]
    noise_samples = np.exp(log_noise_samples)
    
    weight_means = np.mean(weights_samples, axis=0)
    weight_stds = np.std(weights_samples, axis=0)
    
    noise_mean = np.mean(noise_samples)
    noise_std_dev = np.std(noise_samples)
    
    print("\nPosterior Statistics:")
    print("---------------------")
    print(f"Weight Means: {np.round(weight_means, 3)}")
    print(f"Weight Stds:  {np.round(weight_stds, 3)}")
    print(f"Inferred Noise Std: {noise_mean:.3f} +/- {noise_std_dev:.3f}")
    print(f"(True Noise Std used for generation: {TRUE_NOISE_STD})")
    print("(Higher std means we are less confident about that specific parameter)")

    # 5. Prediction & Visualization
    # -------------------------------------------------------------------------
    # ENSEMBLE PREDICTION:
    # Instead of trusting one single network, we ask thousands of different
    # networks (our samples) to vote.
    #
    # HOW CONFIDENCE IS CALCULATED:
    # We don't just plug the "mean weights" into the network. That would be wrong.
    # Instead, we run the input X_test through EVERY sampled parameter vector.
    #
    # 1. For each sample theta_i in our chain:
    #    y_pred_i = Network(X_test, theta_i)
    #
    # 2. We now have thousands of prediction curves.
    #    - In regions with data, all curves will agree (low variance).
    #    - In regions without data, curves will fan out (high variance).
    #
    # 3. The "Confidence" is simply the standard deviation of these predictions.
    #    std_pred = std(y_pred_1, y_pred_2, ..., y_pred_N)
    # -------------------------------------------------------------------------
    X_test = np.linspace(-2.5, 2.5, 100).reshape(-1, 1)
    
    # Compute predictions for all posterior samples
    predictions = []
    for params_samp in samples:
        pred = forward(X_test, params_samp)
        predictions.append(pred.flatten())
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    std_pred = np.std(predictions, axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data')
    plt.plot(X_test, mean_pred, color='red', label='BNN Mean')
    plt.fill_between(X_test.flatten(), 
                     mean_pred - 2*std_pred, 
                     mean_pred + 2*std_pred, 
                     color='red', alpha=0.2, label='Uncertainty (2 std)')
    plt.plot(X_test, X_test**3, 'g--', label='True Function (x^3)')
    
    plt.title('Bare Minimum Bayesian Neural Network (Numpy)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = 'simple_bnn_example.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    simple_bnn_example()
