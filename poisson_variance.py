import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Simulation parameters
n_simulations = 10000

# === Poisson comparison ===

# Case 1: Single Poisson draw with lambda = 20
single_draws = np.random.poisson(lam=20, size=n_simulations)

# Case 2: Sum of 200 Poisson draws with lambda = 0.1
split_draws = np.random.poisson(lam=0.1, size=(n_simulations, 200)).sum(axis=1)

# Normal distribution parameters
mean = 20
std = np.sqrt(20)
x = np.linspace(5, 35, 300)
normal_pdf = norm.pdf(x, loc=mean, scale=std) * n_simulations * (x[1] - x[0])  # scale to histogram

# Plot Poisson comparison
plt.figure(figsize=(12, 6))
plt.hist(single_draws, bins=range(5, 36), alpha=0.6, label='Poisson(λ=20)', color='blue', edgecolor='black')
plt.hist(split_draws, bins=range(5, 36), alpha=0.6, label='Sum of 200×Poisson(λ=0.1)', color='orange', edgecolor='black')
plt.plot(x, normal_pdf, label='Normal Approximation', color='green', linewidth=2)

plt.title('Comparison of Poisson Variants and Normal Approximation')
plt.xlabel('Total Daily Admissions')
plt.ylabel('Frequency (out of 10,000 simulations)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Triangular distribution comparison ===

# Case 1: Single draw from triangular (100, 300, 1000)
tri_single = np.random.triangular(left=100, mode=300, right=1000, size=n_simulations)

# Case 2: Sum of 100 draws from triangular (1, 3, 10)
tri_split = np.random.triangular(left=1, mode=3, right=10, size=(n_simulations, 100)).sum(axis=1)

# Plot Triangular comparison
plt.figure(figsize=(12, 6))
plt.hist(tri_single, bins=50, alpha=0.6, label='Triangular(100, 300, 1000)', color='red', edgecolor='black')
plt.hist(tri_split, bins=50, alpha=0.6, label='Sum of 100×Triangular(1, 3, 10)', color='blue', edgecolor='black')

plt.title('Comparison of Triangular Distributions')
plt.xlabel('Total Sum')
plt.ylabel('Frequency (out of 10,000 simulations)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()