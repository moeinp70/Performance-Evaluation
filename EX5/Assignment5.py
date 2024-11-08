import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, gamma, pareto

# Exponential distribution CDF
def generate_exponential_cdf(lambda_, N=10000):
    samples = np.random.exponential(1/lambda_, N)
    x = np.sort(samples)
    empirical_cdf = np.arange(1, N + 1) / N
    theoretical_cdf = 1 - np.exp(-lambda_ * x)

    plt.figure()
    plt.plot(x, empirical_cdf, label='Empirical CDF', linestyle='none', marker='o', markersize=2)
    plt.plot(x, theoretical_cdf, label='Theoretical CDF', color='orange')
    plt.title('Exponential Distribution CDF ')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

#  Pareto distribution CDF
def generate_pareto_cdf(a, m, N=10000):
    samples = (np.random.pareto(a, N) + 1) * m
    x = np.sort(samples)
    empirical_cdf = np.arange(1, N + 1) / N
    theoretical_cdf = 1 - (m / x) ** a

    plt.figure()
    plt.plot(x, empirical_cdf, label='Empirical CDF', linestyle='none', marker='o', markersize=2)
    plt.plot(x, theoretical_cdf, label='Theoretical CDF', color='orange')
    plt.title('Pareto Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

#  Erlang distribution CDF
def generate_erlang_cdf(k, lambda_, N=10000):
    samples = np.random.gamma(k, 1/lambda_, N)
    x = np.sort(samples)
    empirical_cdf = np.arange(1, N + 1) / N
    theoretical_cdf = gamma.cdf(x, a=k, scale=1/lambda_)

    plt.figure()
    plt.plot(x, empirical_cdf, label='Empirical CDF', linestyle='none', marker='o', markersize=2)
    plt.plot(x, theoretical_cdf, label='Theoretical CDF', color='orange')
    plt.title('Erlang Distribution CDF')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

# Hypo-Exponential distribution CDF
# Function for the closed-form CDF of Hypo-Exponential distribution
def hypo_exponential_cdf(x, lambda1, lambda2):
    if lambda1 == lambda2:
        # Special case where both rates are equal
        return 1 - np.exp(-lambda1 * x) * (1 + lambda1 * x)
    else:
        return 1 - ((lambda2 * np.exp(-lambda1 * x) - lambda1 * np.exp(-lambda2 * x)) / (lambda2 - lambda1))

# Function to generate and plot Hypo-Exponential distribution CDF
def generate_hypo_exponential_cdf(lambda1, lambda2, N=10000):
    samples = np.random.exponential(1/lambda1, N) + np.random.exponential(1/lambda2, N)
    x = np.sort(samples)
    empirical_cdf = np.arange(1, N + 1) / N
    theoretical_cdf = hypo_exponential_cdf(x, lambda1, lambda2)

    # Plotting the CDF
    plt.figure()
    plt.plot(x, empirical_cdf, label='Empirical CDF', linestyle='none', marker='o', markersize=2)
    plt.plot(x, theoretical_cdf, label='Theoretical CDF', color='orange')
    plt.title(f'Hypo-Exponential Distribution CDF (lambda1 = {lambda1}, lambda2 = {lambda2})')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()


# Hyper-Exponential distribution CDF
def generate_hyper_exponential_cdf(lambda1, lambda2, p1, N=10000):
    branch = np.random.choice([0, 1], size=N, p=[p1, 1 - p1])
    samples = np.where(branch == 0, np.random.exponential(1/lambda1, N), np.random.exponential(1/lambda2, N))
    x = np.sort(samples)
    empirical_cdf = np.arange(1, N + 1) / N
    theoretical_cdf = p1 * (1 - np.exp(-lambda1 * x)) + (1 - p1) * (1 - np.exp(-lambda2 * x))

    plt.figure()
    plt.plot(x, empirical_cdf, label='Empirical CDF', linestyle='none', marker='o', markersize=2)
    plt.plot(x, theoretical_cdf, label='Theoretical CDF', color='orange')
    plt.title('Hyper-Exponential Distribution ')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.show()

generate_exponential_cdf(lambda_=0.25)
generate_pareto_cdf(a=2.5, m=3)
generate_erlang_cdf(k=8, lambda_=0.8)
generate_hypo_exponential_cdf(lambda1=0.25, lambda2=0.4)
generate_hyper_exponential_cdf(lambda1=1, lambda2=0.05, p1=0.75)
