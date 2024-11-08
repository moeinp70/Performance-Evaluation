import numpy as np

trace1 = np.loadtxt("Trace1.csv", delimiter=",")
trace2 = np.loadtxt("Trace2.csv", delimiter=",")

def calculate_moments(data):
    mean = np.mean(data)
    second_moment = np.mean(data ** 2)  # Second moment
    third_moment = np.mean(data ** 3)   # Third moment
    return mean, second_moment, third_moment

trace1_moments = calculate_moments(trace1)
trace2_moments = calculate_moments(trace2)

print(f"Trace 1 Moments: Mean = {trace1_moments[0]}, Second Moment = {trace1_moments[1]}, Third Moment = {trace1_moments[2]}")
print(f"Trace 2 Moments: Mean = {trace2_moments[0]}, Second Moment = {trace2_moments[1]}, Third Moment = {trace2_moments[2]}")


def fit_uniform(data):
    mean = np.mean(data)  # Mean (First Moment)
    second_moment = np.mean(data ** 2)  # Second Moment

    # Calculate a and b based on the moments
    a = mean - np.sqrt(3 * (second_moment - mean ** 2))
    b = mean + np.sqrt(3 * (second_moment - mean ** 2))

    return a, b


trace1_uniform_params = fit_uniform(trace1)
trace2_uniform_params = fit_uniform(trace2)

print(f"Trace 1 Uniform Params: a = {trace1_uniform_params[0]}, b = {trace1_uniform_params[1]}")
print(f"Trace 2 Uniform Params: a = {trace2_uniform_params[0]}, b = {trace2_uniform_params[1]}")


def fit_exponential(data):
    rate = 1 / np.mean(data)  # Rate parameter is the inverse of the mean
    return rate

trace1_exponential_param = fit_exponential(trace1)
trace2_exponential_param = fit_exponential(trace2)

print(f"Trace 1 Exponential Param (rate): {trace1_exponential_param}")
print(f"Trace 2 Exponential Param (rate): {trace2_exponential_param}")


# Erlang distribution
def fit_erlang(data):
    mean = np.mean(data)
    variance = np.var(data)

    k = mean ** 2 / variance  # Shape parameter (k)
    rate = mean / variance  # Rate parameter (lambda)

    return int(np.round(k)), rate  # k should be an integer


trace1_erlang_params = fit_erlang(trace1)
trace2_erlang_params = fit_erlang(trace2)

print(f"Trace 1 Erlang Params: Shape (k) = {trace1_erlang_params[0]}, Rate (λ) = {trace1_erlang_params[1]}")
print(f"Trace 2 Erlang Params: Shape (k) = {trace2_erlang_params[0]}, Rate (λ) = {trace2_erlang_params[1]}")








from scipy.optimize import fsolve
from scipy.special import gamma

# Function to compute Weibull
def weibull_moments(params, moments):
    k, lambda_ = params
    mean, variance = moments
    # Weibull moment equations
    eq1 = lambda_ * gamma(1 + 1/k) - mean
    eq2 = lambda_**2 * (gamma(1 + 2/k) - (gamma(1 + 1/k))**2) - variance
    return [eq1, eq2]

mean_trace1 = np.mean(trace1)
variance_trace1 = np.var(trace1)
shape_weibull_trace1, scale_weibull_trace1 = fsolve(weibull_moments, [1.0, 1.0], args=([mean_trace1, variance_trace1]))

mean_trace2 = np.mean(trace2)
variance_trace2 = np.var(trace2)
shape_weibull_trace2, scale_weibull_trace2 = fsolve(weibull_moments, [1.0, 1.0], args=([mean_trace2, variance_trace2]))

print(f"Trace 1 Weibull Params: Shape = {shape_weibull_trace1}, Scale = {scale_weibull_trace1}")
print(f"Trace 2 Weibull Params: Shape = {shape_weibull_trace2}, Scale = {scale_weibull_trace2}")



# Function to compute Pareto
def pareto_moments(params, moments):
    alpha, m = params
    mean, variance = moments
    # Pareto moment equations
    eq1 = (alpha * m) / (alpha - 1) - mean
    eq2 = (alpha * m**2) / ((alpha - 1)**2 * (alpha - 2)) - variance
    return [eq1, eq2]

alpha_pareto_trace1, m_pareto_trace1 = fsolve(pareto_moments, [3.0, mean_trace1], args=([mean_trace1, variance_trace1]))

alpha_pareto_trace2, m_pareto_trace2 = fsolve(pareto_moments, [3.0, mean_trace2], args=([mean_trace2, variance_trace2]))

print(f"Trace 1 Pareto Params: Alpha = {alpha_pareto_trace1}, Scale (m) = {m_pareto_trace1}")
print(f"Trace 2 Pareto Params: Alpha = {alpha_pareto_trace2}, Scale (m) = {m_pareto_trace2}")






from scipy.optimize import minimize


# Negative Log-Likelihood for Hyper-Exponential
def hyperexp_neg_log_likelihood(params, data):
    p, lambda1, lambda2 = params
    likelihoods = p * lambda1 * np.exp(-lambda1 * data) + (1 - p) * lambda2 * np.exp(-lambda2 * data)
    return -np.sum(np.log(likelihoods))


# Negative Log-Likelihood for Hypo-Exponential
def hypoexp_neg_log_likelihood(params, data):
    lambda1, lambda2 = params
    if lambda1 <= 0 or lambda2 <= 0 or lambda1 == lambda2:
        return np.inf  # Ensure valid parameters
    likelihoods = (lambda1 * lambda2) / (lambda1 - lambda2) * (np.exp(-lambda2 * data) - np.exp(-lambda1 * data))
    return -np.sum(np.log(likelihoods))


# Initial guesses for Hyper-Exponential MLE
initial_params_hyper = [0.5, 1 / np.mean(trace1), 1 / np.mean(trace1) * 2]

# Perform MLE for Hyper-Exponential (Trace 1)
result_hyper_trace1 = minimize(hyperexp_neg_log_likelihood, initial_params_hyper, args=(trace1,), bounds=[(0, 1), (0.0001, None), (0.0001, None)])
p1_hyperexp_trace1, lambda1_hyperexp_trace1, lambda2_hyperexp_trace1 = result_hyper_trace1.x

# Perform MLE for Hyper-Exponential (Trace 2)
result_hyper_trace2 = minimize(hyperexp_neg_log_likelihood, initial_params_hyper, args=(trace2,), bounds=[(0, 1), (0.0001, None), (0.0001, None)])
p1_hyperexp_trace2, lambda1_hyperexp_trace2, lambda2_hyperexp_trace2 = result_hyper_trace2.x

# Initial guesses for Hypo-Exponential MLE
initial_params_hypo = [1 / np.mean(trace1), 1 / (2 * np.mean(trace1))]

# Perform MLE for Hypo-Exponential (Trace 1)
result_hypo_trace1 = minimize(hypoexp_neg_log_likelihood, initial_params_hypo, args=(trace1,), bounds=[(0.0001, None), (0.0001, None)])
lambda1_hypoexp_trace1, lambda2_hypoexp_trace1 = result_hypo_trace1.x

# Perform MLE for Hypo-Exponential (Trace 2)
result_hypo_trace2 = minimize(hypoexp_neg_log_likelihood, initial_params_hypo, args=(trace2,), bounds=[(0.0001, None), (0.0001, None)])
lambda1_hypoexp_trace2, lambda2_hypoexp_trace2 = result_hypo_trace2.x

# Print results for Trace 1
print("Trace 1 Results:")
print(f" Hyper Exponential First Rate (lambda_1): {lambda1_hyperexp_trace1}")
print(f" Hyper Exponential Second Rate (lambda_2): {lambda2_hyperexp_trace1}")
print(f" Hyper Exponential Probability of First Branch (p_1): {p1_hyperexp_trace1}")
print(f" Hypo Exponential First Rate (lambda_1): {lambda1_hypoexp_trace1}")
print(f" Hypo Exponential Second Rate (lambda_2): {lambda2_hypoexp_trace1}")

# Print results for Trace 2
print("Trace 2 Results:")
print(f" Hyper Exponential First Rate (lambda_1): {lambda1_hyperexp_trace2}")
print(f" Hyper Exponential Second Rate (lambda_2): {lambda2_hyperexp_trace2}")
print(f"Hyper Exponential Probability of First Branch (p_1): {p1_hyperexp_trace2}")
print(f"Hypo Exponential First Rate (lambda_1): {lambda1_hypoexp_trace2}")
print(f" Hypo Exponential Second Rate (lambda_2): {lambda2_hypoexp_trace2}")

import matplotlib.pyplot as plt
from scipy.stats import uniform, expon, gamma, weibull_min, pareto


# Function for the empirical CDF
def empirical_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(data) + 1) / len(data)
    return sorted_data, cdf


# Plotting function for Trace 1 and Trace 2
def plot_cdfs(trace_data, trace_label, trace1_uniform_params, trace1_exponential_param, trace1_erlang_params,
              shape_weibull, scale_weibull, alpha_pareto, scale_pareto, lambda1_hyperexp, lambda2_hyperexp, p1_hyperexp,
              lambda1_hypoexp, lambda2_hypoexp):
    sorted_data, empirical_cdf_values = empirical_cdf(trace_data)

    # Generate x values for plotting
    x_values = np.linspace(0, np.max(sorted_data), 1000)

    # 1. Uniform CDF
    uniform_cdf_values = uniform.cdf(x_values, loc=trace1_uniform_params[0],
                                     scale=trace1_uniform_params[1] - trace1_uniform_params[0])

    # 2. Exponential CDF
    exp_cdf_values = expon.cdf(x_values, scale=1 / trace1_exponential_param)

    # 3. Erlang (Gamma) CDF
    erlang_cdf_values = gamma.cdf(x_values, a=trace1_erlang_params[0], scale=1 / trace1_erlang_params[1])

    # 4. Weibull CDF
    weibull_cdf_values = weibull_min.cdf(x_values, shape_weibull, scale=scale_weibull)

    # 5. Pareto CDF
    pareto_cdf_values = pareto.cdf(x_values, b=alpha_pareto, scale=scale_pareto)

    # 6. Hyper-Exponential CDF
    def hyperexp_cdf(x, p, lambda1, lambda2):
        return p * (1 - np.exp(-lambda1 * x)) + (1 - p) * (1 - np.exp(-lambda2 * x))

    hyperexp_cdf_values = hyperexp_cdf(x_values, p1_hyperexp, lambda1_hyperexp, lambda2_hyperexp)

    # 7. Hypo-Exponential CDF
    def hypoexp_cdf(x, lambda1, lambda2):
        return 1 - ((lambda2 * np.exp(-lambda1 * x) - lambda1 * np.exp(-lambda2 * x)) / (lambda2 - lambda1))

    hypoexp_cdf_values = hypoexp_cdf(x_values, lambda1_hypoexp, lambda2_hypoexp)

    # Plot the empirical CDF and all the fitted CDFs
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, empirical_cdf_values, label='Empirical CDF', marker='.', linestyle='none')
    plt.plot(x_values, uniform_cdf_values, label='Uniform CDF')
    plt.plot(x_values, exp_cdf_values, label='Exponential CDF')
    plt.plot(x_values, erlang_cdf_values, label='Erlang CDF')
    plt.plot(x_values, weibull_cdf_values, label='Weibull CDF')
    plt.plot(x_values, pareto_cdf_values, label='Pareto CDF')
    plt.plot(x_values, hyperexp_cdf_values, label='Hyper-Exponential CDF')
    plt.plot(x_values, hypoexp_cdf_values, label='Hypo-Exponential CDF')

    plt.title(f'CDF Comparison for {trace_label}')
    plt.xlabel('x')
    plt.ylabel('CDF')
    plt.legend()
    plt.grid(True)
    plt.show()


# Plot for Trace 1
plot_cdfs(trace1, "Trace 1", trace1_uniform_params, trace1_exponential_param, trace1_erlang_params,
          shape_weibull_trace1, scale_weibull_trace1, alpha_pareto_trace1, m_pareto_trace1,
          lambda1_hyperexp_trace1, lambda2_hyperexp_trace1, p1_hyperexp_trace1, lambda1_hypoexp_trace1,
          lambda2_hypoexp_trace1)

# Plot for Trace 2
plot_cdfs(trace2, "Trace 2", trace2_uniform_params, trace2_exponential_param, trace2_erlang_params,
          shape_weibull_trace2, scale_weibull_trace2, alpha_pareto_trace2, m_pareto_trace2,
          lambda1_hyperexp_trace2, lambda2_hyperexp_trace2, p1_hyperexp_trace2, lambda1_hypoexp_trace2,
          lambda2_hypoexp_trace2)
