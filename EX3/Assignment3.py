import numpy as np

# Load the CSV files
trace1 = np.loadtxt("Trace1.csv", delimiter=",")
trace2 = np.loadtxt("Trace2.csv", delimiter=",")
trace3 = np.loadtxt("Trace3.csv", delimiter=",")
# Mean and moments for Trace 1
mean_trace1 = np.mean(trace1)
second_moment_trace1 = np.mean(trace1 ** 2)
third_moment_trace1 = np.mean(trace1 ** 3)
fourth_moment_trace1 = np.mean(trace1 ** 4)

print(f"Trace 1 Mean: {mean_trace1}")
print(f"Trace 1 Second Moment: {second_moment_trace1}")
print(f"Trace 1 Third Moment: {third_moment_trace1}")
print(f"Trace 1 Fourth Moment: {fourth_moment_trace1}")

# Variance and centered moments for Trace 1
variance_trace1 = np.var(trace1)
third_centered_moment_trace1 = np.mean((trace1 - mean_trace1) ** 3)
fourth_centered_moment_trace1 = np.mean((trace1 - mean_trace1) ** 4)

print(f"Trace 1 Variance: {variance_trace1}")
print(f"Trace 1 Third Centered Moment: {third_centered_moment_trace1}")
print(f"Trace 1 Fourth Centered Moment: {fourth_centered_moment_trace1}")



from scipy.stats import skew, kurtosis

# Skewness and Kurtosis for Trace 1
skewness_trace1 = skew(trace1)
fourth_standardized_moment_trace1 = kurtosis(trace1, fisher=False)

print(f"Trace 1 Skewness: {skewness_trace1}")
print(f"Trace 1 Fourth Standardized Moment: {fourth_standardized_moment_trace1}")


# Standard Deviation, Coefficient of Variation, and Excess Kurtosis for Trace 1
std_dev_trace1 = np.std(trace1)
coef_variation_trace1 = std_dev_trace1 / mean_trace1
excess_kurtosis_trace1 = kurtosis(trace1)  # Excess kurtosis

print(f"Trace 1 Standard Deviation: {std_dev_trace1}")
print(f"Trace 1 Coefficient of Variation: {coef_variation_trace1}")
print(f"Trace 1 Excess Kurtosis: {excess_kurtosis_trace1}")


# Percentiles, median, and quartiles for Trace 1
median_trace1 = np.median(trace1)
first_quartile_trace1 = np.percentile(trace1, 25)
third_quartile_trace1 = np.percentile(trace1, 75)
p5_trace1 = np.percentile(trace1, 5)
p90_trace1 = np.percentile(trace1, 90)

print(f"Trace 1 Median: {median_trace1}")
print(f"Trace 1 First Quartile: {first_quartile_trace1}")
print(f"Trace 1 Third Quartile: {third_quartile_trace1}")
print(f"Trace 1 5th Percentile: {p5_trace1}")
print(f"Trace 1 90th Percentile: {p90_trace1}")




import matplotlib.pyplot as plt

# Pearson correlation for lags m=1 to m=100 for Trace 1
def plot_pearson_corr(data, file_label):
    autocorrs = [np.corrcoef(data[:-m], data[m:])[0, 1] for m in range(1, 101)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), autocorrs, label=f'Pearson Correlation for {file_label}')
    plt.title(f'Pearson Correlation Coefficient for Lags (1-100) - {file_label}')
    plt.xlabel('Lag (m)')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_pearson_corr(trace1, "Trace1")

# CDF plot for Trace 1
def plot_cdf(data, file_label):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, yvals, label=f'CDF for {file_label}')
    plt.title(f'CDF - {file_label}')
    plt.xlabel('Inter-Arrival Times')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.show()

plot_cdf(trace1, "Trace1")
