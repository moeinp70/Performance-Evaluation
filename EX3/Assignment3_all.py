import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

files = ["Trace1.csv", "Trace2.csv", "Trace3.csv"]
traces = [np.loadtxt(f, delimiter=",") for f in files]

# Function to compute and display all statistics for each trace
def compute_and_display_statistics(data, trace_label):
    # Step 1: Mean and the second, third, and fourth moments
    mean = np.mean(data)
    second_moment = np.mean(data ** 2)
    third_moment = np.mean(data ** 3)
    fourth_moment = np.mean(data ** 4)
    
    # Step 2: Variance, third and fourth centered moments
    variance = np.var(data)
    third_centered_moment = np.mean((data - mean) ** 3)
    fourth_centered_moment = np.mean((data - mean) ** 4)
    
    # Step 3: Skewness and the fourth standardized moment
    skewness = skew(data)
    fourth_standardized_moment = kurtosis(data, fisher=False)
    
    # Step 4: Standard deviation, coefficient of variation, and excess kurtosis
    std_dev = np.std(data)
    coef_variation = std_dev / mean
    excess_kurtosis = kurtosis(data)
    
    # Step 5: Median, first and third quartiles, 5th and 90th percentiles
    median = np.median(data)
    first_quartile = np.percentile(data, 25)
    third_quartile = np.percentile(data, 75)
    p5 = np.percentile(data, 5)
    p90 = np.percentile(data, 90)
    
    # Display results for this trace
    print(f"\nStatistics for {trace_label}:")
    print(f"Mean: {mean}")
    print(f"Second Moment: {second_moment}")
    print(f"Third Moment: {third_moment}")
    print(f"Fourth Moment: {fourth_moment}")
    print(f"Variance: {variance}")
    print(f"Third Centered Moment: {third_centered_moment}")
    print(f"Fourth Centered Moment: {fourth_centered_moment}")
    print(f"Skewness: {skewness}")
    print(f"Fourth Standardized Moment: {fourth_standardized_moment}")
    print(f"Standard Deviation: {std_dev}")
    print(f"Coefficient of Variation: {coef_variation}")
    print(f"Excess Kurtosis: {excess_kurtosis}")
    print(f"Median: {median}")
    print(f"First Quartile: {first_quartile}")
    print(f"Third Quartile: {third_quartile}")
    print(f"5th Percentile: {p5}")
    print(f"90th Percentile: {p90}")
    
    return {
        "mean": mean,
        "second_moment": second_moment,
        "third_moment": third_moment,
        "fourth_moment": fourth_moment,
        "variance": variance,
        "third_centered_moment": third_centered_moment,
        "fourth_centered_moment": fourth_centered_moment,
        "skewness": skewness,
        "fourth_standardized_moment": fourth_standardized_moment,
        "std_dev": std_dev,
        "coef_variation": coef_variation,
        "excess_kurtosis": excess_kurtosis,
        "median": median,
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "p5": p5,
        "p90": p90
    }

# Step 6: Pearson Correlation Coefficient for Lags (m = 1 to m = 100)
def plot_pearson_corr(data, trace_label):
    autocorrs = [np.corrcoef(data[:-m], data[m:])[0, 1] for m in range(1, 101)]
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 101), autocorrs, label=f'Pearson Correlation for {trace_label}')
    plt.title(f'Pearson Correlation Coefficient for Lags (1-100) - {trace_label}')
    plt.xlabel('Lag (m)')
    plt.ylabel('Pearson Correlation Coefficient')
    plt.grid(True)
    plt.legend()
    plt.show()

# Step 7: CDF Plot
def plot_cdf(data, trace_label):
    sorted_data = np.sort(data)
    yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, yvals, label=f'CDF for {trace_label}')
    plt.title(f'Cumulative Distribution Function (CDF) - {trace_label}')
    plt.xlabel('Inter-Arrival Times')
    plt.ylabel('CDF')
    plt.grid(True)
    plt.legend()
    plt.show()

# Loop through all traces and compute statistics, plot graphs
for i, trace in enumerate(traces):
    trace_label = f"Trace {i+1}"
    
    # Compute and display statistics for this trace
    compute_and_display_statistics(trace, trace_label)
    
    # Plot Pearson correlation coefficient
    plot_pearson_corr(trace, trace_label)
    
    # Plot CDF
    plot_cdf(trace, trace_label)
