import numpy as np

# Parameters for Scenario 1 and Scenario 2
srv_weibull_k, srv_weibull_lambda = 0.333, 2.5  # Weibull for Scenario 1
arr_lambda1, arr_lambda2, arr_p1 = 0.025, 0.1, 0.35  # Hyper-exponential for Scenario 1
srv_uniform_a, srv_uniform_b = 1, 10  # Uniform for Scenario 2
arr_erlang_k, arr_erlang_lambda = 8, 1.25  # Erlang for Scenario 2

# Simulation parameters
M = 5000
max_rel_err = 0.02
confidence_level = 1.96  # z-score for 95% confidence
K0 = 100
DeltaK = 100
MaxK = 15000

# Distributions for arrival and service times
# Corrected hyper-exponential distribution for arrival times
def hyper_exponential(lambda1, lambda2, p1, size):
    branch = np.random.choice([0, 1], size=size, p=[p1, 1 - p1])
    return np.where(branch == 0, np.random.exponential(1 / lambda1, size), np.random.exponential(1 / lambda2, size))

def weibull(srv_weibull_k, srv_weibull_lambda, size):
    return srv_weibull_lambda * (-np.log(1 - np.random.rand(size))) ** (1 / srv_weibull_k)

def uniform(a, b, size):
    return np.random.uniform(a, b, size)

def erlang(k, lambd, size):
    return np.sum(-np.log(1 - np.random.rand(size, k)) / lambd, axis=1)

# Scenario runner function with detailed output for all metrics
def run_scenario(arrival_func, service_func, scenario_name):
    K = K0
    U1, U2, Th1, Th2, NJ1, NJ2, RT1, RT2 = 0, 0, 0, 0, 0, 0, 0, 0  # Summations for variance calculations

    while K <= MaxK:
        for _ in range(DeltaK):
            # Generate arrival and service times
            arrival_times = np.cumsum(arrival_func(M))
            service_times = service_func(M)

            # Calculate completion times
            completion_times = np.zeros(M)
            completion_times[0] = arrival_times[0] + service_times[0]
            for i in range(1, M):
                completion_times[i] = max(arrival_times[i], completion_times[i - 1]) + service_times[i]

            # Metrics for the current batch
            T = completion_times[-1] - arrival_times[0]
            B = np.sum(service_times)
            Utilization = B / T
            Throughput = M / T
            AvgRespTime = np.mean(completion_times - arrival_times)
            AvgJobs = Throughput * AvgRespTime  # Using Little's Law

            # Accumulate sums for all metrics
            U1 += Utilization
            U2 += Utilization ** 2
            Th1 += Throughput
            Th2 += Throughput ** 2
            NJ1 += AvgJobs
            NJ2 += AvgJobs ** 2
            RT1 += AvgRespTime
            RT2 += AvgRespTime ** 2

        # Calculate means and variances for each metric
        EU, EU2 = U1 / K, U2 / K
        ETh, ETh2 = Th1 / K, Th2 / K
        ENJ, ENJ2 = NJ1 / K, NJ2 / K
        ERT, ERT2 = RT1 / K, RT2 / K

        VarU = EU2 - EU ** 2
        VarTh = ETh2 - ETh ** 2
        VarNJ = ENJ2 - ENJ ** 2
        VarRT = ERT2 - ERT ** 2

        # Confidence intervals for each metric
        def confidence_interval(mean, var, K):
            half_width = confidence_level * np.sqrt(var / K)
            return mean - half_width, mean + half_width

        utilization_ci = confidence_interval(EU, VarU, K)
        throughput_ci = confidence_interval(ETh, VarTh, K)
        avg_jobs_ci = confidence_interval(ENJ, VarNJ, K)
        avg_response_time_ci = confidence_interval(ERT, VarRT, K)

        # Relative errors for each metric
        def relative_error(ci):
            return 2 * (ci[1] - ci[0]) / (ci[1] + ci[0])

        utilization_rel_err = relative_error(utilization_ci)
        throughput_rel_err = relative_error(throughput_ci)
        avg_jobs_rel_err = relative_error(avg_jobs_ci)
        avg_response_time_rel_err = relative_error(avg_response_time_ci)

        # Stopping criteria: Stop when all metrics meet the relative error threshold
        if (utilization_rel_err < max_rel_err and throughput_rel_err < max_rel_err and
            avg_jobs_rel_err < max_rel_err and avg_response_time_rel_err < max_rel_err):
            break

        # Increment batch count by DeltaK if criteria are not met
        K += DeltaK

    # Output formatting to match the example provided
    print(f"============ {scenario_name} ============")
    print(f"Maximum Relative Error reached in {K} batches")
    print(f"Utilization in {utilization_ci}, with 95.0% confidence. Relative Error: {utilization_rel_err}")
    print(f"Throughput in {throughput_ci}, with 95.0% confidence. Relative Error: {throughput_rel_err}")
    print(f"Average #jobs in {avg_jobs_ci}, with 95.0% confidence. Relative Error: {avg_jobs_rel_err}")
    print(f"Average Resp. Time in {avg_response_time_ci}, with 95.0% confidence. Relative Error: {avg_response_time_rel_err}")
    print("=========================================")

# Run both scenarios with the correct structure and output format
print("Running Scenario 1...")
run_scenario(lambda M: hyper_exponential(arr_lambda1, arr_lambda2, arr_p1, M),
             lambda M: weibull(srv_weibull_k, srv_weibull_lambda, M),
             "SCENARIO 1")

print("\nRunning Scenario 2...")
run_scenario(lambda M: erlang(arr_erlang_k, arr_erlang_lambda, M),
             lambda M: uniform(srv_uniform_a, srv_uniform_b, M),
             "SCENARIO 2")
