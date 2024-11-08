
import pandas as pd
import numpy as np

# Load data from CSV files
inter_arrival_times = pd.read_csv('Logger1.csv', header=None).squeeze("columns").values
service_times = pd.read_csv('Logger2.csv', header=None).squeeze("columns").values

# Average inter-arrival time and average service time
average_inter_arrival_time = np.mean(inter_arrival_times)
average_service_time = np.mean(service_times)

# Question 1: Maximum arrival rate for average response time < 20 minutes
target_response_time_1 = 20

a= ((20 / average_service_time)-1) / 20
print(a)
def calculate_response_time(arrival_rate, service_time):
    utilization = arrival_rate * service_time

    if utilization >= 1:
        return float('inf')  # System is unstable
    return service_time / (1 - utilization)

# Find maximum arrival rate for response time less than 20 minutes
max_arrival_rate = 0
arrival_rate_step = 0.01  # Step for increasing arrival rate

while True:
    response_time = calculate_response_time(max_arrival_rate, average_service_time)
    if response_time >= target_response_time_1:
        break
    max_arrival_rate += arrival_rate_step

print(f"1. Maximum arrival rate for a response time less than 20 minutes: {max_arrival_rate:.8f} cars per minute")

# Question 2: Fraction beta for response time < 15 minutes with arrival rate of 1.2 jobs per minute
target_arrival_rate_2 = 1.2
target_response_time_2 = 15

def calculate_response_time_with_beta(beta):
    adjusted_service_time = beta * average_service_time
    return calculate_response_time(target_arrival_rate_2, adjusted_service_time)

# Find minimum beta for response time less than 15 minutes
beta_values = np.linspace(0.01, 1.0, 1000)  # Testing values for beta
min_beta = None

for beta in beta_values:
    response_time = calculate_response_time_with_beta(beta)
    if response_time < target_response_time_2:
        min_beta = beta
        break

if min_beta is not None:
    print(f"2. Fraction beta to achieve response time less than 15 minutes: {min_beta:.2f}")
else:
    print("No feasible beta found to achieve response time less than 15 minutes.")
