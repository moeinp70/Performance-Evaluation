import numpy as np

#arrival_times = np.loadtxt("Logger1.csv", delimiter=";")  # Inter-arrival times in minutes
service_durations = np.loadtxt("Logger2.csv", delimiter=";")  # Service times in minutes

avg_service_duration = np.mean(service_durations)
target_response_time_20min = 20  

# Calculate maximum arrival rate to achieve a response time of 20 minutes
max_allowed_arrival_rate = ((target_response_time_20min / avg_service_duration) - 1) / target_response_time_20min

print(f"1. Maximum Arrival Rate for 20-minute Response Time: {max_allowed_arrival_rate:.8f} jobs/min")

increased_arrival_rate = 1.2  
target_response_time = 15  

# Calculate the required beta to reduce the service time
required_service_reduction = target_response_time / (avg_service_duration + increased_arrival_rate * target_response_time * avg_service_duration)

print(f"2. Required Service Time Reduction Factor (β): {required_service_reduction:.8f}")
