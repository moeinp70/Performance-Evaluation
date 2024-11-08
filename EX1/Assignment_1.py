import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files 
sensor1 = np.loadtxt("Logger1.csv", delimiter=";")
sensor2 = np.loadtxt("Logger2.csv", delimiter=";")

# Metrics Calculation using numpy
total_cars_1 = len(sensor1)
total_cars_2 = len(sensor2)

# Ensure the total number of cars is consistent between sensor1 and sensor2
assert total_cars_1 == total_cars_2, "Mismatch in the number of cars between sensor1 and sensor2."

# Total observation time based on sensor1 (arrival) and sensor2 (exit)
total_time_1 = np.max(sensor1) - np.min(sensor1)  # Arrival time window
total_time_2 = np.max(sensor2) - np.min(sensor2)  # Exit time window

# Arrival rate (λ) based on sensor1 (first sensor)
arrival_rate = total_cars_1 / total_time_1

# Throughput (X) based on sensor2 (second sensor)
throughput = total_cars_2 / total_time_2

# Average inter-arrival time (A) between cars
inter_arrival_times = np.diff(sensor1)
average_inter_arrival_time = np.mean(inter_arrival_times)

# Service times for each car (time spent on the road segment)
service_times = sensor2 - sensor1
average_service_time = np.mean(service_times)

# Total busy time (sum of all service times)
total_service_time = np.sum(service_times)

# Utilization (U) is the total busy time divided by the total observation time from sensor2
utilization = total_service_time / total_time_2

# Average number of jobs (N) using Little's Law: N = λ * S
average_number_of_jobs = arrival_rate * average_service_time

# Average response time (R), which is equal to the average service time when no queuing is involved
average_response_time = average_service_time

# Print the requested data
print(f"Arrival Rate (cars/min): {arrival_rate}")
print(f"Throughput (cars/min): {throughput}")
print(f"Average Inter-Arrival Time (min): {average_inter_arrival_time}")
print(f"Utilization: {utilization}")
print(f"Average Service Time (min): {average_service_time}")
print(f"Average Number of Jobs: {average_number_of_jobs}")
print(f"Average Response Time (min): {average_response_time}")

# Plot the number of cars in the road segment 
plt.figure(figsize=(8, 5))
plt.hist(service_times, bins=25, range=(0, 25), edgecolor='black')
plt.title('Number of Cars in the Road Segment (0 to 25)')
plt.xlabel('Number of Cars')
plt.ylabel('Frequency')
plt.show()

# Plot response time distribution 
plt.figure(figsize=(8, 5))
plt.hist(service_times, bins=range(1, 41), edgecolor='black')
plt.title('Response Time Distribution (1 to 40 minutes)')
plt.xlabel('Response Time (minutes)')
plt.ylabel('Frequency')
plt.show()

# Plot service time distribution
plt.figure(figsize=(8, 5))
plt.hist(service_times, bins=np.arange(0.1, 5.1, 0.1), edgecolor='black')
plt.title('Service Time Distribution (0.1 to 5 minutes)')
plt.xlabel('Service Time (minutes)')
plt.ylabel('Frequency')
plt.show()
