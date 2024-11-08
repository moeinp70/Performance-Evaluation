import numpy as np
import pandas as pd
from scipy.linalg import solve

# Load Data
data = pd.DataFrame([
    [240, 20, 5, 30, 10, 5],
    [300, 10, 40, 30, 50, 3],
    [210, 25, 10, 60, 30, 3],
    [235, 20, 20, 30, 20, 4],
    [350, 10, 50, 20, 50, 5],
    [185, 40, 20, 90, 10, 3],
    [220, 30, 10, 30, 10, 3],
    [320, 10, 5, 20, 5, 3],
    [260, 20, 0, 60, 0, 5],
    [480, 50, 0, 120, 0, 8]
], columns=["Length", "ExtProb", "SkipProb", "ExtLen", "SkipIfExtended", "Fee"])


# Transition rates
lengths = data["Length"].values
extension_probs = data["ExtProb"].values / 100  # Convert percentages to probabilities
skip_probs = data["SkipProb"].values / 100
extension_lengths = data["ExtLen"].values
skip_if_extended = data["SkipIfExtended"].values / 100
fees = data["Fee"].values




# Initialize a 20x20 matrix Q with zeros
Q = np.zeros((20, 20))

# Populate Q with transition rates based on the data
for i in range(10):
    base_rate = 1 / lengths[i]  # Rate of leaving the current song
    ext_rate = base_rate * extension_probs[i]
    skip_rate = base_rate * skip_probs[i]
    continue_rate = base_rate * (1 - extension_probs[i] - skip_probs[i])

    # Main song transitions
    Q[i, i + 1 if i < 9 else 0] = continue_rate       # Move to next song
    Q[i, i + 10] = ext_rate                           # Move to extended version
    if i < 8:  # Skip to the song after next if applicable
        Q[i, i + 2] = skip_rate

    # Extended song transitions
    ext_base_rate = 1 / extension_lengths[i] if extension_lengths[i] > 0 else 0
    Q[i + 10, i + 1 if i < 9 else 0] = ext_base_rate * (1 - skip_if_extended[i])  # Continue to next song after extension
    if i < 8:
        Q[i + 10, i + 2] = ext_base_rate * skip_if_extended[i]  # Skip to song after next from extended

# Set diagonal entries to satisfy row sums to zero
for i in range(20):
    Q[i, i] = -np.sum(Q[i, :])



# Add normalization condition
Q[:, 0] = np.ones(20)
b = np.zeros(20)
b[0] = 1

# Solve for steady-state probabilities
pi = solve(Q.T, b)



prob_song_1 = pi[0]
prob_song_2 = pi[1]
prob_song_5 = pi[4]
prob_song_9 = pi[8]
prob_song_10 = pi[9]

print("Probability of hearing song 1:", pi[0])
print("Probability of hearing song 2:", pi[1])
print("Probability of hearing song 5:", pi[4])
print("Probability of hearing song 9:", pi[8])
print("Probability of hearing song 10:", pi[9])


avg_cost = np.dot(pi[:10], fees)
print("Average cost of songs (in Euros):", avg_cost)


avg_show_duration_seconds = np.sum(pi[:10] * lengths + pi[10:] * extension_lengths)
shows_per_day = 86400 / avg_show_duration_seconds
print("Number of shows per day:", shows_per_day)


avg_show_duration_minutes = avg_show_duration_seconds / 60
print("Average duration of a show (in minutes):", avg_show_duration_minutes)
