import numpy as np

# Simulation parameters
total_simulation_time = 20 * 60  # Total simulation time in minutes (20 hours)
num_simulations = 1000  # Number of times to run the 20-hour simulation

# Ticket prices and probabilities from Table II
ticket_probabilities = [0.9, 0.06, 0.04]
ticket_prices = [2.5, 4, 6]

# Distribution functions

def hyper_exponential(lambda1, lambda2, p1, size=1):
    branch = np.random.choice([0, 1], size=size, p=[p1, 1 - p1])
    return np.where(branch == 0, np.random.exponential(1 / lambda1, size), np.random.exponential(1 / lambda2, size))[0]

def erlang(k, lambd):
    return np.sum(-np.log(1 - np.random.rand(1, k)) / lambd, axis=1)[0]

def hyper_erlang(p1, k1, lambda1, k2, lambda2):
    if np.random.rand() < p1:
        return np.sum(-np.log(1 - np.random.rand(1, k1)) / lambda1, axis=1)[0]
    else:
        return np.random.exponential(1 / lambda2)

# State-specific functions

def waiting_for_user_input():
    duration = hyper_exponential(lambda1=0.4, lambda2=0.1, p1=0.8)
    next_state = np.random.choice(["LeaveWithoutPurchase", "ChoosePaymentOption"], p=[0.2, 0.8])
    return next_state, duration

def choose_payment_option():
    return np.random.choice(["CashTransaction", "ElectronicTransaction"], p=[0.35, 0.65])

def cash_transaction():
    duration = np.random.exponential(1 / 0.4)
    return "PrintingTicket", duration

def electronic_transaction():
    duration = erlang(k=4, lambd=2)
    return "PrintingTicket", duration

def printing_ticket():
    duration = hyper_erlang(p1=0.95, k1=2, lambda1=10, k2=1, lambda2=0.1)
    return "WaitingForUserInput", duration

def get_ticket_price():
    ticket_index = np.random.choice([0, 1, 2], p=ticket_probabilities)
    return ticket_prices[ticket_index]

# Function to simulate one 20-hour period
def simulate_once(total_simulation_time):
    cumulative_times = {
        "WaitingForUserInput": 0,
        "CashTransaction": 0,
        "ElectronicTransaction": 0,
        "PrintingTicket": 0
    }
    transaction_durations = []
    cash_collected = 0
    cash_only_collected = 0  # To track cash collected through cash transactions only
    total_elapsed_time = 0

    while total_elapsed_time < total_simulation_time:
        transaction_start_time = total_elapsed_time

        # Step 1: Waiting for User Input
        next_state, duration = waiting_for_user_input()
        cumulative_times["WaitingForUserInput"] += duration
        total_elapsed_time += duration

        if next_state == "LeaveWithoutPurchase":
            continue

        # Step 2: Payment Selection
        if next_state == "ChoosePaymentOption":
            next_state = choose_payment_option()

        # Cash Transaction
        if next_state == "CashTransaction":
            next_state, duration = cash_transaction()
            cumulative_times["CashTransaction"] += duration
            total_elapsed_time += duration
            ticket_price = get_ticket_price()
            cash_collected += ticket_price
            cash_only_collected += ticket_price  # Track only cash payments here

        # Electronic Transaction
        elif next_state == "ElectronicTransaction":
            next_state, duration = electronic_transaction()
            cumulative_times["ElectronicTransaction"] += duration
            total_elapsed_time += duration
            cash_collected += get_ticket_price()  # Add to total cash but not to cash_only_collected

        # Step 3: Printing Ticket
        if next_state == "PrintingTicket":
            next_state, duration = printing_ticket()
            cumulative_times["PrintingTicket"] += duration
            total_elapsed_time += duration

            # Complete transaction and collect cash
            transaction_duration = total_elapsed_time - transaction_start_time
            transaction_durations.append(transaction_duration)

    # State probabilities for this simulation
    total_time_in_states = sum(cumulative_times.values())
    probabilities = {state: cumulative_times[state] / total_time_in_states for state in cumulative_times}
    average_transaction_duration = np.mean(transaction_durations) if transaction_durations else 0
    average_cash_per_hour = cash_collected / (total_simulation_time / 60)

    return probabilities, average_transaction_duration, cash_collected, cash_only_collected, average_cash_per_hour

# Run multiple simulations and average the results
overall_probabilities = {state: 0 for state in ["WaitingForUserInput", "CashTransaction", "ElectronicTransaction", "PrintingTicket"]}
total_average_transaction_duration = 0
total_cash_collected = 0
total_cash_only_collected = 0  # Track only cash transactions
total_average_cash_per_hour = 0

for _ in range(num_simulations):
    probabilities, avg_duration, cash, cash_only, avg_cash_per_hour = simulate_once(total_simulation_time)

    for state in overall_probabilities:
        overall_probabilities[state] += probabilities[state]
    total_average_transaction_duration += avg_duration
    total_cash_collected += cash
    total_cash_only_collected += cash_only  # Accumulate cash-only transactions
    total_average_cash_per_hour += avg_cash_per_hour

# Calculate averages
for state in overall_probabilities:
    overall_probabilities[state] /= num_simulations
average_transaction_duration = total_average_transaction_duration / num_simulations
average_cash_collected = total_cash_collected / num_simulations
average_cash_only_collected = total_cash_only_collected / num_simulations  # Average for cash-only collection
average_cash_per_hour = total_average_cash_per_hour / num_simulations

# Display results
print(f"Probability of waiting for user input: {overall_probabilities['WaitingForUserInput']:.4f}")
print(f"Probability of handling a cash transaction: {overall_probabilities['CashTransaction']:.4f}")
print(f"Probability of handling an electronic transaction: {overall_probabilities['ElectronicTransaction']:.4f}")
print(f"Probability of printing the ticket: {overall_probabilities['PrintingTicket']:.4f}")
print(f"Average duration of a transaction (in minutes): {average_transaction_duration:.2f}")
print(f"Average amount of cash collected by the machine in 20 hours: €{average_cash_collected:.2f}")
print(f"Average amount of cash collected (coins) from cash transactions in 20 hours: €{average_cash_only_collected:.2f}")
