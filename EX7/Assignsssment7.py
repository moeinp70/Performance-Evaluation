import numpy as np

# Simulation parameters
total_simulation_time = 20 * 60  # Total simulation time in minutes (20 hours)
state_times = {
    "WaitingForUserInput": 0,
    "CashTransaction": 0,
    "ElectronicTransaction": 0,
    "PrintingTicket": 0
}
transaction_durations = []
cash_collected = 0

# Ticket prices and probabilities from the assignment
ticket_probabilities = [0.9, 0.06, 0.04]
ticket_prices = [2.5, 4, 6]


# Distribution functions

def hyper_exponential(lambda1, lambda2, p1, size=1):
    branch = np.random.choice([0, 1], size=size, p=[p1, 1 - p1])
    return np.where(branch == 0, np.random.exponential(1 / lambda1, size), np.random.exponential(1 / lambda2, size))[0]

def hyper_exponential1(lambda1, lambda2, p1):
    """Time for GUI interaction (Hyper-exponential, row A)."""
    return np.random.exponential(1 / lambda1) if np.random.rand() < p1 else np.random.exponential(1 / lambda2)

def erlang(k, lambd):
    """Erlang distribution for Electronic Payment"""
    return np.sum(-np.log(1 - np.random.rand(1, k)) / lambd, axis=1) [0]#np.sum(np.random.exponential(1 / lambd, k))


def hyper_erlang(p1, k1, lambda1, k2, lambda2):
    """Hyper-Erlang distribution for Printing Ticket"""
    if np.random.rand() < p1:
        return np.sum(-np.log(1 - np.random.rand(1, k1)) / lambda1, axis=1) [0] #np.sum(np.random.exponential(1 / lambda1, k1))
    else:
        return np.random.exponential(1 / lambda2)


# State-specific functions with accurate parameters

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


# Function to determine ticket price based on probabilities
def get_ticket_price():
    ticket_index = np.random.choice([0, 1, 2], p=[0.9, 0.06, 0.04])
    ticket_prices = [2.5, 4, 6]
    return ticket_prices[ticket_index]


# Main simulation function
def simulate(total_simulation_time):
    global cash_collected
    state = "WaitingForUserInput"
    total_elapsed_time = 0
    transaction_start_time = 0

    # Track cumulative time for each state
    cumulative_times = {
        "WaitingForUserInput": 0,
        "CashTransaction": 0,
        "ElectronicTransaction": 0,
        "PrintingTicket": 0
    }

    while total_elapsed_time < total_simulation_time:
        if state == "WaitingForUserInput":
            transaction_start_time = total_elapsed_time
            next_state, duration = waiting_for_user_input()
            cumulative_times["WaitingForUserInput"] += duration

        elif state == "CashTransaction":
            next_state, duration = cash_transaction()
            cumulative_times["CashTransaction"] += duration

        elif state == "ElectronicTransaction":
            next_state, duration = electronic_transaction()
            cumulative_times["ElectronicTransaction"] += duration

        elif state == "PrintingTicket":
            next_state, duration = printing_ticket()
            cumulative_times["PrintingTicket"] += duration
            transaction_duration = total_elapsed_time + duration - transaction_start_time
            transaction_durations.append(transaction_duration)
            cash_collected += get_ticket_price()

        elif state == "LeaveWithoutPurchase":
            next_state = "WaitingForUserInput"
            duration = 0

        elif state == "ChoosePaymentOption":
            next_state = choose_payment_option()
            duration = 0

        # Update total time and transition to next state
        total_elapsed_time += duration
        state = next_state

    # Calculate state probabilities
    probabilities = {state: cumulative_times[state] / total_elapsed_time for state in cumulative_times}

    # Calculate average transaction duration
    average_transaction_duration = np.mean(transaction_durations) if transaction_durations else 0

    # Calculate average cash collected per hour
    average_cash_per_hour = cash_collected / (total_simulation_time / 60)

    return probabilities, average_transaction_duration, cash_collected, average_cash_per_hour


# Run the simulation
probabilities, average_transaction_duration, total_cash_collected, average_cash_per_hour = simulate(
    total_simulation_time)

# Display results
print("Estimated Steady-State Probabilities:")
for state, prob in probabilities.items():
    print(f"{state}: {prob:.4f}")

print("\nResults:")
print(f"Average Transaction Duration: {average_transaction_duration:.2f} minutes")
print(f"Total Cash Collected in 20 Hours: €{total_cash_collected:.2f}")
print(f"Average Cash Collected per Hour: €{average_cash_per_hour:.2f}")
