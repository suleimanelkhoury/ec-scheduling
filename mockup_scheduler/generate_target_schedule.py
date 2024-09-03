import json
import numpy as np

# Generate a target schedule with 96 values between 0.0 and 1.0
initial_schedule = np.random.uniform(-2.0, 5.0, 96).tolist()

# Add Moving Average to the target schedule to smooth it
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

# Apply moving average with a window size of 5
target_schedule = moving_average(initial_schedule, window_size=5).tolist()

# Save the target schedule to a JSON file
with open('target_schedule.json', 'w') as f:
    json.dump(target_schedule, f)

print("Target schedule saved to target_schedule.json")