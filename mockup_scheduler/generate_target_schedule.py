import json
import numpy as np

# Generate a target schedule with 96 values between 0.0 and 1.0
target_schedule = np.random.uniform(-2.0, 5.0, 96).tolist()

# Save the target schedule to a JSON file
with open('target_schedule.json', 'w') as f:
    json.dump(target_schedule, f)

print("Target schedule saved to target_schedule.json")