import redis
import json
import numpy as np
import time
import sys

# Initialize Redis client
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
NoneType = type(None)
# Load the target schedule from the JSON file
with open('target_schedule.json', 'r') as f:
    target_schedule = json.load(f)

# Initialize lists to store RMSD and cost results
Cost_list = []
RMSD_list = []

pubsub = redis_client.pubsub()
pubsub.subscribe("algorithm.EA.epoch.1")

# Define cost rates
COST_PV = 0.0  # Free
COST_WT = 0.0  # Free
COST_BATTERY_DISCHARGE = -0.1  # Earn when discharging (positive power generation)
COST_BATTERY_CHARGE = 0.1  # Cost when charging (negative power generation)
COST_CHP = 0.5  # High cost
def loading_symbol(message, counter=0):
    sys.stdout.write(message + " " + '|/-\\'[(counter %4)])
    sys.stdout.flush()
    time.sleep(1)
    sys.stdout.write((len(message)+2)*'\b')
    return counter + 1

# Function to calculate root mean standard deviation
def root_mean_std_deviation(schedule):
    deviations = [(value - target_schedule[i]) ** 2 for i, value in enumerate(schedule)]
    rmsd = np.sqrt(np.mean(deviations))
    return rmsd

def cost_objective(all_power_generations):
    cost = 0.0

    # all_power_generations is a 2D array: rows are resources, columns are power generation values

    # PVs (assuming they are the first resource)
    pv_schedule = all_power_generations[0] if len(all_power_generations) > 4 else [0] * 96
    cost += np.sum(pv_schedule) * COST_PV

    # WTs (assuming they are the second resource)
    wt_schedule = all_power_generations[1] if len(all_power_generations) > 4 else [0] * 96
    cost += np.sum(wt_schedule) * COST_WT

    # Battery1 (assuming it is the third resource)
    battery1_schedule = all_power_generations[2] if len(all_power_generations) > 4 else [0] * 96
    for power in battery1_schedule:
        if power < 0:  # Discharging
            cost += power * COST_BATTERY_DISCHARGE
        else:  # Charging
            cost += power * COST_BATTERY_CHARGE

    # Battery2 (assuming it is the fourth resource)
    battery2_schedule = all_power_generations[3] if len(all_power_generations) > 4 else [0] * 96
    for power in battery2_schedule:
        if power < 0:  # Discharging
            cost += power * COST_BATTERY_DISCHARGE
        else:  # Charging
            cost += power * COST_BATTERY_CHARGE

    # CHP (assuming it is the fifth resource)
    chp_schedule = all_power_generations[4] if len(all_power_generations) > 4 else [0] * 96
    cost += np.sum(chp_schedule) * COST_CHP

    return cost

# Function to normalize objectives
def normalize(value, max_value):
    return value / max_value


# Function to handle incoming messages
# Example usage within handle_schedules function
def handle_schedules(schedules):
    global Cost_list, RMSD_list

    for plan in schedules:
        resource_plans = plan['resourcePlan']
        all_power_generations = []

        for resource in resource_plans:
            power_generation = np.array(resource['powerGeneration'])
            all_power_generations.append(power_generation)

        all_power_generations = np.array(all_power_generations)

        cost = cost_objective(all_power_generations)
        total_schedule = np.sum(all_power_generations, axis=0).tolist()
        rmsd = root_mean_std_deviation(np.array(total_schedule))

        normalized_cost = normalize(cost, 57.6)
        normalized_rmsd = normalize(rmsd, 7)

        Cost_list.append(normalized_cost)
        RMSD_list.append(normalized_rmsd)

def wait_for_schedules(PubSub):
    i = 0
    while True:
        message = PubSub.get_message()
        if isinstance(message, NoneType):
            i = loading_symbol("Waiting for ec-scheduler to publish new activity matrix", i)
            continue
        if not isinstance(message, NoneType):
            if message["data"] != 1:
                break
    message = message["data"]
    message = json.loads(message)
    return message
# Function to send evaluation results
def send_evaluation_results():
    results = [
        f"{normalized_cost} {normalized_rmsd}"
        for normalized_cost, normalized_rmsd in zip(Cost_list, RMSD_list)
    ]
    results_str = "\n".join(results)
    print(f"Sending evaluation results: {results_str}")
    redis_client.set("proof.result.1", results_str)

# Main loop to process multiple generations
num_generations = 51

for generation in range(num_generations):
    schedule_data = wait_for_schedules(pubsub)
    print(f"Processing generation {generation + 1}/{num_generations}")
    #print(f"Received schedules: {schedule_data}")
    handle_schedules(schedule_data)
    send_evaluation_results()
    #print("Evaluation results sent to Redis.")
    # Clear lists for the next generation
    Cost_list = []
    RMSD_list = []
