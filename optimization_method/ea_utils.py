import numpy
import pickle
from deap import tools
import pandas as pd
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import redis
import json
import os
from genotyp_phenotyp import GenotypPhenotyp

# =========================
# SECTION 1: Initialize Redis, define the selection function, and save the final schedule
# =========================

# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379)


def selWeighted(individuals, k, use_weights=True):
    """Select the k best individuals among the input individuals.

    Parameters:
    - individuals: A list of individuals to select from.
    - k: The number of individuals to select.
    - use_weights: Boolean parameter indicating whether to use the weighted fitness values.

    Returns:
    - A list of selected individuals.
    """
    if use_weights:
        sorted_individuals = sorted(individuals,
                                    key=lambda ind: (0.6 * ind.fitness.values[0]) + (0.4 * ind.fitness.values[1]),
                                    reverse=True)
    else:
        sorted_individuals = sorted(individuals, key=lambda ind: (ind.fitness.values[0]) + (ind.fitness.values[1]),
                                    reverse=True)

    return sorted_individuals[:k]


# Function to fetch the final schedule JSON data from EMS and save it to a file
def fetch_and_save_json(path):
    time.sleep(2)
    json_data = r.get('ems_schedule_set')
    if json_data:
        file_name = f"{path}.json"

        with open(file_name, 'w') as outfile:
            json.dump(json.loads(json_data), outfile)
        # Clear the data in the Redis key
        r.delete('ems_schedule_set')
        print(f"Saved JSON data to file: {file_name}")
    else:
        print("No JSON data found in Redis")


# =========================
# SECTION 2: Define the helper functions for the parallel evaluation
# =========================

# Helper function to bundle chunk with its microservice index
def chunk_with_service_index(pop, num_chunks):
    chunk_size = len(pop) // num_chunks
    remainder = len(pop) % num_chunks

    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + chunk_size + (1 if i < remainder else 0)
        chunks.append((pop[start:end], i + 1))
        start = end
    return chunks


# Define the standalone function for encapsulating the genotype to phenotype transformation
def encapsulate_genotype_phenotype(args):
    return genotyp_phenotyp_encapsulation(*args)


def evaluate_population(pop, islands):
    with Pool(processes=islands) as pool:
        results = pool.map(encapsulate_genotype_phenotype, chunk_with_service_index(pop, islands))
    fitnesses = [fit for sublist in results for fit in sublist]  # Flatten the list of lists
    return fitnesses


# Encapsulates the main function in GenotypPhenotyp, which converts the gene structure to a schedule readable by EMS
# it also receives the evaluation results from EMS and convert them to exponential notes
def genotyp_phenotyp_encapsulation(chromosome_list, microservice_index):
    # Call the convert method for evaluation
    fitness_values = GenotypPhenotyp.interpretation(GenotypPhenotyp, chromosome_list, microservice_index)
    return fitness_values


# =========================
# SECTION 3: Define the statistics, plotting, and and logging functions
# =========================

# Statistics
stats_fit = tools.Statistics(key=lambda fit: fit)
mstats = tools.MultiStatistics(fitness=stats_fit)
mstats.register("mean_value_weighted", lambda x: int(0.6 * x[0][0] + 0.4 * x[0][1]))
mstats.register("mean_value", lambda x: int(x[0][0] + x[0][1]))

# Logbook for the final day
final_logbook = tools.Logbook()
final_logbook.header = "day", "RMSD", "cost", "RMSD_note", "cost_note", "Weighted_RMSD_note", "Weighted_cost_note", "whole_note", "time", "number_of_ems_evaluations"

# Logbook for every generation
gen_logbook = tools.Logbook()
gen_logbook.header = "day", "gen", "RMSD", "cost", "RMSD_note", "cost_note", "Weighted_RMSD_note", "Weighted_cost_note", "whole_note", "generation_time", "evaluation_time", "number_of_ems_evaluations"

# Statistics to print
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
logbook = tools.Logbook()
logbook.header = ['run_id', 'day', 'gen', 'nevals', 'beval', 'sum'] + stats.fields


# Plot the evolution of every function value over the generations
def plot_evolution(gen_logbook, day, filename):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.clf()  # Clear the current figure
    day_data = [record for record in gen_logbook if record['day'] == day]
    EMS_evaluations = [record['number_of_ems_evaluations'] for record in day_data]
    cost_values = [record['cost_note'] for record in day_data]
    RMSD_values = [record['RMSD_note'] for record in day_data]

    # Calculate the cumulative sum of EMS evaluations
    cumulative_EMS_evaluations = [sum(EMS_evaluations[:i + 1]) for i in range(len(EMS_evaluations))]

    plt.figure(figsize=(10, 6))

    # Plot Cost
    plt.plot(cumulative_EMS_evaluations, cost_values, label="Cost")

    # Plot RMSD
    plt.plot(cumulative_EMS_evaluations, RMSD_values, label="RMSD")

    plt.title(f"Evolution of Parameters over Generations (Day {day})")
    plt.xlabel("Number Of EMS Evaluations")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot to a file
    plt.show()


# Plot the Pareto front for the Hall of Fame individuals
def plot_hall_of_fame(hall_of_fame, day, filename, weighted=False):
    plt.figure(figsize=(10, 6))
    # Extract the fitness values for each individual in the Hall of Fame
    if weighted:
        fitness_values = [ind.original_fitness for ind in hall_of_fame]
    else:
        fitness_values = [ind.fitness.values for ind in hall_of_fame]
    # Separate the fitness values into two lists for the two axes
    x_values, y_values = zip(*fitness_values)
    plt.scatter(x_values, y_values, label='Hall of Fame')
    plt.xlabel('RMSD')
    plt.ylabel('Cost')
    plt.title(f'Hall Of Fame (Day {day})')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# Plot the pareto front at the end of the evaluation
def plot_pareto_front(pareto_front, day, filename, weighted=False):
    plt.figure(figsize=(10, 6))
    for i, front in enumerate(pareto_front):
        # Extract the fitness values for each individual in the Pareto front
        if weighted:
            fitness_values = [ind.original_fitness for ind in front]
        else:
            fitness_values = [ind.fitness.values for ind in front]
        # Separate the fitness values into two lists for the two axes
        x_values, y_values = zip(*fitness_values)
        plt.scatter(x_values, y_values, label=f'Front {i + 1}')
    plt.xlabel('RMSD')
    plt.ylabel('Cost')
    plt.title(f'Pareto Front (Day {day})')
    plt.legend()
    plt.savefig(filename)
    plt.close()


def save_logs(run_id, filename, data, dataframe_columns=None, mode='wb', is_pareto=False):
    """
    Saves the given data to a pickle file and a CSV file in the specified run directory.

    Parameters:
    - run_id: Identifier for the run.
    - filename: Base name for the files (without extension).
    - data: The data to be saved.
    - dataframe_columns: Optional; if provided, specifies how to extract data for the DataFrame.
    - mode: File open mode for the pickle file ('wb' for write binary, 'ab' for append binary).
    - is_pareto: Boolean; if True, data is treated as Pareto front (list of tuples).
    """
    # Prepare data if it's Pareto front
    if is_pareto:
        data = [(ind, ind.fitness.values) for ind in data]

    # Save the data to a pickle file
    with open(f'logs/run_{run_id}/{filename}.pkl', mode) as file:
        pickle.dump(data, file)

    # Convert data to a DataFrame if necessary
    if is_pareto:
        df_data = pd.DataFrame(data, columns=['Individual', 'Fitness Values'])
    elif dataframe_columns is not None:
        df_data = pd.DataFrame(dataframe_columns)
    else:
        df_data = pd.DataFrame(data)

    # Save the DataFrame to a CSV file
    df_data.to_csv(f'logs/run_{run_id}/{filename}.csv', index=False)
