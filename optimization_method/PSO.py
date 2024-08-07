import random
import math
import pickle
import numpy
import operator
import pandas as pd
from deap import base, creator, tools
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
import redis
import json
import os
from genotyp_phenotyp import GenotypPhenotyp

# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379)

# Initialize toolbox
toolbox = base.Toolbox()

# Register the attributes for each gene
toolbox.register("unit_id", random.choice, [1, 2, 3, 4, 6])
toolbox.register("start_time", random.randint, 0, 96)
toolbox.register("duration", random.randint, 1, 96)
toolbox.register("power_fraction_producer", random.uniform, 0.0, 1.0)
toolbox.register("power_fraction_storage", random.uniform, -1.0, 1.0)


# Define a function to create a gene
def create_gene():
    unit_id_value = toolbox.unit_id()
    start_time_value = toolbox.start_time()
    duration_value = toolbox.duration()
    if unit_id_value in [3, 4]:
        power_fraction_value = toolbox.power_fraction_storage()
    else:
        power_fraction_value = toolbox.power_fraction_producer()
    return unit_id_value, start_time_value, duration_value, power_fraction_value


# Define a function to create an individual with a random number of genes
def create_individual():
    num_genes = random.randint(180, 180)  # Range of genes must be constant
    return [create_gene() for _ in range(num_genes)]


# Create the fitness and particle classes
creator.create("FitnessMulti", base.Fitness, weights=(0.6, 0.4))
creator.create("Particle", list, fitness=creator.FitnessMulti, speed=list,
               smin=None, smax=None, best=None)


# Define bounds checking function
def checkBounds(particle):
    for i, gene in enumerate(particle):
        unit_id, start_time, duration, power_fraction = gene

        # Check bounds for each allele in the gene
        unit_id = max(min(int(unit_id), 6), 1)  # Bound unit_id between 1 and 6
        if unit_id == 5:
            unit_id = random.choice([1, 2, 3, 4, 6])
        start_time = max(min(int(start_time), 94), 0)  # Bound start_time between 0 and 96
        duration = max(min(int(duration), 95), 1)  # Bound start_time between 0 and 96
        # Adjust start_time if it leaves no room for duration
        if start_time + duration > 96:
            duration = 96 - start_time

        # Check bounds for power_fraction based on unit_id
        if unit_id in [3, 4]:
            power_fraction = max(min(power_fraction, 1.0), -1.0)
        else:
            power_fraction = max(min(power_fraction, 1.0), 0.0)

        # Update the gene with the bounded values
        particle[i] = (unit_id, start_time, duration, power_fraction)

    return particle

# Calculate the smin and smax values for the velocity limits dynamically based on the scaling factor
def calculate_velocity_limits(scaling_factor):
    bounds = [(0,10),(0, 94), (0, 94), (-1.0, 1.0)]
    smin = []
    smax = []
    for _ in range(240):  # Repeat for the number of genes
        smin_gene = []
        smax_gene = []
        for lower, upper in bounds:  # Skip the first element (unit_id)
            range_val = upper - lower
            smax_gene.append(scaling_factor * range_val)
            smin_gene.append(-scaling_factor * range_val)
        smin.append(smin_gene)
        smax.append(smax_gene)
    return smin, smax


# Generate a particle
def generate(scaling_factor):
    part = creator.Particle(create_individual())
    smin, smax = calculate_velocity_limits(scaling_factor)
    # Initialize speed for each allele in each gene of the individual
    part.speed = [
        [random.uniform(smin_gene[i], smax_gene[i]) for i, _ in enumerate(gene)]
        for gene, smin_gene, smax_gene in zip(part, smin, smax)  # Skip unit_id
    ]

    part.smin = smin
    part.smax = smax

    return part

# Update the particle
def updateParticle(part, best, phi1, phi2):
    for i, gene in enumerate(part):
        # Generate two random coefficients u1 and u2 for each dimension of the particle.
        u1 = [random.uniform(0, phi1) for _ in gene]
        u2 = [random.uniform(0, phi2) for _ in gene]
        # Calculate the cognitive component v_u1 and  the social component v_u2.
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best[i], gene))
        v_u2 = map(operator.mul, u2, map(operator.sub, best[i], gene))
        # Update the speed of the particle by combining its current speed with the cognitive and social components.
        part.speed[i] = list(map(operator.add, part.speed[i], map(operator.add, v_u1, v_u2)))

        # Enforce the speed limits on the particle.
        for j, allele in enumerate(gene):
            if abs(part.speed[i][j]) < part.smin[i][j]:
                part.speed[i][j] = math.copysign(part.smin[i][j], part.speed[i][j])
            elif abs(part.speed[i][j]) > part.smax[i][j]:
                part.speed[i][j] = math.copysign(part.smax[i][j], part.speed[i][j])
        # Ensure the first three speeds are integers, and the fourth remains unchanged.
        for j in range(3):
            part.speed[i][j] = int(part.speed[i][j])

        part[i] = list(map(operator.add, gene, part.speed[i]))
    # Apply bounds checking to the particle
    #print(part.speed[i])
    part[:] = checkBounds(part)
    return part

# Register the toolbox functions with parameters for smin and smax
def register_toolbox(scaling_factor, phi1, phi2):
    toolbox.register("particle", generate, scaling_factor=scaling_factor)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=phi1, phi2=phi2)

def selWeighted(individuals, k, use_weights=True):
    """Select the k best individuals among the input individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param use_weights: Boolean parameter indicating whether to use weighted sorting.
    :returns: A list of selected individuals.
    """
    if use_weights:
        sorted_individuals = sorted(individuals,
                                    key=lambda part: (0.6 * part.fitness.values[0]) + (0.4 * part.fitness.values[1]),
                                    reverse=True)
    else:
        sorted_individuals = sorted(individuals, key=lambda part: (part.fitness.values[0]) + (part.fitness.values[1]),
                                    reverse=True)

    return sorted_individuals[:k]


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
    cumulative_EMS_evaluations = [sum(EMS_evaluations[:i+1]) for i in range(len(EMS_evaluations))]

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
def plot_hall_of_fame(hall_of_fame, day, filename):
    plt.figure(figsize=(10, 6))
    # Extract the fitness values for each individual in the Hall of Fame
    fitness_values = [ind.original_fitness for ind in hall_of_fame]
    # Separate the fitness values into two lists for the two axes
    x_values, y_values = zip(*fitness_values)
    plt.scatter(x_values, y_values, label='Hall of Fame')
    plt.xlabel('RMSD')
    plt.ylabel('Cost')
    plt.title(f'Hall Of Fame (Day {day})')
    plt.legend()
    plt.savefig(filename)
    plt.close()

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

# Encapsulates the main function in GenotypPhenotyp, which converts the gene structure to a schedule readable by EMS
# it also receives the evaluation results from EMS and convert them to exponential notes
def genptyp_phenotyp_encapsulation(chromosome_list, microservice_index):
    # Use the microservice_index if needed for selecting the microservice
    #print(f"Evaluating with microservice {microservice_index}")
    # Call the convert method for evaluation
    fitness_values = GenotypPhenotyp.main(GenotypPhenotyp, chromosome_list, microservice_index)
    return fitness_values
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
    return genptyp_phenotyp_encapsulation(*args)

def evaluate_population(pop, islands):
    with Pool(processes=islands) as pool:
        results = pool.map(encapsulate_genotype_phenotype, chunk_with_service_index(pop, islands))
    fitnesses = [fit for sublist in results for fit in sublist]  # Flatten the list of lists
    return fitnesses

# Define the main algorithm function
def algorithm_weighted(population_num, scaling_factor, phi1, phi2, NGEN, num_days, islands, run_id):
    register_toolbox(scaling_factor, phi1, phi2)
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)
    # Initialize HallOfFame to keep the top 5 individuals
    hof = tools.HallOfFame(20)
    for day in range(num_days):
        day_start = round(time.time() * 1000)

        pop = toolbox.population(population_num)
        best = None
        for g in range(NGEN):

            # Selection and variation operators
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            fitness_values_list = evaluate_population(pop, islands)
            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Assign the returned fitness values to each particle
            for part, fitness_values in zip(pop, fitness_values_list):
                scaled_fit = [fitness_values[0] * 0.6, fitness_values[1] * 0.4]  # Scaling fitness values to match the weights
                part.original_fitness = fitness_values[:2]
                part.original_evaluation = fitness_values[2:]  # save the original evaluation values
                part.fitness.values = scaled_fit
                # Check if the current particle fitness is the local best one
                if not part.best or part.best.fitness.values[0] + part.best.fitness.values[1] < part.fitness.values[0] + part.fitness.values[1]:
                    part.best = creator.Particle(part)
                    part.best.original_fitness = part.original_fitness
                    part.best.original_evaluation = part.original_evaluation
                    part.best.fitness.values = part.fitness.values
                # Check if the current particle is the best in the population
                if not best or best.fitness.values[0] + best.fitness.values[1] < part.fitness.values[0] + part.fitness.values[1]:
                    best = creator.Particle(part)
                    best.original_fitness = part.original_fitness
                    best.original_evaluation = part.original_evaluation
                    best.fitness.values = part.fitness.values
            for part in pop:
                toolbox.update(part, best)
            generation_time_end = round(time.time() * 1000)
            generation_time = generation_time_end - generation_time_start

            # Update the HallOfFame with the current population
            hof.update(pop)
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best.fitness.values])
            gen_logbook.record(day=day, gen=g, RMSD=best.original_evaluation[0],
                               cost=best.original_evaluation[1], RMSD_note=best.original_fitness[0],
                               cost_note=best.original_fitness[1],
                               Weighted_RMSD_note=best.fitness.values[0],
                               Weighted_cost_note=best.fitness.values[1],
                               whole_note=record['fitness']['mean_value'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=population_num)
            record = stats.compile(pop)
            logbook.record(run_id=run_id, day=day, gen=g, nevals=len(pop), beval=best.original_fitness, sum= best.fitness.values[0] + best.fitness.values[1], **record)
            print(logbook.stream)

        # After NGEN generations, select the best individual
        pop.append(best)
        best_ind = selWeighted(pop, 1, False)[0]
        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = genptyp_phenotyp_encapsulation([best_ind], 1)[0]
        best_ind.original_fitness = best_ind_values[:2]
        best_ind.original_evaluation = best_ind_values[2:]
        best_ind.fitness.values = [best_ind.original_fitness[0] * 0.6, best_ind.original_fitness[1] * 0.4]

        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind.fitness.values])
        final_logbook.record(day=day, RMSD=best_ind.original_evaluation[0],
                             cost=best_ind.original_evaluation[1], RMSD_note=best_ind.original_fitness[0],
                             cost_note=best_ind.original_fitness[1], Weighted_RMSD_note=best_ind.fitness.values[0],
                             Weighted_cost_note=best_ind.fitness.values[1], whole_note=record['fitness']['mean_value'],
                             time=day_end - day_start, number_of_ems_evaluations=population_num*NGEN+1)
        gen_logbook.record(day=day, gen=g + 1, RMSD=best_ind.original_evaluation[0],
                           cost=best_ind.original_evaluation[1], RMSD_note=best_ind.original_fitness[0],
                           cost_note=best_ind.original_fitness[1],
                           Weighted_RMSD_note=best_ind.fitness.values[0],
                           Weighted_cost_note=best_ind.fitness.values[1],
                           whole_note=record['fitness']['mean_value'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=population_num)
        # Plot the evolution and pareto front and save them to PNG files
        plot_evolution(gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.png")
        fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
        plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.png")
        print(f"Day {day}, best result: {[best_ind.original_evaluation[0], best_ind.original_evaluation[1]]}")

    global_time_end = round(time.time() * 1000)
    print("Total Time: ", global_time_end - global_time_start, "ms")

    # Save the best individuals from every day to a pickle and txt file
    with open(f'logs/run_{run_id}/evolution_log.pkl', 'ab') as log_file:
        pickle.dump(final_logbook, log_file)
    df_final_logbook = pd.DataFrame(final_logbook)
    df_final_logbook.to_csv(f'logs/run_{run_id}/evolution_log.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/evolution_log_full.pkl', 'wb') as best_ind_file:
        pickle.dump(gen_logbook, best_ind_file)
    df_gen_logbook = pd.DataFrame(gen_logbook)
    df_gen_logbook.to_csv(f'logs/run_{run_id}/evolution_log_full.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/statistics_full.pkl', 'wb') as Statistics_file:
        pickle.dump(logbook, Statistics_file)
    df_logbook = pd.DataFrame(logbook)
    df_logbook.to_csv(f'logs/run_{run_id}/statistics_full.csv', index=False)

    # Save the best individuals from the Hall of Fame
    with open(f'logs/run_{run_id}/hall_of_fame.pkl', 'wb') as hof_file:
        pickle.dump(hof, hof_file)
    df_hof = pd.DataFrame([ind.fitness.values for ind in hof])
    df_hof.to_csv(f'logs/run_{run_id}/hall_of_fame.csv', index=False)
    return best_ind.original_fitness[0], best_ind.original_fitness[1]


def algorithm_unweighted(population_num, scaling_factor, phi1, phi2, NGEN, num_days, islands, run_id):
    register_toolbox(scaling_factor, phi1, phi2)
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)

    # Initialize HallOfFame to keep the top 5 individuals
    hof = tools.HallOfFame(20)
    for day in range(num_days):
        day_start = round(time.time() * 1000)

        pop = toolbox.population(population_num)
        best = None
        for g in range(NGEN):
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            fitness_values_list = evaluate_population(pop, islands)
            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Assign the returned fitness values to each particle
            for part, fitness_values in zip(pop, fitness_values_list):
                part.fitness.values = fitness_values[:2]
                part.original_evaluation = fitness_values[2:]  # save the original evaluation values
                if not part.best or part.best.fitness.values[0] + part.best.fitness.values[1] < part.fitness.values[0] + part.fitness.values[1]:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                    part.best.original_evaluation = part.original_evaluation
                if not best or best.fitness.values[0] + best.fitness.values[1] < part.fitness.values[0] + part.fitness.values[1]:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
                    best.original_evaluation = part.original_evaluation
            for part in pop:
                toolbox.update(part, best)
            generation_time_end = round(time.time() * 1000)
            generation_time = generation_time_end - generation_time_start

            # Update the HallOfFame with the current population
            hof.update(pop)
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best.fitness.values])
            gen_logbook.record(day=day, gen=g,
                               RMSD=best.original_evaluation[0],
                               cost=best.original_evaluation[1],
                               RMSD_note=best.fitness.values[0],
                               cost_note=best.fitness.values[1],
                               whole_note=record['fitness']['mean_value_weighted'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=population_num)
            record = stats.compile(pop)
            logbook.record(run_id=run_id, day=day, gen=g, nevals=len(pop), beval=best.fitness.values, sum= 0.6*best.fitness.values[0] + 0.4* best.fitness.values[1],**record)
            print(logbook.stream)

        # Add best to the population
        pop.append(best)
        best_ind = selWeighted(pop, 1)[0]

        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_values = evaluate_population([best_ind], 1)[0]
        best_ind.fitness.values = best_values[:2]
        best_ind.original_evaluation = best_values[2:]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind.fitness.values])
        final_logbook.record(day=day,
                             RMSD=best_ind.original_evaluation[0],
                             cost=best_ind.original_evaluation[1],
                             RMSD_note=best_ind.fitness.values[0],
                             cost_note=best_ind.fitness.values[1], whole_note=record['fitness']['mean_value_weighted'],
                             time=day_end - day_start, number_of_ems_evaluations=population_num*NGEN+1)
        gen_logbook.record(day=day, gen=g+1,
                           RMSD=best_ind.original_evaluation[0],
                           cost=best_ind.original_evaluation[1],
                           RMSD_note=best_ind.fitness.values[0],
                           cost_note=best_ind.fitness.values[1],
                           whole_note=record['fitness']['mean_value_weighted'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=population_num)

        # Plot the evolution and pareto front and save them to PNG files
        plot_evolution(gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.png")
        fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
        plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.png")
        print(f"Day {day}, best result: {[best_ind.original_evaluation[0], best_ind.original_evaluation[1]]}")

    global_time_end = round(time.time() * 1000)
    print("Total Time: ", global_time_end - global_time_start, "ms")

    # Save the best individuals from every day to a pickle and txt file
    with open(f'logs/run_{run_id}/evolution_log.pkl', 'ab') as log_file:
        pickle.dump(final_logbook, log_file)
    df_final_logbook = pd.DataFrame(final_logbook)
    df_final_logbook.to_csv(f'logs/run_{run_id}/evolution_log.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/evolution_log_full.pkl', 'wb') as best_ind_file:
        pickle.dump(gen_logbook, best_ind_file)
    df_gen_logbook = pd.DataFrame(gen_logbook)
    df_gen_logbook.to_csv(f'logs/run_{run_id}/evolution_log_full.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/statistics_full.pkl', 'wb') as Statistics_file:
        pickle.dump(logbook, Statistics_file)
    df_logbook = pd.DataFrame(logbook)
    df_logbook.to_csv(f'logs/run_{run_id}/statistics_full.csv', index=False)

    # Save the best individuals from the Hall of Fame
    with open(f'logs/run_{run_id}/hall_of_fame.pkl', 'wb') as hof_file:
        pickle.dump(hof, hof_file)
    df_hof = pd.DataFrame([ind.fitness.values for ind in hof])
    df_hof.to_csv(f'logs/run_{run_id}/hall_of_fame.csv', index=False)

    return best_ind.fitness.values

"""
if __name__ == "__main__":
    algorithm_weighted(
        5,
        0.05, 1.0, 1.0, 1000, 1,
        1
    )
"""
