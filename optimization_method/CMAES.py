import numpy
from deap import benchmarks
from Strategy import Strategy
import pandas as pd
import pickle
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

# The cma module uses the numpy random number generator
numpy.random.seed(128)

creator.create("FitnessMul", base.Fitness, weights=(0.6, 0.4))
creator.create("IndividualFlat", list, fitness=creator.FitnessMul)

toolbox = base.Toolbox()
toolbox.register("evaluate", benchmarks.rastrigin)

def map_values(individual):
    # Define the mapping ranges
    unit_id = [1, 2, 3, 4, 6]
    start_time = list(range(0, 95))
    duration = list(range(1, 96))
    power_fraction = numpy.linspace(0, 1, 100)

    # Initialize the result list
    result = []

    # Process each chunk of 4 values
    for i in range(0, len(individual), 4):
        chunk = (
            unit_id[int(individual[i] % len(unit_id))],
            start_time[int(individual[i+1] % len(start_time))],
            duration[int(individual[i+2] % len(duration))],
            power_fraction[int(individual[i+3] % len(power_fraction))]
        )
        result.append(chunk)

    return result



def selWeighted(individuals, k, use_weights=True):
    """Select the k best individuals among the input individuals.

    :param individuals: A list of individuals to select from.
    :param k: The number of individuals to select.
    :param use_weights: Boolean parameter indicating whether to use weighted sorting.
    :returns: A list of selected individuals.
    """
    if use_weights:
        sorted_individuals = sorted(individuals,
                                    key=lambda ind: (0.6 * ind.fitness.values[0]) + (0.4 * ind.fitness.values[1]),
                                    reverse=True)
    else:
        sorted_individuals = sorted(individuals, key=lambda ind: (ind.fitness.values[0]) + (ind.fitness.values[1]),
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
gen_logbook.header = "gen", "day", "RMSD", "cost", "RMSD_note", "cost_note", "Weighted_RMSD_note", "Weighted_cost_note", "whole_note", "generation_time", "evaluation_time", "number_of_ems_evaluations"

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", numpy.mean)
stats.register("std", numpy.std)
stats.register("min", numpy.min)
stats.register("max", numpy.max)
logbook = tools.Logbook()
logbook.header = ['run_id', 'gen', 'nevals', 'beval', 'sum'] + stats.fields


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

# Register the toolbox functions with parameters for smin and smax
def register_toolbox(N, centroid, sigma, lambda_):
    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    strategy = Strategy(centroid=[centroid]*N, sigma=sigma, lambda_=int(lambda_/4*N))
    toolbox.register("generate", strategy.generate, creator.IndividualFlat)
    toolbox.register("update", strategy.update)
    toolbox.register("map_values", map_values)

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
    # print(f"Evaluating with microservice {microservice_index}")
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


def algorithm_unweighted(lambda_, N, centroid, sigma, NGEN, num_days, islands, run_id):
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    register_toolbox(N, centroid, sigma, lambda_)
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)
    # Initialize HallOfFame to keep the top 5 individuals
    hof = tools.HallOfFame(20)
    for day in range(num_days):
        day_start = round(time.time() * 1000)

        for g in range(NGEN):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Generate a new population
            population = toolbox.generate()

            # change the population shape for the map_values function
            population_mapped = [toolbox.map_values(ind) for ind in population]

            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start

            # Evaluate the individuals
            fitnesses = evaluate_population(population_mapped, islands)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit[:2]
                ind.original_evaluation = fit[2:]  # save the original evaluation values

            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            best_ind_gen_flat = selWeighted(population, 1)[0]

            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2
            # Update the HallOfFame with the current population
            hof.update(population)
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best_ind_gen_flat.fitness.values])
            gen_logbook.record(day=day,gen=g+1,
                               RMSD=best_ind_gen_flat.original_evaluation[0],
                               cost=best_ind_gen_flat.original_evaluation[1],
                               RMSD_note=best_ind_gen_flat.fitness.values[0],
                               cost_note=best_ind_gen_flat.fitness.values[1],
                               whole_note=record['fitness']['mean_value_weighted'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=len(population))
            record = stats.compile(population)
            logbook.record(run_id=run_id, gen=g, nevals=len(population), beval=best_ind_gen_flat.fitness.values, sum= 0.6 *best_ind_gen_flat.fitness.values[0] + 0.4* best_ind_gen_flat.fitness.values[1],**record)
            print(logbook.stream)

        # After NGEN generations, select the best individual
        best_ind_flat = selWeighted(population, 1)[0]
        best_ind = toolbox.map_values(best_ind_flat)
        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = genptyp_phenotyp_encapsulation([best_ind], 1)[0]
        best_ind_flat.fitness.values = best_ind_values[:2]
        best_ind_flat.original_evaluation = best_ind_values[2:]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
        print("Best individual is %s, %s" % (best_ind_flat.fitness.values))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind.fitness.values])
        final_logbook.record(day=day,
                             RMSD=best_ind_flat.original_evaluation[0],
                             cost=best_ind_flat.original_evaluation[1],
                             RMSD_note=best_ind_flat.fitness.values[0],
                             cost_note=best_ind_flat.fitness.values[1], whole_note=record['fitness']['mean_value_weighted'],
                             time=day_end - day_start, number_of_ems_evaluations=len(population)*N+1)
        gen_logbook.record(day=day, gen=g + 2,
                           RMSD=best_ind_flat.original_evaluation[0],
                           cost=best_ind_flat.original_evaluation[1],
                           RMSD_note=best_ind_flat.fitness.values[0],
                           cost_note=best_ind_flat.fitness.values[1],
                           whole_note=record['fitness']['mean_value_weighted'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=len(population))

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


def algorithm_weighted(lambda_, N, centroid, sigma, NGEN, num_days, islands, run_id):
    global_time_start = round(time.time() * 1000)
    register_toolbox(N, centroid, sigma, lambda_)
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

        for g in range(NGEN):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Generate a new population
            population = toolbox.generate()

            # change the population shape for the map_values function
            population_mapped = [toolbox.map_values(ind) for ind in population]
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start

            # Evaluate the individuals
            fitnesses = evaluate_population(population_mapped, islands)
            for ind, fit in zip(population, fitnesses):
                scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]
                ind.original_fitness = fit[:2]  # keep the original fitness
                ind.original_evaluation = fit[2:]  # save the original evaluation values
                ind.fitness.values = scaled_fit

            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            best_ind_gen_flat = selWeighted(population, 1)[0]

            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2
            # Update the HallOfFame with the current population
            hof.update(population)
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best_ind_gen_flat.fitness.values])
            gen_logbook.record(day=day,gen=g+1,
                               RMSD=best_ind_gen_flat.original_evaluation[0],
                               cost=best_ind_gen_flat.original_evaluation[1], RMSD_note=best_ind_gen_flat.original_fitness[0],
                               cost_note=best_ind_gen_flat.original_fitness[1],
                               Weighted_RMSD_note=best_ind_gen_flat.fitness.values[0],
                               Weighted_cost_note=best_ind_gen_flat.fitness.values[1],
                               whole_note=record['fitness']['mean_value'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=len(population))

            record = stats.compile(population)
            logbook.record(run_id=run_id, gen=g, nevals=len(population), beval=best_ind_gen_flat.original_fitness, sum= best_ind_gen_flat.fitness.values[0] + best_ind_gen_flat.fitness.values[1], **record)
            print(logbook.stream)

        # After NGEN generations, select the best individual
        best_ind_flat = selWeighted(population, 1)[0]
        best_ind = toolbox.map_values(best_ind_flat)
        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = genptyp_phenotyp_encapsulation([best_ind], 1)[0]
        best_ind_flat.original_fitness = best_ind_values[:2]
        best_ind_flat.original_evaluation = best_ind_values[2:]
        best_ind_flat.fitness.values = [best_ind_flat.original_fitness[0] * 0.6, best_ind_flat.original_fitness[1] * 0.4]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
        print("Best individual is %s, %s" % (best_ind_flat.original_fitness))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind_flat.fitness.values])
        final_logbook.record(day=day,
                             RMSD=best_ind_flat.original_evaluation[0],
                             cost=best_ind_flat.original_evaluation[1],
                             RMSD_note=best_ind_flat.original_fitness[0],
                             cost_note=best_ind_flat.original_fitness[1], Weighted_RMSD_note=best_ind_flat.fitness.values[0],
                             Weighted_cost_note=best_ind_flat.fitness.values[1], whole_note=record['fitness']['mean_value'],
                             time=day_end - day_start, number_of_ems_evaluations=len(population)*N+1)
        gen_logbook.record(day=day, gen=g + 1,
                           RMSD=best_ind_flat.original_evaluation[0],
                           cost=best_ind_flat.original_evaluation[1], RMSD_note=best_ind_flat.original_fitness[0],
                           cost_note=best_ind_flat.original_fitness[1],
                           Weighted_RMSD_note=best_ind_flat.fitness.values[0],
                           Weighted_cost_note=best_ind_flat.fitness.values[1],
                           whole_note=record['fitness']['mean_value'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=len(population))

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
    return best_ind_flat.original_fitness[0], best_ind_flat.original_fitness[1]


"""
if __name__ == "__main__":
    algorithm_weighted(0.56,720, 5.0, 5.0, 250, 1, 1)
"""