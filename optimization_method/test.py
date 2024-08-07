import numpy
from deap import base
from deap import benchmarks
from Strategy import Strategy
from deap import creator
from deap import tools
from multiprocessing import Pool
import random
import math
import pandas as pd
import pickle
from deap import base, creator, tools
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time
from genotyp_phenotyp import GenotypPhenotyp

# The cma module uses the numpy random number generator
numpy.random.seed(128)

creator.create("Fitness", base.Fitness, weights=(0.6, 0.4))
creator.create("Individual", list, fitness=creator.Fitness)

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
logbook.header = ['gen', 'nevals', 'beval'] + stats.fields


# Plot the evolution of every function value over the generations
def plot_evolution(gen_logbook, day, filename):
    plt.clf()  # Clear the current figure
    day_data = [record for record in gen_logbook if record['day'] == day]
    generations = [record['gen'] for record in day_data]
    cost_values = [record['cost_note'] for record in day_data]
    RMSD_values = [record['RMSD_note'] for record in day_data]

    plt.figure(figsize=(10, 6))

    # Plot Cost
    plt.plot(generations, cost_values, label="Cost")

    # Plot RMSD
    plt.plot(generations, RMSD_values, label="RMSD")

    plt.title(f"Evolution of Parameters over Generations (Day {day})")
    plt.xlabel("Generation")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)  # Save the plot to a file
    plt.show()


# Register the toolbox functions with parameters for smin and smax
def register_toolbox(N, centroid, sigma, lambda_):
    # The CMA-ES algorithm takes a population of one individual as argument
    # The centroid is set to a vector of 5.0 see http://www.lri.fr/~hansen/cmaes_inmatlab.html
    # for more details about the rastrigin and other tests for CMA-ES
    strategy = Strategy(centroid=[centroid]*N, sigma=sigma, lambda_=int(lambda_/4*N))
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)
    toolbox.register("map_values", map_values)

# Encapsulates the main function in GenotypPhenotyp, which converts the gene structure to a schedule readable by EMS
# it also receives the evaluation results from EMS and convert them to exponential notes
def genptyp_phenotyp_encapsulation(chromosome_list, microservice_index):
    # Use the microservice_index if needed for selecting the microservice
    # print(f"Evaluating with microservice {microservice_index}")
    # Call the convert method for evaluation
    fitness_values = GenotypPhenotyp.main_old(GenotypPhenotyp, chromosome_list, microservice_index)
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

def algorithm_unweighted(NGEN, N, centroid, sigma, lambda_):
    register_toolbox(N, centroid, sigma, lambda_)

    hof = tools.HallOfFame(1)


    for gen in range(NGEN):
        # Generate a new population
        population = toolbox.generate()
        # change the population shape for the map_values function
        population_mapped = [toolbox.map_values(ind) for ind in population]
        # Evaluate the individuals
        fitnesses = evaluate_population(population_mapped, 1)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit[:2]
            ind.original_evaluation = fit[2:]  # save the original evaluation values

        # Update the hall of fame with the generated individuals
        hof.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(population), beval= hof[0].fitness.values[0] ,**record)
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
    print("Best individual is %s, %s" % (best_ind, best_ind_flat.fitness.values))
    return hof[0].fitness.values[0]


if __name__ == "__main__":
    algorithm_unweighted(250,32, 0.5, 0.5, 20)