import random
import math
import pickle
import pandas as pd
import numpy
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

# Initialization
toolbox = base.Toolbox()

# Define the attributes for each gene
# toolbox.register("unit_id", random.randint, 1, 3)  # Adjust the range as needed
toolbox.register("unit_id", random.choice, [1, 2, 3, 4, 6])
toolbox.register("start_time", random.randint, 0, 96)  # Adjust the range as needed
toolbox.register("duration", random.randint, 1, 96)  # Adjust the range as needed
toolbox.register("power_fraction_producer", random.uniform, 0.0, 1.0)
toolbox.register("power_fraction_storage", random.uniform, -1.0, 1.0)


# Define a function to create a gene
def create_gene():
    unit_id_value = toolbox.unit_id()
    start_time_value = toolbox.start_time()
    duration_value = toolbox.duration()

    # Check if unit_id is a storage
    if unit_id_value in [3, 4]:
        # If unit_id is 3, set power_fraction between -1.0 and 1.0
        power_fraction_value = toolbox.power_fraction_storage()
    else:
        # If unit_id is not 3, set power_fraction between 0.0 and 1.0
        power_fraction_value = toolbox.power_fraction_producer()

    return unit_id_value, start_time_value, duration_value, power_fraction_value


# Define a function to create a Chromosome with a random number of genes
def create_individual():
    num_genes = random.randint(120, 240)  # Adjust the range as needed
    return [create_gene() for _ in range(num_genes)]


# Create the individual and population and fitness
creator.create("Fitness", base.Fitness, weights=(0.6, 0.4))  # RMSD, Cost
creator.create("Individual", list, fitness=creator.Fitness)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Decorator to check bounds (not needed since EMS also does that)
def checkBounds():
    def decorator(func):
        def wrapper(*args, **kwargs):
            offspring = func(*args, **kwargs)
            for child in offspring:

                for i, gene in enumerate(child):
                    # Assuming gene has the format (unit_id, start_time, duration, power_fraction)

                    unit_id, start_time, duration, power_fraction = gene

                    # Check bounds for each allele in the gene
                    unit_id = max(min(unit_id, 6), 1)  # Bound unit_id between 1 and 3
                    if unit_id == 5:
                        unit_id = random.choice([1, 2, 3, 4, 6])
                    start_time = max(min(start_time, 94), 0)  # Bound start_time between 0 and 94
                    # Check if start_time + duration exceeds 95, then adjust duration
                    if start_time + duration > 96:
                        duration = 96 - start_time

                    # duration = max(min(duration, 95), 1)  # Bound duration between 1 and 95

                    # Check bounds for power_fraction based on unit_id
                    if unit_id in [3, 4]:
                        power_fraction = max(min(power_fraction, 1.0), -1.0)
                    else:
                        power_fraction = max(min(power_fraction, 1.0), 0.0)

                    # Update the gene with the bounded values
                    child[i] = (unit_id, start_time, duration, power_fraction)

            return offspring

        return wrapper

    return decorator


# Encapsulates the main function in GenotypPhenotyp, which converts the gene structure to a schedule readable by EMS
# it also receives the evaluation results from EMS and convert them to exponential notes
def genptyp_phenotyp_encapsulation(chromosome_list, microservice_index):
    # Use the microservice_index if needed for selecting the microservice
    # print(f"Evaluating with microservice {microservice_index}")
    # Call the convert method for evaluation
    fitness_values = GenotypPhenotyp.main(GenotypPhenotyp, chromosome_list, microservice_index)
    return fitness_values


def cxSimulatedBinary(ind1, ind2, eta):
    """Executes a simulated binary crossover

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (gene1, gene2) in enumerate(zip(ind1, ind2)):
        # Unpack genes into their components
        unit_id1, start_time1, duration1, power_fraction1 = gene1
        unit_id2, start_time2, duration2, power_fraction2 = gene2

        # Apply crossover for each component
        for j, (x1, x2) in enumerate(zip([unit_id1, start_time1, duration1, power_fraction1],
                                         [unit_id2, start_time2, duration2, power_fraction2])):
            rand = random.random()
            if rand <= 0.5:
                beta = 2. * rand
            else:
                beta = 1. / (2. * (1. - rand))
            beta **= 1. / (eta + 1.)
            if j == 0:
                ind1[i] = tuple(list(ind1[i][:j]) + [x1] + list(ind1[i][j + 1:]))
                ind2[i] = tuple(list(ind2[i][:j]) + [x2] + list(ind2[i][j + 1:]))
            elif j in [1, 2]:  # Use uniform crossover for unit_id, start_time, duration
                ind1[i] = tuple(
                    list(ind1[i][:j]) + [math.floor(0.5 * (((1 + beta) * x1) + ((1 - beta) * x2)))] + list(
                        ind1[i][j + 1:]))
                ind2[i] = tuple(
                    list(ind2[i][:j]) + [math.floor(0.5 * (((1 - beta) * x1) + ((1 + beta) * x2)))] + list(
                        ind2[i][j + 1:]))
            else:
                ind1[i] = tuple(
                    list(ind1[i][:j]) + [0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))] + list(ind1[i][j + 1:]))
                ind2[i] = tuple(
                    list(ind2[i][:j]) + [0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))] + list(ind2[i][j + 1:]))
            # Use simulated binary crossover for power_fraction

    return ind1, ind2


def mutPolynomialGene(individual, eta, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: Individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)

    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)
            if i == 0:
                # Leave the individual[i] as it is
                pass
            else:
                if rand < 0.5:
                    delta_q = (2.0 * rand) ** mut_pow - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - rand)) ** mut_pow

                x = x + delta_q
                if i == 3:
                    individual[i] = x
                else:
                    individual[i] = int(x)

    return tuple(individual),


def mutGaussianPolynomialGene(individual, mu, sigma, eta, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.

    :param individual: Individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)

    for i in range(size):
        if random.random() <= indpb:
            x = individual[i]
            rand = random.random()
            mut_pow = 1.0 / (eta + 1.)

            if i == 0:
                # Leave the individual[i] as it is
                pass
            elif i == 1 or i == 2:
                # Perform Gaussian mutation on individual[i]
                # individual[i] = random.randint(0, 96)
                individual[i] = max(0, min(int(individual[i] + random.gauss(mu, sigma)), 96))
            else:
                # Perform polynomial mutation on individual[i]
                if rand < 0.5:
                    delta_q = (2.0 * rand) ** mut_pow - 1.0
                else:
                    delta_q = 1.0 - (2.0 * (1.0 - rand)) ** mut_pow

                x = x + delta_q
                individual[i] = x

    return tuple(individual),


def mutPolynomialIndividual(individual, eta, indpb):
    """Polynomial mutation applied to the whole individual

    :param individual: Tuple gene to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    for i, gene in enumerate(individual):
        list_gene = list(gene)
        individual[i], = mutPolynomialGene(list_gene, eta=eta, indpb=indpb)
    return individual,


def mutGaussianPolynomialIndividual(individual, mu, sigma, eta, indpb):
    """Polynomial mutation applied to the whole individual

    :param individual: Tuple gene to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    for i, gene in enumerate(individual):
        list_gene = list(gene)
        individual[i], = mutGaussianPolynomialGene(list_gene, mu=mu, sigma=sigma, eta=eta, indpb=indpb)
    return individual,


def cxUniformSimulatedBinary(ind1, ind2, eta):
    """Executes a uniform crossover for the integer part and simulated binary crossover
    for the float part.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param eta: Crowding degree of the crossover. A high eta will produce
                children resembling to their parents, while a small eta will
                produce solutions much more different.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    for i, (gene1, gene2) in enumerate(zip(ind1, ind2)):
        # Unpack genes into their components
        unit_id1, start_time1, duration1, power_fraction1 = gene1
        unit_id2, start_time2, duration2, power_fraction2 = gene2

        # Apply crossover for each component
        for j, (x1, x2) in enumerate(zip([unit_id1, start_time1, duration1, power_fraction1],
                                         [unit_id2, start_time2, duration2, power_fraction2])):
            rand = random.random()

            if j in [1, 2]:  # Use uniform crossover for unit_id, start_time, duration
                if rand <= 0.5:
                    # We assign the whole tuple itself, instead of changing elements inside tuple,
                    # because of the immutable nature of tuples
                    ind1[i] = tuple(list(ind1[i][:j]) + [x1] + list(ind1[i][j + 1:]))
                    ind2[i] = tuple(list(ind2[i][:j]) + [x2] + list(ind2[i][j + 1:]))
                else:
                    ind1[i] = tuple(list(ind1[i][:j]) + [x2] + list(ind1[i][j + 1:]))
                    ind2[i] = tuple(list(ind2[i][:j]) + [x1] + list(ind2[i][j + 1:]))
            elif j == 0:
                ind1[i] = tuple(list(ind1[i][:j]) + [x1] + list(ind1[i][j + 1:]))
                ind2[i] = tuple(list(ind2[i][:j]) + [x2] + list(ind2[i][j + 1:]))
            else:
                if rand <= 0.5:
                    beta = 2. * rand
                else:
                    beta = 1. / (2. * (1. - rand))
                beta **= 1. / (eta + 1.)

                # Use simulated binary crossover for power_fraction
                ind1[i] = tuple(
                    list(ind1[i][:j]) + [0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))] + list(ind1[i][j + 1:]))
                ind2[i] = tuple(
                    list(ind2[i][:j]) + [0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))] + list(ind2[i][j + 1:]))

    return ind1, ind2


def selTournamentDCD(individuals, k=None):
    # https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L119
    tools.emo.assignCrowdingDist(individuals)
    # https://github.com/DEAP/deap/blob/master/deap/tools/emo.py#L145
    return tools.selTournamentDCD(individuals, k)



# Operators
# toolbox.register("mate", cxSimulatedBinary)
toolbox.register("mate", cxUniformSimulatedBinary)
#toolbox.register("mutate", mutPolynomialIndividual)
toolbox.register("mutate", mutGaussianPolynomialIndividual)
toolbox.register("select", tools.selNSGA2)

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


# Decorators

toolbox.decorate("mate", checkBounds())
toolbox.decorate("mutate", checkBounds())

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


# Initialize the Pareto front
pareto_front = tools.ParetoFront()


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


# Plot the pareto front at the end of the evaluation
def plot_pareto_front(pareto_front, day, filename):
    plt.figure(figsize=(10, 6))
    for i, front in enumerate(pareto_front):
        # Extract the fitness values for each individual in the Pareto front
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

def algorithm_weighted(population_num, CXPB, eta ,indpb, mu, sigma, NGEN, num_days, islands, run_id):
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)
    for day in range(num_days):
        day_start = round(time.time() * 1000)
        invalid_ind_length_full = 0
        pop = toolbox.population(population_num)
        evaluation_time_one_start = round(time.time() * 1000)
        # Evaluate the entire population in parallel
        fitnesses = evaluate_population(pop, islands)
        for ind, fit in zip(pop, fitnesses):
            scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]  # Scaling fitness values to match the weights
            ind.original_fitness = fit[:2]  # keep the original fitness
            ind.original_evaluation = fit[2:]  # save the original evaluation values
            ind.fitness.values = scaled_fit
        evaluation_time_one_end = round(time.time() * 1000)
        evaluation_time_one = evaluation_time_one_end - evaluation_time_one_start

        pop = toolbox.select(pop, len(pop))  # used to assign nondominated ranks and crowding distances
        best_ind_gen = selWeighted(pop, 1, False)[0]
        generation_one_end = round(time.time() * 1000)
        generation_one = generation_one_end - day_start
        record = mstats.compile([best_ind_gen.fitness.values])
        gen_logbook.record(day=day, gen=0, RMSD=best_ind_gen.original_evaluation[0],
                           cost=best_ind_gen.original_evaluation[1], RMSD_note=best_ind_gen.original_fitness[0],
                           cost_note=best_ind_gen.original_fitness[1],
                           Weighted_RMSD_note=best_ind_gen.fitness.values[0],
                           Weighted_cost_note=best_ind_gen.fitness.values[1],
                           whole_note=record['fitness']['mean_value'], generation_time=generation_one - evaluation_time_one,
                           evaluation_time=evaluation_time_one, number_of_ems_evaluations=population_num)

        for g in range(NGEN - 1):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Select the next generation individuals
            offspring = tools.selTournamentDCD(pop, len(pop))
            # Clone the selected individuals

            offspring = list(map(toolbox.clone, offspring))

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2, eta=eta)

                toolbox.mutate(ind1, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                toolbox.mutate(ind2, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                del ind1.fitness.values, ind2.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind_length = len(invalid_ind)
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start
            # Evaluate the individuals with an invalid fitness in parallel
            fitnesses = evaluate_population(invalid_ind, islands)
            for ind, fit in zip(invalid_ind, fitnesses):
                scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]
                ind.original_fitness = fit[:2]  # keep the original fitness
                ind.original_evaluation = fit[2:]  # save the original evaluation values
                ind.fitness.values = scaled_fit
            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            # The population is then selected from the offspring + population to ensure elitism
            pop = toolbox.select(pop + offspring, population_num)
            best_ind_gen = selWeighted(pop, 1, False)[0]
            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2
            invalid_ind_length_full += invalid_ind_length
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best_ind_gen.fitness.values])
            gen_logbook.record(day=day, gen=g+1, RMSD=best_ind_gen.original_evaluation[0],
                               cost=best_ind_gen.original_evaluation[1], RMSD_note=best_ind_gen.original_fitness[0],
                               cost_note=best_ind_gen.original_fitness[1],
                               Weighted_RMSD_note=best_ind_gen.fitness.values[0],
                               Weighted_cost_note=best_ind_gen.fitness.values[1],
                               whole_note=record['fitness']['mean_value'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=invalid_ind_length)
            record = stats.compile(invalid_ind)
            logbook.record(run_id=run_id, day=day, gen=g, nevals=len(invalid_ind), beval=best_ind_gen.original_fitness,sum= best_ind_gen.fitness.values[0] + best_ind_gen.fitness.values[1], **record)
            print(logbook.stream)
        # Update the Pareto front with the offspring
        pareto_front.update(offspring)
        # After NGEN generations, select the best individual
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
        # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
        print("Best individual is %s, %s" % (best_ind.original_fitness))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        print(best_ind_values, best_ind.original_fitness, best_ind.original_evaluation, best_ind.fitness.values)
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind.fitness.values])
        final_logbook.record(day=day, RMSD=best_ind.original_evaluation[0],
                             cost=best_ind.original_evaluation[1], RMSD_note=best_ind.original_fitness[0],
                             cost_note=best_ind.original_fitness[1], Weighted_RMSD_note=best_ind.fitness.values[0],
                             Weighted_cost_note=best_ind.fitness.values[1], whole_note=record['fitness']['mean_value'],
                             time=day_end - day_start, number_of_ems_evaluations=invalid_ind_length_full)
        gen_logbook.record(day=day, gen=g + 2, RMSD=best_ind_gen.original_evaluation[0],
                           cost=best_ind_gen.original_evaluation[1], RMSD_note=best_ind_gen.original_fitness[0],
                           cost_note=best_ind_gen.original_fitness[1],
                           Weighted_RMSD_note=best_ind_gen.fitness.values[0],
                           Weighted_cost_note=best_ind_gen.fitness.values[1],
                           whole_note=record['fitness']['mean_value'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=invalid_ind_length)

        # Plot the evolution and pareto front and save them to PNG files
        plot_evolution(gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.png")
        plot_pareto_front([pareto_front], day, f'logs/run_{run_id}/plots/pareto_front_day_{day}.png')
        fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
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

    # Save the Pareto front to a pickle file
    with open(f'logs/run_{run_id}/pareto_front.pkl', 'wb') as pareto_file:
        # Create a list of tuples where each tuple is (individual, fitness_values)
        pareto_with_fitness = [(ind, ind.fitness.values) for ind in pareto_front]
        pickle.dump(pareto_with_fitness, pareto_file)
    df_pareto_front = pd.DataFrame(pareto_with_fitness)
    df_pareto_front.to_csv(f'logs/run_{run_id}/pareto_front.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/statistics_full.pkl', 'wb') as Statistics_file:
        pickle.dump(logbook, Statistics_file)
    df_logbook = pd.DataFrame(logbook)
    df_logbook.to_csv(f'logs/run_{run_id}/statistics_full.csv', index=False)
    return best_ind.original_fitness[0], best_ind.original_fitness[1]


def algorithm_unweighted(population_num, CXPB, eta ,indpb, mu, sigma, NGEN, num_days, islands,run_id):
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)
    for day in range(num_days):
        day_start = round(time.time() * 1000)
        invalid_ind_length_full = 0
        pop = toolbox.population(population_num)
        evaluation_time_one_start = round(time.time() * 1000)
        # Evaluate the entire population in parallel
        fitnesses = evaluate_population(pop, islands)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit[:2]
            ind.original_evaluation = fit[2:]  # save the original evaluation values
        evaluation_time_one_end = round(time.time() * 1000)
        evaluation_time_one = evaluation_time_one_end - evaluation_time_one_start


        pop = toolbox.select(pop, len(pop))  # used to assign nondominated ranks and crowding distances
        best_ind_gen = selWeighted(pop, 1)[0]
        generation_one_end = round(time.time() * 1000)
        generation_one = generation_one_end - day_start
        record = mstats.compile([best_ind_gen.fitness.values])
        gen_logbook.record(day=day, gen=0, RMSD=best_ind_gen.original_evaluation[0],
                           cost=best_ind_gen.original_evaluation[1], RMSD_note=best_ind_gen.fitness.values[0],
                           cost_note=best_ind_gen.fitness.values[1],
                           whole_note=record['fitness']['mean_value'], generation_time=generation_one - evaluation_time_one,
                           evaluation_time=evaluation_time_one, number_of_ems_evaluations=population_num)
        for g in range(NGEN - 1):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Select the next generation individuals
            offspring = tools.selTournamentDCD(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2, eta=eta)

                toolbox.mutate(ind1, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                toolbox.mutate(ind2, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                del ind1.fitness.values, ind2.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind_length = len(invalid_ind)
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start

            # Evaluate the individuals with an invalid fitness in parallel
            fitnesses = evaluate_population(invalid_ind, islands)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit[:2]
                ind.original_evaluation = fit[2:]  # save the original evaluation values
            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            # The population is then selected from the offspring + population to ensure elitism
            pop = toolbox.select(pop + offspring, population_num)

            best_ind_gen = selWeighted(pop, 1)[0]
            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2

            invalid_ind_length_full += invalid_ind_length
            # Update the statistics and logbook for the current generation
            record = mstats.compile([best_ind_gen.fitness.values])
            gen_logbook.record(day=day,gen=g+1,
                               RMSD=best_ind_gen.original_evaluation[0],
                               cost=best_ind_gen.original_evaluation[1],
                               RMSD_note=best_ind_gen.fitness.values[0],
                               cost_note=best_ind_gen.fitness.values[1],
                               whole_note=record['fitness']['mean_value_weighted'], generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=invalid_ind_length)
            record = stats.compile(invalid_ind)
            logbook.record(run_id=run_id, day=day, gen=g, nevals=len(invalid_ind), beval=best_ind_gen.fitness.values, sum= 0.6*best_ind_gen.fitness.values[0] + 0.4* best_ind_gen.fitness.values[1], **record)
            print(logbook.stream)

        # Update the Pareto front with the offspring
        pareto_front.update(offspring)

        # After NGEN generations, select the best individual
        best_ind = selWeighted(pop, 1)[0]
        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = genptyp_phenotyp_encapsulation([best_ind], 1)[0]
        best_ind.fitness.values = best_ind_values[:2]
        best_ind.original_evaluation = best_ind_values[2:]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        # print "Best individual is %s, %s" % (hof[0], hof[0].fitness.values)
        print("Best individual is %s, %s" % (best_ind.fitness.values))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = mstats.compile([best_ind.fitness.values])
        final_logbook.record(day=day,
                             RMSD=best_ind.original_evaluation[0],
                             cost=best_ind.original_evaluation[1],
                             RMSD_note=best_ind.fitness.values[0],
                             cost_note=best_ind.fitness.values[1], whole_note=record['fitness']['mean_value_weighted'],
                             time=day_end - day_start, number_of_ems_evaluations=invalid_ind_length_full)
        gen_logbook.record(day=day, gen=g + 2,
                           RMSD=best_ind_gen.original_evaluation[0],
                           cost=best_ind_gen.original_evaluation[1],
                           RMSD_note=best_ind_gen.fitness.values[0],
                           cost_note=best_ind_gen.fitness.values[1],
                           whole_note=record['fitness']['mean_value_weighted'], generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=invalid_ind_length)

        # Plot the evolution and pareto front and save them to PNG files
        plot_evolution(gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.png")
        plot_pareto_front([pareto_front], day, f'logs/run_{run_id}/plots/pareto_front_day_{day}.png')
        fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
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

    # Save the Pareto front to a pickle file
    with open(f'logs/run_{run_id}/pareto_front.pkl', 'wb') as pareto_file:
        # Create a list of tuples where each tuple is (individual, fitness_values)
        pareto_with_fitness = [(ind, ind.fitness.values) for ind in pareto_front]
        pickle.dump(pareto_with_fitness, pareto_file)
    df_pareto_front = pd.DataFrame(pareto_with_fitness)
    df_pareto_front.to_csv(f'logs/run_{run_id}/pareto_front.csv', index=False)

    # Save the best individuals in every generation list to a separate pickle and txt file
    with open(f'logs/run_{run_id}/statistics_full.pkl', 'wb') as Statistics_file:
        pickle.dump(logbook, Statistics_file)
    df_logbook = pd.DataFrame(logbook)
    df_logbook.to_csv(f'logs/run_{run_id}/statistics_full.csv', index=False)
    return best_ind.fitness.values
"""
if __name__ == "__main__":
    algorithm_unweighted(
        100,
        0.9, 50, 0.05, 5, 50,
        50,
        1, 1
    )
"""