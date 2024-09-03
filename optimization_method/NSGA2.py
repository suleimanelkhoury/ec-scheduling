import random
import math
from deap import base, creator, tools
import time
import redis
import os
import ea_utils

# =========================
# SECTION 1: Initialize Redis, configure DEAP's Toolbox, and define bounds for the genes
# =========================

# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379)

# Initialization
toolbox = base.Toolbox()

# Define the attributes for each gene
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


# Define a function to create an individual with a random number of genes
def create_individual():
    num_genes = random.randint(120, 240)  # Adjust the range as needed
    return [create_gene() for _ in range(num_genes)]


# Create the fitness and individual classes, and register them with the toolbox
creator.create("Fitness", base.Fitness, weights=(0.6, 0.4))  # RMSD, Cost
creator.create("Individual", list, fitness=creator.Fitness)
toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Decorator to check bounds of the alleles in the genes in the individual
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


# =========================
# SECTION 2: Define the crossover and mutation operators and register them with the toolbox
# =========================

def cxSimulatedBinary(ind1, ind2, eta):
    """Executes a simulated binary crossover for the float and int part of the individuals, where the int part is truncated

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
    C by Deb, implemented for a gene.

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
    C by Deb, implemented for the float part of the gene, while Gaussian mutation is applied to the integer part.

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
    """Polynomial mutation applied to the whole individual.

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
    """Polynomial and Gaussian mutation applied to the whole individual.

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


# Operators
toolbox.register("mate", cxUniformSimulatedBinary)
toolbox.register("mutate", mutGaussianPolynomialIndividual)
toolbox.register("select", tools.selNSGA2)
# Decorators
toolbox.decorate("mate", checkBounds())
toolbox.decorate("mutate", checkBounds())


# =========================
# SECTION 3: Algorithm's main loop
# =========================

def algorithm(population_num, CXPB, eta, indpb, mu, sigma, NGEN, num_days, islands, run_id, weighted=False):
    """
    Execute the main algorithm loop, handling both weighted and unweighted versions based on the `weighted` flag.

    Parameters:
    - population_num (int): Population size.
    - CXPB (float): Crossover probability.
    - eta (float): Crowding degree of the mutation or crossover.
    - indpb (float): Independent probability for each attribute to be mutated.
    - mu (float): Mean for the mutation's Gaussian distribution.
    - sigma (float): Standard deviation for the mutation's Gaussian distribution.
    - NGEN (int): Number of generations to run for each day.
    - num_days (int): Total number of days to run the evolution.
    - islands (int): Number of islands used for evaluation.
    - run_id (str): Identifier for the run, used for logging purposes.
    - weighted (bool): If True, apply weights to the fitness values during evolution; otherwise, use unweighted values.

    Returns:
    - best_ind_flat.fitness.values (list): The fitness values of the best individual.
    """

    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)

    for day in range(num_days):
        day_start = round(time.time() * 1000)
        # Parameter used to calculate the total number of evaluations over the whole day
        invalid_ind_length_full = population_num
        # Initialize the population
        pop = toolbox.population(population_num)
        evaluation_time_one_start = round(time.time() * 1000)

        # Initialize the Pareto front
        pareto_front = tools.ParetoFront()
        # Evaluate the entire population in parallel
        fitnesses = ea_utils.evaluate_population(pop, islands)

        for ind, fit in zip(pop, fitnesses):
            if weighted:
                scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]
                ind.original_fitness = fit[:2]  # keep the original fitness
                ind.fitness.values = scaled_fit
            else:
                ind.fitness.values = fit[:2]
            ind.original_evaluation = fit[2:]  # save the original evaluation values

        evaluation_time_one_end = round(time.time() * 1000)
        evaluation_time_one = evaluation_time_one_end - evaluation_time_one_start

        pop = toolbox.select(pop, len(pop))  # used to assign nondominated ranks and crowding distances
        # Select the best individual from the population
        best_ind_gen = ea_utils.selWeighted(pop, 1, False)[0]

        generation_one_end = round(time.time() * 1000)
        generation_one = generation_one_end - day_start

        # Log the best individual from the first generation
        record = ea_utils.mstats.compile([best_ind_gen.fitness.values])
        ea_utils.gen_logbook.record(day=day, gen=0, RMSD=best_ind_gen.original_evaluation[0],
                                    cost=best_ind_gen.original_evaluation[1],
                                    RMSD_note=best_ind_gen.original_fitness[0] if weighted else
                                    best_ind_gen.fitness.values[0],
                                    cost_note=best_ind_gen.original_fitness[1] if weighted else
                                    best_ind_gen.fitness.values[1],
                                    Weighted_RMSD_note=best_ind_gen.fitness.values[0] if weighted else None,
                                    Weighted_cost_note=best_ind_gen.fitness.values[1] if weighted else None,
                                    whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                        'mean_value_weighted'],
                                    generation_time=generation_one - evaluation_time_one,
                                    evaluation_time=evaluation_time_one, number_of_ems_evaluations=population_num)
        # After evaluating the first generation, proceed with the rest
        for g in range(NGEN - 1):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Select the next generation individuals
            offspring = tools.selTournamentDCD(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Perform crossover and mutation on the offspring
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= CXPB:
                    toolbox.mate(ind1, ind2, eta=eta)

                toolbox.mutate(ind1, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                toolbox.mutate(ind2, eta=eta, indpb=indpb, mu=mu, sigma=sigma)
                del ind1.fitness.values, ind2.fitness.values

            # Invalidate the individuals that have been newly generated using the operators
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            invalid_ind_length = len(invalid_ind)

            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start

            # Evaluate the individuals with an invalid fitness in parallel
            fitnesses = ea_utils.evaluate_population(invalid_ind, islands)
            for ind, fit in zip(invalid_ind, fitnesses):
                if weighted:
                    scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]
                    ind.original_fitness = fit[:2]  # keep the original fitness
                    ind.fitness.values = scaled_fit
                else:
                    ind.fitness.values = fit[:2]
                ind.original_evaluation = fit[2:]  # save the original evaluation values

            # Log the time used in the evaluation
            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            # The population is then selected from the offspring + population to ensure elitism
            pop = toolbox.select(pop + offspring, population_num)

            # Select the best individual from the population
            if weighted:
                best_ind_gen = ea_utils.selWeighted(pop, 1, False)[0]
            else:
                best_ind_gen = ea_utils.selWeighted(pop, 1, True)[0]

            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2

            # Parameter used to calculate the total number of evaluations over the whole day
            invalid_ind_length_full += invalid_ind_length

            # Update the statistics and logbook for the current generation
            record = ea_utils.mstats.compile([best_ind_gen.fitness.values])
            ea_utils.gen_logbook.record(day=day, gen=g + 1,
                                        RMSD=best_ind_gen.original_evaluation[0],
                                        cost=best_ind_gen.original_evaluation[1],
                                        RMSD_note=best_ind_gen.original_fitness[0] if weighted else
                                        best_ind_gen.fitness.values[0],
                                        cost_note=best_ind_gen.original_fitness[1] if weighted else
                                        best_ind_gen.fitness.values[1],
                                        Weighted_RMSD_note=best_ind_gen.fitness.values[0] if weighted else None,
                                        Weighted_cost_note=best_ind_gen.fitness.values[1] if weighted else None,
                                        whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                            'mean_value_weighted'],
                                        generation_time=generation_time, evaluation_time=evaluation_time,
                                        number_of_ems_evaluations=invalid_ind_length)

            record = ea_utils.stats.compile(invalid_ind)
            ea_utils.logbook.record(run_id=run_id, day=day, gen=g, nevals=len(invalid_ind),
                                    beval=best_ind_gen.original_fitness if weighted else best_ind_gen.fitness.values,
                                    sum=sum(best_ind_gen.fitness.values)
                                    if weighted else (0.6 * best_ind_gen.fitness.values[0] + 0.4 *
                                                      best_ind_gen.fitness.values[1]), **record)
            print(ea_utils.logbook.stream)

        # Update the Pareto front with the offspring
        pareto_front.update(offspring)

        # After NGEN generations, select the best individual
        if weighted:
            best_ind = ea_utils.selWeighted(pop, 1, False)[0]
        else:
            best_ind = ea_utils.selWeighted(pop, 1, True)[0]

        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = ea_utils.genotyp_phenotyp_encapsulation([best_ind], 1)[0]

        if weighted:
            best_ind.original_fitness = best_ind_values[:2]
            best_ind.fitness.values = [best_ind.original_fitness[0] * 0.6, best_ind.original_fitness[1] * 0.4]
        else:
            best_ind.fitness.values = best_ind_values[:2]

        best_ind.original_evaluation = best_ind_values[2:]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        print("Best individual is %s, %s" % (best_ind.original_fitness if weighted else best_ind.fitness.values))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        print(best_ind_values, best_ind.original_fitness if weighted else best_ind.fitness.values,
              best_ind.original_evaluation, best_ind.fitness.values)
        # Update the statistics and logbook for the final evaluation
        record = ea_utils.mstats.compile([best_ind.fitness.values])
        ea_utils.final_logbook.record(day=day, RMSD=best_ind.original_evaluation[0],
                                      cost=best_ind.original_evaluation[1],
                                      RMSD_note=best_ind.original_fitness[0] if weighted else best_ind.fitness.values[
                                          0],
                                      cost_note=best_ind.original_fitness[1] if weighted else best_ind.fitness.values[
                                          1],
                                      Weighted_RMSD_note=best_ind.fitness.values[0] if weighted else None,
                                      Weighted_cost_note=best_ind.fitness.values[1] if weighted else None,
                                      whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                          'mean_value_weighted'],
                                      time=day_end - day_start, number_of_ems_evaluations=invalid_ind_length_full)
        ea_utils.gen_logbook.record(day=day, gen=g + 2,
                                    RMSD=best_ind_gen.original_evaluation[0],
                                    cost=best_ind_gen.original_evaluation[1],
                                    RMSD_note=best_ind_gen.original_fitness[0] if weighted else
                                    best_ind_gen.fitness.values[0],
                                    cost_note=best_ind_gen.original_fitness[1] if weighted else
                                    best_ind_gen.fitness.values[1],
                                    Weighted_RMSD_note=best_ind_gen.fitness.values[0] if weighted else None,
                                    Weighted_cost_note=best_ind_gen.fitness.values[1] if weighted else None,
                                    whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                        'mean_value_weighted'],
                                    generation_time=0, evaluation_time=chosen_individual_evaluation,
                                    number_of_ems_evaluations=1)

        # Plot the evolution and pareto front and save them to PDF files
        ea_utils.plot_evolution(ea_utils.gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.pdf")
        if weighted:
            ea_utils.plot_pareto_front([pareto_front], day, f'logs/run_{run_id}/plots/pareto_front_day_{day}.pdf',
                                       weighted=True)
        else:
            ea_utils.plot_pareto_front([pareto_front], day, f'logs/run_{run_id}/plots/pareto_front_day_{day}.pdf',
                                       weighted=False)
        ea_utils.fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
        print(f"Day {day}, best result: {[best_ind.original_evaluation[0], best_ind.original_evaluation[1]]}")

    global_time_end = round(time.time() * 1000)
    print("Total Time: ", global_time_end - global_time_start, "ms")

    # Save the best individuals from every day to a pickle and txt file
    ea_utils.save_logs(run_id, 'evolution_log', ea_utils.final_logbook, mode='ab')
    # Save the best individuals in every generation list to a separate pickle and txt file
    ea_utils.save_logs(run_id, 'evolution_log_full', ea_utils.gen_logbook)
    # Save the best individuals in every generation list to a separate pickle and txt file
    ea_utils.save_logs(run_id, 'statistics_full', ea_utils.logbook)
    # Usage of the save_logs function for NSGA-II Pareto front
    ea_utils.save_logs(run_id, 'pareto_front', pareto_front, is_pareto=True)
    if weighted:
        return best_ind.original_fitness
    else:
        return best_ind.fitness.values
