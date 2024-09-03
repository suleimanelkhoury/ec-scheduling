import random
import math
import operator
from deap import base, creator, tools
import time
import redis
import os
import ea_utils

# =========================
# SECTION 1: Initialize Redis, Configure the population, and define Boundary Values.
# =========================

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
    bounds = [(0, 10), (0, 94), (0, 94), (-1.0, 1.0)]
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


# =========================
# SECTION 2: Define the generation and update rules for the algorithm, and register the toolbox functions.
# =========================

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


# Update the particle based on the Cognitive coefficient phi1 and Social coefficient phi2
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
    part[:] = checkBounds(part)
    return part


# Register the toolbox functions with parameters for smin and smax
def register_toolbox(scaling_factor, phi1, phi2):
    toolbox.register("particle", generate, scaling_factor=scaling_factor)
    toolbox.register("population", tools.initRepeat, list, toolbox.particle)
    toolbox.register("update", updateParticle, phi1=phi1, phi2=phi2)


# =========================
# SECTION 3: Algorithm's main loop
# =========================

# Define the main algorithm function
def algorithm(population_num, scaling_factor, phi1, phi2, NGEN, num_days, islands, run_id, weighted=True):
    """
    Execute the main algorithm loop, handling both weighted and unweighted versions based on the `weighted` flag.

    Parameters:
    - population_num (int): Population size.
    - scaling_factor (float): The scaling factor used in the mutation operation.
    - phi1 (float): Cognitive coefficient for the particle swarm optimization update rule.
    - phi2 (float): Social coefficient for the particle swarm optimization update rule.
    - NGEN (int): Number of generations to run for each day.
    - num_days (int): Total number of days to run the evolution.
    - islands (int): Number of islands used for evaluation.
    - run_id (str): Identifier for the run, used for logging purposes.
    - weighted (bool): If True, apply weights to the fitness values during evolution; otherwise, use unweighted values.

    Returns:
    - best_ind_flat.fitness.values (list): The fitness values of the best individual.
    """
    register_toolbox(scaling_factor, phi1, phi2)
    global_time_start = round(time.time() * 1000)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)

    for day in range(num_days):
        day_start = round(time.time() * 1000)
        # Initialize HallOfFame to keep the top individuals
        hof = tools.HallOfFame(20)
        # Initialize the population
        pop = toolbox.population(population_num)
        best = None
        for g in range(NGEN):
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            # Evaluate the population
            fitness_values_list = ea_utils.evaluate_population(pop, islands)

            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)

            # Assign the returned fitness values to each particle
            for part, fitness_values in zip(pop, fitness_values_list):
                if weighted:
                    scaled_fit = [fitness_values[0] * 0.6, fitness_values[1] * 0.4]  # Apply weights
                    part.original_fitness = fitness_values[:2]
                    part.fitness.values = scaled_fit
                else:
                    part.fitness.values = fitness_values[:2]

                part.original_evaluation = fitness_values[2:]  # save the original evaluation values

                # Check if the current particle fitness is the local best one
                if not part.best or sum(part.best.fitness.values) < sum(part.fitness.values):
                    part.best = creator.Particle(part)
                    part.best.original_evaluation = part.original_evaluation
                    part.best.fitness.values = part.fitness.values
                    if weighted:
                        part.best.original_fitness = part.original_fitness

                # Check if the current particle is the best in the population
                if not best or sum(best.fitness.values) < sum(part.fitness.values):
                    best = creator.Particle(part)
                    best.original_evaluation = part.original_evaluation
                    best.fitness.values = part.fitness.values
                    if weighted:
                        best.original_fitness = part.original_fitness

            for part in pop:
                toolbox.update(part, best)

            generation_time_end = round(time.time() * 1000)
            generation_time = generation_time_end - generation_time_start

            # Update the HallOfFame with the current population
            hof.update(pop)
            # Update the statistics and logbook for the current generation
            record = ea_utils.mstats.compile([best.fitness.values])
            ea_utils.gen_logbook.record(day=day, gen=g + 1,
                                        RMSD=best.original_evaluation[0],
                                        cost=best.original_evaluation[1],
                                        RMSD_note=best.original_fitness[0] if weighted else best.fitness.values[0],
                                        cost_note=best.original_fitness[1] if weighted else best.fitness.values[1],
                                        Weighted_RMSD_note=best.fitness.values[0] if weighted else None,
                                        Weighted_cost_note=best.fitness.values[1] if weighted else None,
                                        whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                            'mean_value_weighted'],
                                        generation_time=generation_time,
                                        evaluation_time=evaluation_time, number_of_ems_evaluations=population_num)

            record = ea_utils.stats.compile(pop)
            ea_utils.logbook.record(run_id=run_id, day=day, gen=g, nevals=len(pop),
                                    beval=best.original_fitness if weighted else best.fitness.values,
                                    sum=sum(best.fitness.values)
                                    if weighted else (0.6 * best.fitness.values[0] + 0.4 * best.fitness.values[1]),
                                    **record)
            print(ea_utils.logbook.stream)

        # After NGEN generations, select the best individual
        pop.append(best)
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
        print(f"Time for Day {day}: {day_end - day_start} ms")

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
                                      time=day_end - day_start, number_of_ems_evaluations=population_num * NGEN + 1)
        ea_utils.gen_logbook.record(day=day, gen=g + 1,
                                    RMSD=best_ind.original_evaluation[0],
                                    cost=best_ind.original_evaluation[1],
                                    RMSD_note=best_ind.original_fitness[0] if weighted else best_ind.fitness.values[0],
                                    cost_note=best_ind.original_fitness[1] if weighted else best_ind.fitness.values[1],
                                    Weighted_RMSD_note=best_ind.fitness.values[0] if weighted else None,
                                    Weighted_cost_note=best_ind.fitness.values[1] if weighted else None,
                                    whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                        'mean_value_weighted'],
                                    generation_time=0,
                                    evaluation_time=chosen_individual_evaluation,
                                    number_of_ems_evaluations=population_num)

        # Plot the evolution and pareto front and save them to PDF files
        ea_utils.plot_evolution(ea_utils.gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.pdf")
        ea_utils.fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
        if weighted:
            ea_utils.plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.pdf", weighted=True)
        else:
            ea_utils.plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.pdf", weighted=False)
        print(f"Day {day}, best result: {[best_ind.original_evaluation[0], best_ind.original_evaluation[1]]}")

    global_time_end = round(time.time() * 1000)
    print("Total Time: ", global_time_end - global_time_start, "ms")

    # Save the best individuals from every day to a pickle and txt file
    ea_utils.save_logs(run_id, 'evolution_log', ea_utils.final_logbook, mode='ab')
    # Save the best individuals in every generation list to a separate pickle and txt file
    ea_utils.save_logs(run_id, 'evolution_log_full', ea_utils.gen_logbook)
    # Save the best individuals in every generation list to a separate pickle and txt file
    ea_utils.save_logs(run_id, 'statistics_full', ea_utils.logbook)
    # Save the best individuals from the Hall of Fame
    ea_utils.save_logs(run_id, 'hall_of_fame', hof, dataframe_columns=[ind.fitness.values for ind in hof])

    if weighted:
        return best_ind.original_fitness
    else:
        return best_ind.fitness.values
