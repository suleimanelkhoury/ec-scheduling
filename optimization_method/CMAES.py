import numpy
from Strategy import Strategy
from deap import base, creator, tools
import time
import redis
import os
import ea_utils

# =========================
# SECTION 1: Initialize Redis, Configure DEAP, Define individual mapper function, and Register the algorithm's parameters
# =========================

# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379)

# The cma module uses the numpy random number generator
numpy.random.seed(128)

# Initialize the individual as a flat list, and initialize the toolbox
creator.create("FitnessMul", base.Fitness, weights=(0.6, 0.4))
creator.create("IndividualFlat", list, fitness=creator.FitnessMul)
toolbox = base.Toolbox()


# Normalize each value of the individual to a range between min and max
def normalize_to_range(values, min_val, max_val):
    # Normalize values to the range [min_val, max_val]
    min_v = min(values)
    max_v = max(values)
    return [(v - min_v) / (max_v - min_v) * (max_val - min_val) + min_val for v in values]


# Map the values of the individual to the corresponding alleles, and split the individual list into 4 values per chunk
def map_values(individual):
    # Define the mapping ranges
    unit_ids = [1, 2, 3, 4, 6]
    # Normalize individual values to range [0, 95]
    normalized_individual = normalize_to_range(individual, 0, 95)
    # Initialize the result list
    result = []

    # Process each chunk of 4 values
    for i in range(0, len(normalized_individual), 4):
        # Normalize the values to integer index ranges
        uid_index = int(normalized_individual[i] * (len(unit_ids) - 1) / 95)
        uid = unit_ids[uid_index]
        start_time = int(normalized_individual[i + 1])
        duration = int(normalized_individual[i + 2])

        # Map power_fraction based on unit_id
        power_fraction = normalized_individual[i + 3]

        if uid in [1, 2, 6]:
            # Scale power_fraction to be between 0.0 and 1.0
            power_fraction = power_fraction / 95
        else:
            # Scale power_fraction to be between -1.0 and 1.0
            power_fraction = (power_fraction / 95) * 2 - 1
        # Create a chunk with mapped values
        chunk = (
            uid,
            start_time,
            duration,
            power_fraction
        )
        result.append(chunk)

    return result


# Register the toolbox functions with parameters for the CMA-ES algorithm
def register_toolbox(N, centroid, sigma, lambda_):
    # Use the strategy python class to initialize the CMAES strategy
    strategy = Strategy(centroid=[centroid] * N, sigma=sigma, lambda_=lambda_)
    toolbox.register("generate", strategy.generate, creator.IndividualFlat)
    toolbox.register("update", strategy.update)
    toolbox.register("map_values", map_values)


# =========================
# SECTION 2: Algorithm's main loop
# =========================

def algorithm(pop, N, centroid, sigma, NGEN, num_days, islands, run_id, weighted=True):
    """
    Execute the main algorithm loop, handling both weighted and unweighted versions based on the `weighted` flag.

    Parameters:
    - pop (int): Population size.
    - N (int): Number of alleles in the flattened chromosome.
    - centroid (float): Mean or center of the population distribution.
    - sigma (float): Standard deviation.
    - NGEN (int): Number of generations per day.
    - num_days (int): Total number of days to run the evolution.
    - islands (int): Number of islands used for evaluation.
    - run_id (str): Identifier for the run, used for logging purposes.
    - weighted (bool): If True, apply weights to the fitness values during evolution; otherwise, use unweighted values.

    Returns:
    - best_ind_flat.fitness.values (list): The fitness values of the best individual.
    """

    # Calculate the whole execution time
    global_time_start = round(time.time() * 1000)
    register_toolbox(N, centroid, sigma, pop)
    # Subscribe to the Redis channel
    pubsub = r.pubsub()
    pubsub.subscribe('ems_schedule_set')
    # Ensure the directory exists
    os.makedirs(f'logs/run_{run_id}', exist_ok=True)
    os.makedirs(f'logs/run_{run_id}/ems', exist_ok=True)

    for day in range(num_days):
        day_start = round(time.time() * 1000)
        # Initialize HallOfFame to keep the top 5 individuals
        hof = tools.HallOfFame(20)
        for g in range(NGEN):
            # Log the time used in the DEAP code
            generation_time_start = round(time.time() * 1000)
            # Generate a new population
            population = toolbox.generate()

            # Change the population shape for the map_values function
            population_mapped = [toolbox.map_values(ind) for ind in population]
            # Log the time used in the evaluation
            evaluation_time_start = round(time.time() * 1000)
            generation_time_1 = evaluation_time_start - generation_time_start

            # Evaluate the individuals
            fitnesses = ea_utils.evaluate_population(population_mapped, islands)

            # Update the fitness values based on the weighted flag
            for ind, fit in zip(population, fitnesses):
                if weighted:
                    scaled_fit = [fit[0] * 0.6, fit[1] * 0.4]
                    ind.original_fitness = fit[:2]  # Keep the original fitness
                    ind.original_evaluation = fit[2:]  # Save the original evaluation values
                    ind.fitness.values = scaled_fit
                else:
                    ind.fitness.values = fit[:2]
                    ind.original_evaluation = fit[2:]  # Save the original evaluation values

            evaluation_time_end = round(time.time() * 1000)
            evaluation_time = evaluation_time_end - evaluation_time_start

            # Select the best individual to log
            if weighted:
                best_ind_gen_flat = ea_utils.selWeighted(population, 1, use_weights=False)[0]
            else:
                best_ind_gen_flat = ea_utils.selWeighted(population, 1, use_weights=True)[0]

            # Update the strategy with the evaluated individuals
            toolbox.update(population)
            # Log the time used in the DEAP code
            generation_time_end = round(time.time() * 1000)
            generation_time_2 = generation_time_end - evaluation_time_end
            generation_time = generation_time_1 + generation_time_2
            # Update the HallOfFame with the current population
            hof.update(population)

            # Update the statistics and logbook for the current generation
            record = ea_utils.mstats.compile([best_ind_gen_flat.fitness.values])
            ea_utils.gen_logbook.record(day=day, gen=g + 1,
                               RMSD=best_ind_gen_flat.original_evaluation[0],
                               cost=best_ind_gen_flat.original_evaluation[1],
                               RMSD_note=best_ind_gen_flat.original_fitness[0] if weighted else
                               best_ind_gen_flat.fitness.values[0],
                               cost_note=best_ind_gen_flat.original_fitness[1] if weighted else
                               best_ind_gen_flat.fitness.values[1],
                               Weighted_RMSD_note=best_ind_gen_flat.fitness.values[0] if weighted else None,
                               Weighted_cost_note=best_ind_gen_flat.fitness.values[1] if weighted else None,
                               whole_note=record['fitness']['mean_value'] if weighted else record['fitness'][
                                   'mean_value_weighted'],
                               generation_time=generation_time,
                               evaluation_time=evaluation_time, number_of_ems_evaluations=len(population))

            record = ea_utils.stats.compile(population)
            ea_utils.logbook.record(run_id=run_id, day=day, gen=g, nevals=len(population),
                           beval=best_ind_gen_flat.original_fitness if weighted else best_ind_gen_flat.fitness.values,
                           sum=sum(best_ind_gen_flat.fitness.values)
                           if weighted else (0.6 * best_ind_gen_flat.fitness.values[0] + 0.4 * best_ind_gen_flat.fitness.values[1]), **record)
            print(ea_utils.logbook.stream)

        # After NGEN generations, select the best individual
        if weighted:
            best_ind_flat = ea_utils.selWeighted(population, 1, use_weights=False)[0]
        else:
            best_ind_flat = ea_utils.selWeighted(population, 1, use_weights=True)[0]
        # Map the best individual to the correct shape
        best_ind = toolbox.map_values(best_ind_flat)
        # Log the time used in the evaluation for the chosen individual
        chosen_individual_evaluation_start = round(time.time() * 1000)
        # Reevaluate the best individual with a default microservice index, e.g., 1
        best_ind_values = ea_utils.genotyp_phenotyp_encapsulation([best_ind], 1)[0]
        if weighted:
            best_ind_flat.original_fitness = best_ind_values[:2]
            best_ind_flat.original_evaluation = best_ind_values[2:]
            best_ind_flat.fitness.values = [best_ind_flat.original_fitness[0] * 0.6,
                                            best_ind_flat.original_fitness[1] * 0.4]
        else:
            best_ind_flat.fitness.values = best_ind_values[:2]
            best_ind_flat.original_evaluation = best_ind_values[2:]
        chosen_individual_evaluation_end = round(time.time() * 1000)
        chosen_individual_evaluation = chosen_individual_evaluation_end - chosen_individual_evaluation_start
        day_end = round(time.time() * 1000)
        print("Best individual is %s, %s" % (
            best_ind_flat.original_fitness if weighted else best_ind_flat.fitness.values))
        print(f"Time for Day {day}: {day_end - day_start} ms")
        # Update the statistics and logbook for the final evaluation
        record = ea_utils.mstats.compile([best_ind_flat.fitness.values])

        ea_utils.final_logbook.record(day=day, RMSD=best_ind_flat.original_evaluation[0],
                             cost=best_ind_flat.original_evaluation[1],
                             RMSD_note=best_ind_flat.original_fitness[0] if weighted else best_ind_flat.fitness.values[0],
                             cost_note=best_ind_flat.original_fitness[1] if weighted else best_ind_flat.fitness.values[1],
                             Weighted_RMSD_note=best_ind_flat.fitness.values[0] if weighted else None,
                             Weighted_cost_note=best_ind_flat.fitness.values[1] if weighted else None,
                             whole_note=record['fitness']['mean_value'] if weighted else record['fitness']['mean_value_weighted'],
                             time=day_end - day_start, number_of_ems_evaluations=len(population) * N + 1)
        ea_utils.gen_logbook.record(day=day, gen=g + 2,
                           RMSD=best_ind_flat.original_evaluation[0],
                           cost=best_ind_flat.original_evaluation[1],
                           RMSD_note=best_ind_flat.original_fitness[0] if weighted else best_ind_flat.fitness.values[0],
                           cost_note=best_ind_flat.original_fitness[1] if weighted else best_ind_flat.fitness.values[1],
                           Weighted_RMSD_note=best_ind_flat.fitness.values[0] if weighted else None,
                           Weighted_cost_note=best_ind_flat.fitness.values[1] if weighted else None,
                           whole_note=record['fitness']['mean_value'] if weighted else record['fitness']['mean_value_weighted'],
                           generation_time=0,
                           evaluation_time=chosen_individual_evaluation, number_of_ems_evaluations=len(population))

        # Plot the evolution and pareto front and save them to PDF files
        ea_utils.plot_evolution(ea_utils.gen_logbook, day, f"logs/run_{run_id}/plots/evolution_plot_day_{day}.pdf")
        ea_utils.fetch_and_save_json(f'logs/run_{run_id}/ems/timetable_run_{run_id}_day_{day}')
        if weighted:
            ea_utils.plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.pdf", weighted=True)
        else:
            ea_utils.plot_hall_of_fame(hof, day, f"logs/run_{run_id}/plots/hall_of_fame_day_{day}.pdf", weighted=False)
        print(f"Day {day}, best result: {[best_ind_flat.original_evaluation[0], best_ind_flat.original_evaluation[1]]}")

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
        return best_ind_flat.original_fitness
    else:
        return best_ind_flat.fitness.values
