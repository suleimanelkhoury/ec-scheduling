import time
import json
import redis
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
import concurrent.futures


# Define an exponential function
def exponential_function(x, a, b, c):
    return a * np.exp(b * x) + c


class ResourcePlan:
    """
    Represents the resource plan for a specific facility within a scheduling plan.

    Attributes:
    - resource_id (int): The identifier for the resource.
    - power_generation (list): A list representing the power generation fractions over time.

    Methods:
    - get_resource_id():
        Returns the identifier of the facility.

    - set_resource_id(resource_id):
        Sets the identifier of the facility.

    - get_power_fraction():
        Returns the power generation fractions for the facility.

    - set_power_fraction(power_fraction):
        Sets the power generation fractions for the facility.
    """

    def __init__(self):
        self.resource_id = 0
        self.power_generation = []

    def get_resource_id(self):
        return self.resource_id

    def set_resource_id(self, resource_id):
        self.resource_id = resource_id

    def get_power_fraction(self):
        return self.power_generation

    def set_power_fraction(self, power_fraction):
        self.power_generation = power_fraction


class SchedulingPlan:
    """
    Represents a scheduling plan consisting of resource plans for the different facilities.

    Attributes:
    - plan_id (int): The identifier for the scheduling plan.
    - nr_of_genes (int): The number of genes (tasks) in the scheduling plan.
    - resource_plan (ResourcePlan): The resource plan for the facilities associated with this scheduling plan.

    Methods:
    - get_plan_id(): Returns the identifier of the scheduling plan.
    - set_plan_id(plan_id): Sets the identifier of the scheduling plan.
    - get_resource_plan(): Returns the resource plan for the facilities associated with the scheduling plan.
    - print_resource_plan(): Prints the resource plan details (resource ID and power fraction).
    - set_resource_plan(resource_plan): Sets the resource plan for the facilities.
    - get_nr_of_genes(): Returns the number of genes in the scheduling plan.
    - set_nr_of_genes(nr_of_genes): Sets the number of genes in the scheduling plan.
    """

    def __init__(self):
        self.plan_id = 0
        self.nr_of_genes = 0
        self.resource_plan = ResourcePlan()

    def get_plan_id(self):
        return self.plan_id

    def set_plan_id(self, plan_id):
        self.plan_id = plan_id

    def get_resource_plan(self):
        return self.resource_plan

    def print_resource_plan(self):
        for resource_plan_instance in self.resource_plan:
            print(
                f"Resource ID: {resource_plan_instance.get_resource_id()}, Power Fraction: {resource_plan_instance.get_power_fraction()}")

    def set_resource_plan(self, resource_plan):
        self.resource_plan = resource_plan

    def get_nr_of_genes(self):
        return self.nr_of_genes

    def set_nr_of_genes(self, nr_of_genes):
        self.nr_of_genes = nr_of_genes


# Main class for interpreting the chromosomes incoming from algorithm_logic
class GenotypPhenotyp:
    """
    Converts a list of chromosomes into a corresponding list of scheduling plans, sends them for evaluation to EMS,
    receives the evaluation values from EMS, and returns them to the algorithm_logic.

    Methods:
    - interpretation(chromosome_list, microservice_index):
        Interprets a list of chromosomes, converts them to scheduling plans, sorts the results, and returns the
        evaluation values.

    - save_chromosome_to_file(chromosome, file_path):
        Appends a chosen chromosome to a specified file.

    - add_chromosome_id(chromosome_list):
        Adds a unique identifier to each chromosome in a list.

    - process_chromosome(chromosome):
        Processes a single chromosome to create a scheduling plan, updating the power fraction values based on the
        gene information.

    - set_power_fraction_inside(power_fr, start, end, pf, res_id):
        Sets the power fraction for a specific time range within the resource plan.

    - update_power_fraction_values(list_resource_plan, chromosome_genes):
        Updates the power fraction values inside the schedule based on the gene information.

    - sort_results(results):
        Sorts the interpreted scheduling plans into a format accepted by the EMS.

    - publish_ea_epoch_config(schedule, microservice_index):
        Publishes the scheduling plan to the EMS via Redis for evaluation.

    - calculate_exponential_value(x):
        Calculates an exponential value based on the predefined exponential function.

    - receive_value_from_redis(key):
        Waits for and retrieves evaluation values from Redis.

    - save_schedule_to_file(list_of_scheduling_plans, file_path):
        Appends a chosen schedule to a specified file.
    """

    redis_client = redis.StrictRedis(host='redis', port=6379)

    # Main functionality for the whole population of chromosomes
    def interpretation(self, chromosome_list, microservice_index):
        # Extract the number of chromosomes
        num_chromosomes = len(chromosome_list)

        # Write chosen chromosomes
        if num_chromosomes == 1:
            self.save_chromosome_to_file(self, chromosome_list, "/chosen_chromosomes_and_schedules.txt")

        chromosome_list_with_ids = self.add_chromosome_id(self, chromosome_list)

        # Main interpretation functionality parallelized
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit each chromosome to the ThreadPoolExecutor
            futures = [executor.submit(self.process_chromosome, self, chromosome) for chromosome in
                       chromosome_list_with_ids]
            # Wait for all tasks to complete
            concurrent.futures.wait(futures)
            results = [future.result() for future in futures]

        # Add scheduling plans to a list
        list_of_scheduling_plans = []
        for scheduling_plan in results:
            list_of_scheduling_plans.append(scheduling_plan)
            """
            print(
                f"planID: {scheduling_plan.get_plan_id()}, NrOfGenes: {scheduling_plan.get_nr_of_genes()}"
                f"powerGeneration: {scheduling_plan.print_resource_plan()}")
            """

        formated_schedule = self.sort_results(self, results)
        # print(formated_schedule)

        # Publish EA epoch configuration with microservice_index
        self.publish_ea_epoch_config(self, formated_schedule, microservice_index)

        # Receive evaluation values from Redis using microservice_index
        evaluation_values = self.receive_value_from_redis(self, f"proof.result.{microservice_index}")

        if len(list_of_scheduling_plans) == 1:
            self.save_schedule_to_file(self, list_of_scheduling_plans, "/logs/chosen_chromosomes_and_schedules.txt")
        return evaluation_values

    # Appends the chosen chromosomes every day to a file
    def save_chromosome_to_file(self, chromosome, file_path):
        with open(file_path, 'a') as file:
            file.write('************************************' + '\n')
            for gene in chromosome:
                file.write(','.join(map(str, gene)) + '\n')

    # Adds chromosome_ids to Deap population
    def add_chromosome_id(self, chromosome_list):
        new_chromosome_list = []
        for chromosome_id, chromosome in enumerate(chromosome_list):
            new_chromosome = chromosome[:]  # Create a shallow copy of the chromosome
            new_chromosome.append(chromosome_id)
            new_chromosome_list.append(new_chromosome)
        return new_chromosome_list

    # Functionality for processing one chromosome at a timme
    def process_chromosome(self, chromosome):

        number_of_genes = 0
        resources = set()
        scheduling_plan = SchedulingPlan()
        list_scheduling_plan = []
        # get the chromosome id
        scheduling_plan.plan_id = chromosome[-1]
        # delete the chromosome id to iterate only over genes
        chromosome_genes = chromosome[:-1]

        for gene in chromosome_genes:
            # Extract the unit ID from the gene
            unit_id = gene[0]
            number_of_genes += 1
            # Add the unit ID to the set
            resources.add(unit_id)

        scheduling_plan.nr_of_genes = number_of_genes
        # Convert the set to a list if needed
        sorted_resources = list(resources)
        list_resource_plan = []
        chromosome_length = 96
        # initiate resource plan list for the chromosome
        for t in range(len(sorted_resources)):
            resource_plan = ResourcePlan()
            resource_plan.set_resource_id(sorted_resources[t])
            resource_plan.set_power_fraction(self.set_power_fraction_inside(self, [0.0] * chromosome_length, 0,
                                                                            chromosome_length, 0.0, 0))
            list_resource_plan.append(resource_plan)
        self.update_power_fraction_values(self, list_resource_plan, chromosome_genes)
        scheduling_plan.set_resource_plan(list_resource_plan)
        return scheduling_plan

    # Sets the power fraction inside the schedule
    def set_power_fraction_inside(self, power_fr, start, end, pf, res_id):
        for s in range(start, end):
            power_fr[s] = pf
        return power_fr

    # updates the power fraction values inside the schedule depending on the values of the chromosome [id, start_time, duration, power_fraction]
    def update_power_fraction_values(self, list_resource_plan, chromosome_genes):
        for gene in chromosome_genes:
            resource_id, start_time, duration, power_fraction = gene
            for resource_plan in list_resource_plan:
                if resource_plan.resource_id == resource_id:
                    # Ensure end_time does not exceed the length of the power fraction list
                    end_time = min(start_time + duration, 96)
                    resource_plan.power_generation[start_time:end_time] = [power_fraction] * (end_time - start_time)

    # function to sort results in a way that EMS accepts
    def sort_results(self, results):
        sorted_results = []

        for scheduling_plan in results:
            sorted_result = {
                "planID": scheduling_plan.get_plan_id(),
                "NrOfGenes": scheduling_plan.get_nr_of_genes(),
                "resourcePlan": []
            }

            for resource_plan in scheduling_plan.get_resource_plan():
                resource_result = {
                    "resourceID": resource_plan.get_resource_id(),
                    "powerGeneration": resource_plan.get_power_fraction()
                }
                sorted_result["resourcePlan"].append(resource_result)

            sorted_results.append(sorted_result)

        return sorted_results

    # Publish schedule to EMS for evaluation
    def publish_ea_epoch_config(self, schedule, microservice_index):
        # Redis topic name
        topic_name = f"algorithm.EA.epoch.{microservice_index}"

        # cleans the topic "proof.result.1" and "algorithm.EA.epoch.1" so it doesn't receive an old value
        self.redis_client.delete(f"proof.result.{microservice_index}")

        # Publish to Redis channel
        schedule_json = json.dumps(schedule)
        self.redis_client.publish(topic_name, schedule_json)
        # print(f"Published Scheduling plan to EMS Channel: {topic_name}")

    # Calculate the exponential value for a given x
    def calculate_exponential_value(self, x):
        # Define the three predefined points
        x_data = np.array([0.0, 0.67, 1.0])
        y_data = np.array([0, 33333, 100000])

        # Define the fitted parameters, and Suppress the OptimizeWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            # Define the fitted parameters
            fitted_params, covariance = curve_fit(exponential_function, x_data, y_data)

        # Create a function to calculate the exponential value for given x
        return int(exponential_function(x, *fitted_params))

    # Wait for ems to send the evaluation value back
    def receive_value_from_redis(self, key):
        # Loop and wait for the result from Redis
        received_value = None
        while received_value is None:
            time.sleep(1)  # Adjust the sleep time as needed
            received_value = self.redis_client.get(key)
        # Split the lines and extract (degree_of_fulfillment, root_mean_square_deviation)
        evaluation_results = received_value.decode(
            'utf-8')  # convert bytes into strings to process every new line as a value
        # result_tuples = [(1.0 - float(line.split()[1]), float(line.split()[0])) for line in evaluation_results.strip().split('\n')]
        result_tuples = [
            (self.calculate_exponential_value(self, 1.0 - float(line.split()[1])),
             self.calculate_exponential_value(self, float(line.split()[0])),
             float(line.split()[1]),
             float(line.split()[0]))
            for line in evaluation_results.strip().split('\n')
        ]
        # print("result_tuples: ", result_tuples)
        return result_tuples

    # Appends the chosen schedule every day to a file
    def save_schedule_to_file(self, list_of_scheduling_plans, file_path):
        with open(file_path, 'a') as file:
            file.write('************************************' + '\n')
            for scheduling_plan in list_of_scheduling_plans:
                file.write(
                    f"Schedule Plan ID: {scheduling_plan.get_plan_id()}, "
                    f"Number Of Genes: {scheduling_plan.get_nr_of_genes()}, "
                    f"Resource plan: {scheduling_plan.print_resource_plan()}\n")
