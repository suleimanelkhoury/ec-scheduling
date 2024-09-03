from flask import Flask, request
from configuration import configuration
import NSGA2
import PSO
import numpy
import CMAES
import subprocess
import os
import json
import time
import redis
import sys
import docker
import logging
import container_handling
import optuna
from multiprocessing import Process, Manager
import csv

# =========================
# SECTION 1: Initialize Flask, Redis, Configuration Handler, and Logging
# =========================

# Create a new Flask application, set the Redis host, and initialize the configuration handler
app = Flask(__name__)
r = redis.StrictRedis(host='redis', port=6379, db=0)
configuration_handler = configuration()

# display logs on the terminal
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# Function to process the experiment setup received from the frontend
def process_json_request(request_data):
    # Read the JSON data from the request
    json_data = json.loads(request_data.decode('utf-8'))

    # Extract the parameters from the JSON data
    population_size = json_data.get('populationSize')
    num_islands = json_data.get('numberOfIslands')[0]
    num_generations = json_data.get('numberrOfGeneration')
    date = json_data.get('date')
    choice = json_data.get('choice')

    configuration_handler.process_json(json_data)

    r.set('proof.island.amount', num_islands)
    r.set('proof.generation', num_generations[0])
    r.set('proof.date', date)

    return population_size, num_islands, num_generations, date, choice


# Define the main algorithm runner with the fine-tuned parameters
def run_algorithm(run_id, num_islands, choice):
    print(f"Running algorithm with choice: {choice}")
    try:
    # Select and execute the algorithm based on the choice parameter
        if choice == 'NSGA2':
            RMSD, Cost = NSGA2.algorithm_unweighted(
                population_num=configuration_handler.population_size[0],
                CXPB=0.9, eta=30, indpb=0.01, mu=5, sigma=30,
                NGEN=configuration_handler.num_generations[0],
                num_days=1, islands=num_islands, run_id=run_id
            )
        elif choice == 'PSO':
            RMSD, Cost = PSO.algorithm(
                population_num=configuration_handler.population_size[0],
                scaling_factor=0.1, phi1=2.0, phi2=2.0,
                NGEN=configuration_handler.num_generations[0],
                num_days=1, islands=num_islands, run_id=run_id, weighted=True
            )
        elif choice == 'CMAES':
            RMSD, Cost = CMAES.algorithm(
                pop=configuration_handler.population_size[0], N=720, centroid=3.5, sigma=3.5,
                NGEN=configuration_handler.num_generations[0],
                num_days=1, islands=num_islands, run_id=run_id, weighted=True
            )
    finally:
        print("done")
        # Clean up resources after the algorithm run
        # container_handling.delete_pods(num_islands)


# Helper function to run the algorithm with a timeout
def run_algorithm_with_timeout(run_id, timeout, num_islands, choice):
    process = Process(target=run_algorithm, args=(run_id, num_islands, choice))
    process.start()
    process.join(timeout)

    # Terminate the process if it exceeds the timeout
    if process.is_alive():
        process.terminate()
        process.join()
        print(f"Run {run_id} timed out after {timeout} seconds")


@app.route('/run', methods=['POST'])
def receive_json():
    population_size, num_islands, num_generations, date, choice = process_json_request(request.data)
    # Create a new container and check its status

    # Create
    container_name = container_handling.docker_container_mockup(num_islands)

    # Check the container status
    if container_name:
        container = container_handling.check_container_status(container_name)
        if container:
            container_handling.create_container_mockup(container_name)
        else:
            logging.error(f"Failed to find container '{container_name}'.")
    else:
        logging.error("Failed to create container.")
    print(f"Received JSON data: {request.data}")
    run_algorithm_with_timeout(1, 3600, num_islands, choice)

    return 'ok'



if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8072", debug=True)