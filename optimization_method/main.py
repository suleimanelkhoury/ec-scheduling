from flask import Flask, request
from configuration import configuration
import NSGA2
import PSO
import CMAES
import subprocess
import os
import json
import time
import redis
import sys
import docker
import logging
#import optuna
#from multiprocessing import Process, Manager
#import csv
# Initialize Flask app
app = Flask(__name__)
# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379, db=0)
# Initialize Docker client
client = docker.from_env()
#import configuration_receiver
configuration_handler = configuration()

# display logs on the terminal
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s.%(msecs)03d %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

@app.route('/run', methods=['POST'])
def receive_json():
    json_data = json.loads(request.data.decode('utf-8'))
    # Process the received JSON data
    population_size = json_data.get('populationSize')
    num_islands = json_data.get('numberOfIslands')[0]
    num_generations = json_data.get('numberrOfGeneration')
    date = json_data.get('date')
    configuration_handler.process_json(json_data)
    r.set('proof.island.amount', num_islands)
    r.set('proof.generation', num_generations[0])
    r.set('proof.date', date)
    # Create a new container and check its status
    island_number = 1  # Replace with your logic to determine island number
    container_name = docker_container_mockup(island_number)
    # Check the container status
    if container_name:
        container = check_container_status(container_name)
        if container:
            create_container_mockup(container_name)
        else:
            logging.error(f"Failed to find container '{container_name}'.")
    else:
        logging.error("Failed to create container.")

    time.sleep(5)
    RMSD, Cost = NSGA2.algorithm_unweighted(
            population_num=configuration_handler.population_size[0],
            CXPB=0.9, eta=30, indpb=0.01, mu=5, sigma=30,
            NGEN=configuration_handler.num_generations[0],
            num_days=7, islands=num_islands, run_id=1
        )

    """
    RMSD, Cost = PSO.algorithm_weighted(
        population_num=configuration_handler.population_size[0],
        scaling_factor=0.1, phi1=2.0, phi2=2.0,
        NGEN=configuration_handler.num_generations[0],
        num_days=1, islands=num_islands, run_id=1
    )


    RMSD, Cost = CMAES.algorithm_weighted(
        lambda_=0.28, N=720, centroid=15, sigma=15,
        NGEN=configuration_handler.num_generations[0],
        num_days=1, islands=num_islands, run_id=1
    )
    """



    return 'ok'


def docker_container_mockup(island_number):
    # Define container parameters
    container_name = f"mockup_scheduler_{island_number}"
    image_name = "mockup-scheduler"
    redis_container_name = "redis"

    try:
        # Get the network that Redis is connected to
        redis_container = client.containers.get(redis_container_name)
        redis_network = list(redis_container.attrs['NetworkSettings']['Networks'].keys())[0]
        print(f"Using network '{redis_network}' for container '{container_name}'.")
        # Create container
        container = client.containers.run(
            image=image_name,
            name=container_name,
            detach=True,
            tty=True,
            mem_limit='1024m',
            cpu_shares=256,
            network=redis_network,  # Use the same network as Redis
            # links are not needed when using the same network
            # auto_remove=True  # Temporarily commented out for debugging
        )

        logging.info(f"Container '{container_name}' created successfully.")
        return container_name
    except docker.errors.ContainerError as e:
        logging.error(f"Container '{container_name}' exited with error: {e.stderr.decode('utf-8')}")
        return None
    except docker.errors.APIError as e:
        logging.error(f"Failed to create container '{container_name}': {e}")
        return None

def check_container_status(container_name):
    try:
        container = client.containers.get(container_name)
        logging.info(f"Container '{container_name}' status: {container.status}")
        return container
    except docker.errors.NotFound:
        logging.info(f"Container '{container_name}' does not exist.")
        return None

def create_container_mockup(container_name):
    try:
        logging.info(f"Container Mockup '{container_name}' created")
    except docker.errors.APIError as e:
        if e.response.status_code == 409:
            logging.info(f"EMS Container '{container_name}' already exists, sending initializing signal")
            # Here you can send a signal or perform an action as needed
        else:
            logging.error(e)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8072", debug=True)
    print('test')