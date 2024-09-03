import docker
import logging
import redis

# Initialize Redis client
r = redis.StrictRedis(host='redis', port=6379, db=0)
# Initialize Docker client
client = docker.from_env()

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
