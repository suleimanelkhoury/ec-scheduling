REDIS_PORT="6379"
# Directory containing the Dockerfiles (path to the directory, not the Dockerfile itself)
DOCKERFILE_DIR_optimization_method="optimization_method"
DOCKERFILE_DIR_mockup_scheduler="mockup_scheduler"
# Names of the Docker images
IMAGE_NAME="ec-scheduler"
IMAGE_NAME2="mockup-scheduler"
# Path to the Docker Compose file
DOCKER_COMPOSE_FILE="docker-compose.yml"

# package needed for the ui
sudo apt-get install libxcb-xinerama0
# Pull the latest Redis image from Docker Hub
echo "Check if we already have latest redis image..."
# Check if the image is already available locally
LOCAL_TAG=$(sudo docker images --format '{{.Tag}}' | grep "^$TAG$")

# Check if the local image has the latest tag
if [[ "$LOCAL_TAG" == "$TAG" ]]; then
    echo "Image 'redis:$TAG' already exists locally."
else
    echo "Image 'redis:$TAG' not found locally. Pulling it..."
    sudo docker pull "redis:$TAG"
fi

# Stop and remove any existing container with the same name
echo "Stopping and removing any existing redis container..."
sudo docker stop redis 2>/dev/null
#sudo docker rm redis 2>/dev/null

# Build the ec-scheduler image
echo "Building Docker image '$IMAGE_NAME' from directory '$DOCKERFILE_DIR_optimization_method'..."
sudo docker build -t $IMAGE_NAME $DOCKERFILE_DIR_optimization_method

# build the mockup-scheduler image
echo "Building Docker image '$IMAGE_NAME2' from directory '$DOCKERFILE_DIR_mockup_scheduler'..."
sudo docker build -t $IMAGE_NAME2 $DOCKERFILE_DIR_mockup_scheduler

# Apply Docker Compose configuration to run both containers
echo "Applying Docker Compose configuration..."
sudo docker compose -f $DOCKER_COMPOSE_FILE up -d

# Check if the Docker Compose command was successful
if [ $? -eq 0 ]; then
  echo "Docker Compose has started the containers successfully."
else
  echo "Failed to start containers with Docker Compose."
  exit 1
fi