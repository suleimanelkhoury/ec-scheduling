#!/bin/bash

# Name of the Docker containers
REDIS_CONTAINER="redis"
EC_SCHEDULER_CONTAINER="ec-scheduler"

# Stop and remove Redis container
echo "Stopping and removing Redis container..."
sudo docker stop $REDIS_CONTAINER 2>/dev/null
sudo docker rm $REDIS_CONTAINER 2>/dev/null

# Stop and remove ec-scheduler container
echo "Stopping and removing ec-scheduler container..."
sudo docker stop $EC_SCHEDULER_CONTAINER 2>/dev/null
sudo docker rm $EC_SCHEDULER_CONTAINER 2>/dev/null

# Remove Redis image
echo "Removing Redis image..."
sudo docker rmi $REDIS_IMAGE 2>/dev/null

# Remove ec-scheduler image
echo "Removing ec-scheduler image..."
sudo docker rmi $EC_SCHEDULER_IMAGE 2>/dev/null
#sudo docker system prune -f
echo "Removing mockup_scheduler image..."
sudo docker stop mockup_scheduler_1
sudo docker rm mockup_scheduler_1
sudo docker rmi mockup_scheduler_1 2>/dev/null
# Check if the commands were successful
if [ $? -eq 0 ]; then
  echo "Successfully stopped and removed containers and images."
else
  echo "Failed to stop and remove containers or images."
  exit 1
fi