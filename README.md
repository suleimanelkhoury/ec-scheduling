# **EC-Scheduler**

## **Project Overview**

Welcome to **EC-Scheduler**! This project provides a comprehensive solution for optimizing and scheduling general energy resources. It consists of three main components:

1. **Optimization Method (`ec-scheduler`)**:
   This part of our work implements three various evolutionary algorithms using the DEAP (Distributed Evolutionary Algorithms in Python) library. It includes:
   - **`NSGA2.py`** : Non-dominated Sorting Genetic Algorithm II.
   - **`PSO.py`** : Particle Swarm Optimization.
   - **`CMAES.py`** : Covariance Matrix Adaptation Evolution Strategy.
   - **`ea_utils.py`**: This file contains the helper, statistics, plotting, and parallelization functionalities for each evolutionary algorithms.
   - **`Strategy.py`**: This file contains the strategy logic for generating and updating the population in CMA-ES.
   - **`genotyp_phenotyp.py`**: This file contains logic used to convert chromosomes into the corresponding scheduling plans, and send and receive the values to and from the Mockup Scheduler.
   - **`main.py`**: This file contains the main Flask app for the project.
   - **`container_handling.py`**: This file contains the logic for handling the containers on the IAI cluster using Kubernetes.

2. **Mockup Scheduler (`mockup_scheduler`)**:
   This component simulates the scheduling of five different types of energy facilities:
   - Photovoltaic (solar panels)
   - Wind Turbine
   - Two Batteries
   - Combined Heat and Power Plant

3. **User Interface (`user_interface`)**:
   Built using PyQt5, this component provides a graphical interface to interact with the optimization methods. It allows users to configure and execute optimization tasks and view results.

## **Getting Started**

To get started with **ec-scheduler**, simply execute the `start_services.sh` script to build and run the necessary Docker containers. This script handles the setup and execution of all services required for the project.

## **Project Structure**

- **`start_services.sh`**: Script to build and run Docker containers.
- **`delete_services.sh`**: Script to delete Docker containers.
- **`optimization_method`**: Directory for the optimization methods.
- **`mockup_scheduler`**: Directory for a simple mockup scheduler project.
- **`user_interface`**: Directory for the PyQt5 UI.

## **Contributing**

Contributions are welcome! To contribute, please fork the repository and submit a pull request.