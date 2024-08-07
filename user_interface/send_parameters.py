import requests
import json

def send_data(
    population_size,
    number_of_islands,
    number_of_generations,
    date,
    url='http://127.0.0.1:8072/run'  # Default URL, can be changed if needed
):
    # Define the JSON data to be sent
    data = {
        "populationSize": [population_size],
        "numberOfIslands": [number_of_islands],
        "numberrOfGeneration": [number_of_generations],
        "date": date,
    }

    # Send the POST request
    response = requests.post(url, json=data)

    # Print the response from the server
    print('Status Code:', response.status_code)
    print('Response JSON:', response.json())

# Example usage:
send_data(
    population_size=100,
    number_of_islands=1,
    number_of_generations=50,
    date="2022-04-11",
)