class configuration:
    def __init__(self):
        self.population_size = None
        self.num_islands = None
        self.num_generations = None
        self.date = None

    def process_json(self, json_data):
        self.population_size = json_data.get('populationSize')
        self.num_islands = json_data.get('numberOfIslands')
        self.num_generations = json_data.get('numberrOfGeneration')
        self.date = json_data.get('date')