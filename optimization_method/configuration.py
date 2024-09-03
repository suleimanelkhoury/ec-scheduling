class configuration:
    """
    A class used to represent and configure settings for the optimization process.

    Attributes
    ----------
    population_size : int or None
        The size of the population in the simulation.
    num_islands : int or None
        The number of islands used for evaluation.
    num_generations : int or None
        The number of generations to optimize over.
    date : str or None
        The start date of the simulation or process.
    choice : str or None
        A variable to store the name of the optimization algorithm used.
    """

    def __init__(self):
        """
        Initializes the configuration class with all attributes set to None.
        """
        self.population_size = None
        self.num_islands = None
        self.num_generations = None
        self.date = None
        self.choice = None

    def process_json(self, json_data):
        """
        Processes a dictionary of JSON data to configure the attributes of the configuration class.

        Parameters
        ----------
        json_data : dict
            A dictionary containing the configuration data. Keys should match the expected attribute names in camelCase.
        """
        self.population_size = json_data.get('populationSize')
        self.num_islands = json_data.get('numberOfIslands')
        self.num_generations = json_data.get('numberrOfGeneration')
        self.date = json_data.get('date')
        self.choice = json_data.get('choice')
