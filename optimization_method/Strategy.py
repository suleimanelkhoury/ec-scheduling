import numpy
from math import sqrt, log, exp

# The CMA-ES algorithm Strategy class.
class Strategy(object):
    """
    A strategy that will keep track of the basic parameters of the CMA-ES
    algorithm ([Hansen2001]_).

    :param centroid: An iterable object that indicates where to start the
                     evolution.
    :param sigma: The initial standard deviation of the distribution.
    :param parameter: One or more parameter to pass to the strategy as
                      described in the following table, optional.

    +----------------+---------------------------+----------------------------+
    | Parameter      | Default                   | Details                    |
    +================+===========================+============================+
    | ``lambda_``    | ``int(4 + 3 * log(N))``   | Number of children to      |
    |                |                           | produce at each generation,|
    |                |                           | ``N`` is the individual's  |
    |                |                           | size (integer).            |
    +----------------+---------------------------+----------------------------+
    | ``mu``         | ``int(lambda_ / 2)``      | The number of parents to   |
    |                |                           | keep from the              |
    |                |                           | lambda children (integer). |
    +----------------+---------------------------+----------------------------+
    | ``cmatrix``    | ``identity(N)``           | The initial covariance     |
    |                |                           | matrix of the distribution |
    |                |                           | that will be sampled.      |
    +----------------+---------------------------+----------------------------+
    | ``weights``    | ``"superlinear"``         | Decrease speed, can be     |
    |                |                           | ``"superlinear"``,         |
    |                |                           | ``"linear"`` or            |
    |                |                           | ``"equal"``.               |
    +----------------+---------------------------+----------------------------+
    | ``cs``         | ``(mueff + 2) /           | Cumulation constant for    |
    |                | (N + mueff + 3)``         | step-size.                 |
    +----------------+---------------------------+----------------------------+
    | ``damps``      | ``1 + 2 * max(0, sqrt((   | Damping for step-size.     |
    |                | mueff - 1) / (N + 1)) - 1)|                            |
    |                | + cs``                    |                            |
    +----------------+---------------------------+----------------------------+
    | ``ccum``       | ``4 / (N + 4)``           | Cumulation constant for    |
    |                |                           | covariance matrix.         |
    +----------------+---------------------------+----------------------------+
    | ``ccov1``      | ``2 / ((N + 1.3)^2 +      | Learning rate for rank-one |
    |                | mueff)``                  | update.                    |
    +----------------+---------------------------+----------------------------+
    | ``ccovmu``     | ``2 * (mueff - 2 + 1 /    | Learning rate for rank-mu  |
    |                | mueff) / ((N + 2)^2 +     | update.                    |
    |                | mueff)``                  |                            |
    +----------------+---------------------------+----------------------------+

    .. [Hansen2001] Hansen and Ostermeier, 2001. Completely Derandomized
       Self-Adaptation in Evolution Strategies. *Evolutionary Computation*

    """
    def __init__(self, centroid, sigma, **kargs):
        self.params = kargs  # Store additional parameters in a dictionary

        # Create a centroid as a numpy array
        self.centroid = numpy.array(centroid)  # Initial mean of the distribution

        self.dim = len(self.centroid)  # Dimensionality of the problem
        self.sigma = sigma  # Initial step-size (standard deviation of the distribution)
        self.pc = numpy.zeros(self.dim)  # Evolution path for the covariance matrix
        self.ps = numpy.zeros(self.dim)  # Evolution path for the step-size
        self.chiN = sqrt(self.dim) * (1 - 1. / (4. * self.dim) + 1. / (21. * self.dim ** 2))  # Expectation of the norm of a Gaussian vector

        self.C = self.params.get("cmatrix", numpy.identity(self.dim))  # Covariance matrix, initialized to identity if not provided
        self.diagD, self.B = numpy.linalg.eigh(self.C)  # Eigen decomposition of C to get initial B and diagD

        indx = numpy.argsort(self.diagD)  # Sort indices of eigenvalues
        self.diagD = self.diagD[indx] ** 0.5  # Square root of eigenvalues
        self.B = self.B[:, indx]  # Eigenvectors sorted according to eigenvalues
        self.BD = self.B * self.diagD  # B times the square root of D

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]  # Condition number of the covariance matrix

        self.lambda_ = self.params.get("lambda_", int(4 + 3 * log(self.dim)))  # Number of offspring
        self.update_count = 0  # Counter for the number of updates
        self.computeParams(self.params)  # Compute additional strategy parameters

    def generate(self, ind_init):
        r"""Generate a population of :math:`\lambda` individuals of type
        *ind_init* from the current strategy.

        :param ind_init: A function object that is able to initialize an
                         individual from a list.
        :returns: A list of individuals.
        """
        arz = numpy.random.standard_normal((self.lambda_, self.dim))  # Generate random samples from a standard normal distribution
        arz = self.centroid + self.sigma * numpy.dot(arz, self.BD.T)  # Transform samples to match the current distribution
        return [ind_init(a) for a in arz]  # Initialize individuals with the generated samples


    def update(self, population):
        """Update the current covariance matrix strategy from the
        *population*.

        :param population: A list of individuals from which to update the
                           parameters.
        """
        population.sort(key=lambda ind: ind.fitness, reverse=True)  # Sort population by fitness in descending order

        old_centroid = self.centroid  # Save the old centroid
        self.centroid = numpy.dot(self.weights, population[0:self.mu])  # Compute the new centroid as a weighted mean of the top mu individuals

        c_diff = self.centroid - old_centroid  # Difference between new and old centroids

        # Cumulation: update evolution path for step-size control
        self.ps = (1 - self.cs) * self.ps + sqrt(self.cs * (2 - self.cs) * self.mueff) / self.sigma * numpy.dot(self.B, (1. / self.diagD) * numpy.dot(self.B.T, c_diff))

        hsig = float((numpy.linalg.norm(self.ps) / sqrt(1. - (1. - self.cs) ** (2. * (self.update_count + 1.))) / self.chiN < (1.4 + 2. / (self.dim + 1.))))  # Heaviside step function for step-size control

        self.update_count += 1  # Increment the update counter

        self.pc = (1 - self.cc) * self.pc + hsig * sqrt(self.cc * (2 - self.cc) * self.mueff) / self.sigma * c_diff  # Update evolution path for covariance matrix

        # Update covariance matrix
        artmp = population[0:self.mu] - old_centroid  # Compute differences from old centroid for top mu individuals
        self.C = (1 - self.ccov1 - self.ccovmu + (1 - hsig) * self.ccov1 * self.cc * (2 - self.cc)) * self.C + self.ccov1 * numpy.outer(self.pc, self.pc) + self.ccovmu * numpy.dot((self.weights * artmp.T), artmp) / self.sigma ** 2

        self.sigma *= numpy.exp((numpy.linalg.norm(self.ps) / self.chiN - 1.) * self.cs / self.damps)  # Update step-size

        self.diagD, self.B = numpy.linalg.eigh(self.C)  # Recompute the eigen decomposition of C
        indx = numpy.argsort(self.diagD)  # Sort indices of eigenvalues

        self.cond = self.diagD[indx[-1]] / self.diagD[indx[0]]  # Update the condition number

        self.diagD = self.diagD[indx] ** 0.5  # Update square root of eigenvalues
        self.B = self.B[:, indx]  # Update eigenvectors
        self.BD = self.B * self.diagD  # Update B times the square root of D

    def computeParams(self, params):
        r"""Computes the parameters depending on :math:`\lambda`. It needs to
        be called again if :math:`\lambda` changes during evolution.

        :param params: A dictionary of the manually set parameters.
        """
        self.mu = params.get("mu", int(self.lambda_ / 2))  # Number of parents to keep

        # Define weights for recombination
        rweights = params.get("weights", "superlinear")
        if rweights == "superlinear":
            self.weights = log(self.mu + 0.5) - numpy.log(numpy.arange(1, self.mu + 1))  # Superlinear decrease
        elif rweights == "linear":
            self.weights = self.mu + 0.5 - numpy.arange(1, self.mu + 1)  # Linear decrease
        elif rweights == "equal":
            self.weights = numpy.ones(self.mu)  # Equal weights
        else:
            raise RuntimeError("Unknown weights : %s" % rweights)

        self.weights /= sum(self.weights)  # Normalize weights
        self.mueff = 1. / sum(self.weights ** 2)  # Variance-effectiveness of the weights

        self.cc = params.get("ccum", 4. / (self.dim + 4.))  # Cumulation constant for covariance matrix
        self.cs = params.get("cs", (self.mueff + 2.) / (self.dim + self.mueff + 3.))  # Cumulation constant for step-size
        self.ccov1 = params.get("ccov1", 2. / ((self.dim + 1.3) ** 2 + self.mueff))  # Learning rate for rank-one update
        self.ccovmu = params.get("ccovmu", 2. * (self.mueff - 2. + 1. / self.mueff) / ((self.dim + 2.) ** 2 + self.mueff))  # Learning rate for rank-mu update
        self.ccovmu = min(1 - self.ccov1, self.ccovmu)  # Ensure sum of learning rates is less than 1
        self.damps = 1. + 2. * max(0, sqrt((self.mueff - 1.) / (self.dim + 1.)) - 1.) + self.cs  # Damping for step-size control
        self.damps = params.get("damps", self.damps)  # Override damping if provided