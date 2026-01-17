import numpy as np


# Class representing a monomial (single term polynomial)
class Monomial:
    """Class for representing and evaluating monomials"""

    # Constructor to initialize a monomial with exponent vector
    def __init__(self, a_vec):
        # Convert input to numpy array (exponent vector)
        self.a_vec = np.array(a_vec)
        # Dimension of the monomial (length of exponent vector)
        self.dimension = len(a_vec)

    # Evaluate the monomial at a given input vector
    def evaluate(self, input_vec):
        # Check that input dimension matches monomial dimension
        assert self.dimension == len(
            input_vec
        ), "Input data dimension is different from the monomial's dimension"
        # Compute product of input_vec[i]^a_vec[i] for all i
        return np.prod(np.power(input_vec, self.a_vec))


# Class for generating and evaluating a basis of monomials
class Monomial_Basis:
    """Class for creating and evaluating a basis of monomials"""

    # Constructor to initialize the monomial basis
    def __init__(self, chi, dimension):
        # Get all permutations of exponents that sum to chi
        permutations = Monomial_Basis._get_all_permutations(chi, dimension)
        # Create a list of Monomial objects from the permutations
        self.monomials = [Monomial(x) for x in permutations]

    # Static method to generate all exponent permutations
    @staticmethod
    def _get_all_permutations(chi, dimension):
        # Base case: chi=0, all exponents are 0
        if chi == 0:
            # Return a list with a single zero vector of given dimension
            return [[0] * dimension]
        # Base case: dimension=1, all exponents from 0 to chi
        elif dimension == 1:
            # Return list of single-element lists [0], [1], ..., [chi]
            return [[i] for i in range(chi + 1)]
        # Recursive case: generate permutations
        else:
            # Initialize results list
            results = []
            # For each possible exponent for the first dimension
            for i in range(chi + 1):
                # Recursively get permutations for remaining dimensions
                # Append i to each permutation from the recursive call
                results += [
                    [i] + x
                    for x in Monomial_Basis._get_all_permutations(
                        chi - i, dimension - 1
                    )
                ]
            return results

    # Evaluate all monomials in the basis at a given input vector
    def evaluate(self, X):
        # Evaluate each monomial and return as numpy array
        return np.array([m.evaluate(X) for m in self.monomials])


# Class for performing polynomial regression
class Regression:
    """Class for performing polynomial regression"""

    # Constructor to initialize regression with data
    def __init__(
        self, X_mat, Y, chi=2, payoff_func=lambda x: np.max(np.sum(x) - 100, 0)
    ):
        # Ensure X_mat is a 2D matrix
        assert len(X_mat.shape) == 2, "X in the regression should be a 2d matrix"
        # Get dimension from input data
        self.dimension = len(X_mat[0])
        # Create monomial basis of degree chi
        self.basis = Monomial_Basis(chi, self.dimension)

        # Find indices where payoff function is positive (in-the-money paths)
        self.index = np.array(
            [i for i in range(len(X_mat)) if payoff_func(X_mat[i]) > 0]
        )

        # Check if any paths are in-the-money
        self.has_intrinsic_value = np.any(self.index)
        # If no in-the-money paths, return early
        if not self.has_intrinsic_value:
            return

        # Extract in-the-money samples
        target_X, target_Y = X_mat[self.index], Y[self.index]

        # Create design matrix for regression
        target_matrix_A = np.array([self.basis.evaluate(x) for x in target_X])
        # Solve least squares regression
        self.coefficients = np.linalg.lstsq(target_matrix_A, target_Y, rcond=None)[0]

    # Evaluate the regression at a new input point
    def evaluate(self, X):
        """
        Evaluate the regression at a new input point X

        Parameters:
        X: a numpy array of input data (e.g., asset prices)
        """
        # Check if regression was initialized with in-the-money paths
        if not self.has_intrinsic_value:
            raise RuntimeError("Least square failed due to ineiligible input")
        # Verify input dimension matches expected dimension
        assert (
            len(X) == self.dimension
        ), "input vector X doesn't meet the regression dimension"
        # Evaluate all monomials in the basis at X
        monomial_terms = self.basis.evaluate(X)
        # Compute weighted sum: coefficients * monomial_terms
        return np.sum(np.multiply(self.coefficients, monomial_terms))
