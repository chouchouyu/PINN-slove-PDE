import numpy as np


class Monomial_Basis:

    def __init__(self, chi, dimension):
        self.chi = chi
        self.dimension = dimension
        self.exponent_vectors = np.array(self._get_all_permutations(chi, dimension))

    @staticmethod
    def _get_all_permutations(chi, dimension):
        if chi == 0:
            return [[0] * dimension]
        elif dimension == 1:
            return [[i] for i in range(chi + 1)]
        else:
            results = []
            for i in range(chi + 1):
                sub_perms = Monomial_Basis._get_all_permutations(chi - i, dimension - 1)
                for sub in sub_perms:
                    results.append([i] + sub)
            return results

    def evaluate(self, X):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        exponents = self.exponent_vectors
        n_samples = X.shape[0]
        n_monomials = exponents.shape[0]

        if self.dimension <= 5:
            result = np.zeros((n_samples, n_monomials))
            for i in range(n_samples):
                x = X[i]
                for j in range(n_monomials):
                    result[i, j] = np.prod(x ** exponents[j])
        else:
            result = np.exp(np.dot(np.log(X), exponents.T))
        return result


class Regression:
    def __init__(self, X, Y, payoff_func, method='fast'):
        self.X = X
        self.Y = Y
        self.payoff_func = payoff_func

        self.has_intrinsic_value = False
        self.index = None

        self._fit()

    def _fit(self):
        X = self.X
        Y = self.Y
        payoff_func = self.payoff_func

        if X.ndim == 3:
            X = X[:, :, 0]

        X_in = X
        Y_in = Y

        if len(Y_in) != len(X_in):
            min_len = min(len(Y_in), len(X_in))
            X_in = X_in[:min_len]
            Y_in = Y_in[:min_len]

        self.dimension = len(X_in[0])
        self.basis = Monomial_Basis(2, self.dimension)

        self.index = np.array([i for i in range(len(X_in)) if payoff_func(X_in[i]) > 0])

        self.has_intrinsic_value = np.any(self.index)
        if not self.has_intrinsic_value:
            return

        target_X, target_Y = X_in[self.index], Y_in[self.index]

        target_matrix_A = self.basis.evaluate(target_X)
        self.coefficients = np.linalg.lstsq(target_matrix_A, target_Y, rcond=None)[0]

    def evaluate(self, X):
        if not self.has_intrinsic_value:
            raise RuntimeError("Least square failed due to ineiligible input")
        assert len(X) == self.dimension, "input vector X doesn't meet the regression dimension"
        monomial_terms = self.basis.evaluate(X)
        return np.sum(np.multiply(self.coefficients, monomial_terms))
