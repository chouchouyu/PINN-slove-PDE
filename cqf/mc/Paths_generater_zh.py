import numpy as np


class Paths_generater:
    def __init__(self, T, days, S0_vec, r: float, vol_vec, dividend_vec, corr_mat):
        assert (
            len(S0_vec)
            == len(vol_vec)
            == len(dividend_vec)
            == corr_mat.shape[0]
            == corr_mat.shape[1]
        ), "Vectors' lengths are different"
        self.dt = T / days
        self.T = T
        self.days = days
        self.S0_vec = S0_vec
        self.r = r
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat
        self.asset_num = len(S0_vec)

    def gbm(self, n_simulations: int) -> np.ndarray:
        price_paths = []  # np.zeros((n_simulations, self.asset_num, self.days + 1))
        L = np.linalg.cholesky(self.corr_mat)
        drift_vec = self.r - self.dividend_vec
        for _ in range(n_simulations):
            price_path = np.zeros((self.asset_num, self.days + 1))
            price_path[:, 0] = self.S0_vec
            for t in range(1, self.days + 1):
                dW = L.dot(np.random.normal(size=self.asset_num)) * np.sqrt(self.dt)
                rand_term = np.multiply(self.vol_vec, dW)
                price_path[:, t] = np.multiply(
                    price_path[:, t - 1],
                    np.exp((drift_vec - self.vol_vec**2 / 2) * self.dt + rand_term),
                )
            price_paths.append(price_path)

        return np.array(price_paths)
