import numpy as np

from Paths_generater import Paths_generater
from Regression import Regression


# Class for pricing American options using the Longstaff-Schwartz algorithm
class American_Option:
    # Constructor to initialize the American option with a path generator and payoff function
    def __init__(self, paths_generater: Paths_generater, payoff_func):
        # Store the path generator instance
        self.paths_generater = paths_generater
        # Store the payoff function
        self.payoff_func = payoff_func

    # Main method to price the American option
    def price(self, n_simulations=1000):
        # Generate simulated price paths
        self.simulations_paths = self.paths_generater.gbm(n_simulations=n_simulations)
        # Initialize cash flow matrix: n_simulations x (days+1)
        cash_flow = np.zeros([n_simulations, self.paths_generater.days + 1])

        # Backward induction: from maturity to time 1
        for t in range(self.paths_generater.days, 0, -1):
            # Get prices at time t for all simulations
            prices_at_t = self.simulations_paths[:, :, t]
            # If it's the maturity date
            if t == self.paths_generater.days:
                # Get prices at maturity
                maturity_price = prices_at_t
                # Calculate payoff at maturity
                maturity_payoff = np.array(list(map(self.payoff_func, maturity_price)))
                # Store payoff in cash flow matrix
                cash_flow[:, -1] = maturity_payoff
            # For earlier time steps
            else:
                # Get discounted future cash flows (continuation value)
                discounted_cashflow = self._get_discounted_cashflow(
                    t, cash_flow, n_simulations
                )
                # Perform regression to estimate continuation value
                r = Regression(
                    prices_at_t, discounted_cashflow, payoff_func=self.payoff_func
                )
                # Check if there are in-the-money paths
                if r.has_intrinsic_value:
                    # Calculate current payoff for all paths
                    all_cur_payoff = np.array(list(map(self.payoff_func, prices_at_t)))
                    # Calculate continuation value for in-the-money paths
                    continuation_value = np.array(
                        [r.evaluate(X) for X in prices_at_t[r.index]]
                    )
                    # Find paths where immediate exercise is optimal
                    exercise_index = r.index[
                        all_cur_payoff[r.index] >= continuation_value
                    ]
                    # If there are optimal exercise paths
                    if len(exercise_index) > 0:
                        # Set future cash flows to zero for exercised paths
                        cash_flow[exercise_index] = np.zeros(
                            cash_flow[exercise_index].shape
                        )
                        # Record exercise payoff at time t
                        cash_flow[exercise_index, t] = all_cur_payoff[exercise_index]
        # Calculate discounted value at time 0
        return self._get_discounted_cashflow_at_t0(cash_flow)

    # Method to calculate discounted future cash flows
    def _get_discounted_cashflow(
        self, t: int, cashflow_matrix: np.ndarray, n_paths
    ) -> np.ndarray:
        """
        Corrected version: uses the correct number of time steps N (self.paths_generater.days)
        """
        # Number of time steps
        N = self.paths_generater.days
        # Array of time indices from 0 to N
        time_indices = np.arange(N + 1)
        # Calculate discount factors from time t to each time point
        discount_factors = np.exp(
            (t - time_indices) * self.paths_generater.dt * self.paths_generater.r
        )

        # Create mask for cash flows after time t
        mask = np.zeros_like(cashflow_matrix, dtype=bool)
        mask[:, t + 1 : N + 1] = cashflow_matrix[:, t + 1 : N + 1] != 0

        # Find the first non-zero cash flow for each path (working backwards)
        reversed_mask = np.fliplr(mask)
        reversed_indices = np.argmax(reversed_mask, axis=1)
        first_nonzero_indices = mask.shape[1] - reversed_indices - 1

        # Check which paths have non-zero cash flows
        has_nonzero = np.any(mask, axis=1)

        # Initialize result array
        result = np.zeros(n_paths)
        # Calculate discounted cash flow for paths with non-zero cash flows
        result[has_nonzero] = (
            cashflow_matrix[has_nonzero, first_nonzero_indices[has_nonzero]]
            * discount_factors[first_nonzero_indices[has_nonzero]]
        )
        return result

    # Method to calculate discounted value at time 0
    def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        """
        Implementation consistent with American.py

        Parameters:
        - cashflow_matrix: cash flow matrix

        Returns: average discounted value
        """
        # Extract cash flows from time 1 onwards (consistent with American.py)
        future_cashflows = cashflow_matrix[:, 1:]

        # Find the first non-zero cash flow position for each path
        first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)

        # Calculate corresponding time indices (starting from 1)
        time_indices = first_nonzero_positions + 1

        # Calculate discount factors
        discount_factors = np.exp(
            -self.paths_generater.r * time_indices * self.paths_generater.dt
        )

        # Get corresponding cash flows and discount them
        discounted_values = (
            future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions]
            * discount_factors
        )

        # Average over all paths, including those with no cash flows (consistent with American.py)
        return discounted_values.mean()
