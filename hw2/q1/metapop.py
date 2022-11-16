from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os


class MetaPopModel(object):
    """
    Class for Metapopulation model
    """

    def __init__(
        self,
        net_params: np.ndarray, # A 3x3 matrix of alpha_ij
        beta: float,
        gamma: float,
        pop_size: np.ndarray,
    ) -> None:
        self.net_params = net_params
        self.beta = beta
        self.gamma = gamma
        self.pop_size = pop_size
        self.communities = pop_size.shape[0]

    def ode(self, times, init, parms):
        """Metapopulation model ODE"""

        init_ = init.reshape((self.communities, 3))

        self.pop_size = init_.sum(axis=1)
        S, I, R = init_[:, 0], init_[:, 1], init_[:, 2]
        # S,I,R are now 1D arrays of length self.communities
        beta, gamma = parms

        S_eff = None
        I_eff = None
        R_eff = None
        print(S)
        ############################################################
        # Compute effective S, I, R from S, I, R, pop_size and net_params
        # YOUR CODE HERE
        inS = (S * (self.net_params / self.pop_size)).sum(axis=0)
        outS = (S * (self.net_params / self.pop_size)).sum(axis=1)
        inI = (I * (self.net_params / self.pop_size)).sum(axis=0)
        outI = (I * (self.net_params / self.pop_size)).sum(axis=1)
        inR = (R * (self.net_params / self.pop_size)).sum(axis=0)
        outR = (R * (self.net_params / self.pop_size)).sum(axis=1)

        S_eff = S + inS - outS
        I_eff = I + inI - outI
        R_eff = R + inR - outR
        ############################################################

        dSdt = np.outer(S_eff, -beta * I_eff / self.pop_size).sum(axis=1)
        dIdt = -dSdt - gamma * I_eff
        dRdt = gamma * I_eff

        return np.array([dSdt, dIdt, dRdt]).T.ravel()

    def solve(self, init: List[float], parms: List[float], times: np.ndarray):
        """Solve Metapopulation model"""
        sol = solve_ivp(
            lambda t, y: self.ode(t, y, parms),
            (times[0], times[-1]),
            init,
            t_eval=times,
        )
        return sol.y

    def plot_soln(
        self,
        init: List[float],
        parms: List[float],
        times: np.ndarray,
        save_path: Optional[str] = None,
    ):
        sol = self.solve(init, parms, times)
        sol = sol.reshape((self.communities, 3, -1))
        for i, s in enumerate(sol):
            save_path_pop = f"{save_path}_{i}.png" if save_path else None
            self.plot(s[0], s[1], s[2], save_path_pop)

    def plot(self, s, i, r, save_path: Optional[str] = None):
        """Plot Metapopulation model"""
        plt.clf()
        plt.plot(s, label="S")
        plt.plot(i, label="I")
        plt.plot(r, label="R")
        plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Population fraction")
        if save_path:
            # Make dir if no exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    pops = np.array([1000.0, 200.0, 300.0])
    params = [0.5, 0.3]
    net_params = (
        np.array([[0.0, 0.51, 0.10], [0.02, 0.0, 0.10], [0.02, 0.01, 0.0]]) * pops
    )

    init = np.array([[900.0, 100.0, 0.0], [100.0, 10.0, 90.0], [100.0, 50.0, 150.0]])

    model = MetaPopModel(net_params, params[0], params[1], np.array(pops))
    model.plot_soln(init.ravel(), params, np.linspace(0, 100, 1000), "plots/metapop")
    # soln = model.solve(init.ravel(), params, np.linspace(0, 100, 1000))
