import numpy as np
from typing import Callable

"""
Contains various functions for calculating costs for ququart swaps of thermal machines, based on analytically derived formulae
"""

def full_swap(alpha_goal: float, E1: float, E2: float, E3: float, temp: float = 10) -> tuple[float, float]:
    beta = 1 / temp
    alpha_diff = (2 * alpha_goal - 1)
    q1 = np.exp(-beta * E1)
    q2 = np.exp(-beta * E2)
    q3 = np.exp(-beta * E3)
    z = 1 + q1 + q2 + q3
    alpha = (1 + q1) / z
    t = 1 if alpha_goal < alpha else np.inf
    cost = (E1 + q1 * (-2 * E1 + E2 + E3) + q2 * (-2 * E2 + E3 + E1) + q3 * (-E3 + E2)) / (2 * z)
    return t, cost


def scan_gaps(alpha_goal: float, gap_func: Callable[[float, float, float, float, float], tuple[float, float]], iters: int = 100, temp: float = 10, gap_min=0.01, gap_max=1000) -> list[
    tuple[float, float, float, float, float]]:
    energies = []
    for gap in np.geomspace(gap_min, gap_max, iters):
        for gap2 in np.geomspace(gap_min, gap_max, iters):
            for gap3 in np.geomspace(gap_min, gap_max, iters):
                t, cost = gap_func(alpha_goal, gap, gap + gap2, gap + gap2 + gap3, temp)
                if t > 1 or t < 0:
                    continue
                energies.append((gap, gap2, gap3, cost, t))
    return energies


def scan_analytical(alpha_goal, gap_func, iters=150, gap_min=0.1, gap_max=500):
    energies = scan_gaps(alpha_goal, gap_func, iters=iters, gap_min=gap_min, gap_max=gap_max)
    if len(energies) == 0:
        return alpha_goal, np.nan, np.nan, np.nan, np.nan
    _, _, _, e_values, _ = zip(*energies)
    min_gaps = energies[np.argmin(e_values)]
    return alpha_goal, *min_gaps