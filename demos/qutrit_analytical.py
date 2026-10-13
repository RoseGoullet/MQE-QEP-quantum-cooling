import numpy as np
from typing import Callable

"""
Contains various functions for calculating costs for qutrit swaps of thermal machines, based on analytically derived formulae
"""

def case1_first_swap(alpha_goal: float, E1: float, E2: float, temp: float = 10) -> tuple[float, float]:
    beta = 1 / temp
    alpha_diff = (2 * alpha_goal - 1)
    q1 = np.exp(-beta * E1)
    q2 = np.exp(-beta * E2)
    z = 1 + q1 + q2
    t = z * alpha_diff / (1 - q1)
    cost = E1 / 2 * alpha_diff
    return t, cost


def case1_first_analytical(alpha_goal: float, temp: float = 10) -> tuple[float, float]:
    return np.log(alpha_goal / (1 - alpha_goal)) * temp, np.nan


def case1_second_swap(alpha_goal: float, E1: float, E2: float, temp: float = 10) -> tuple[float, float]:
    beta = 1 / temp
    q1 = np.exp(-beta * E1)
    q2 = np.exp(-beta * E2)
    z = 1 + q1 + q2
    g2 = E2 - E1
    t = (2 * z * alpha_goal - 2 - q2) / (q1 - q2)
    cost = 1 / (2 * z) * (E1 * (1 - q1) + g2 * (2 * z * alpha_goal - 2 - q2))
    return t, cost


def case2_first_swap(alpha_goal: float, E1: float, E2: float, temp: float = 10) -> tuple[float, float]:
    beta = 1 / temp
    alpha_diff = (2 * alpha_goal - 1)
    q1 = np.exp(-beta * E1)
    q2 = np.exp(-beta * E2)
    z = 1 + q1 + q2
    g2 = E2 - E1
    t = z * alpha_diff / (q1 - q2)
    cost = g2 / 2 * alpha_diff
    return t, cost


def case2_first_analytical(alpha_goal: float, temp: float = 10) -> tuple[float, float]:
    return 0, np.log(2 * alpha_goal / (3 - 4 * alpha_goal)) * temp


def case2_second_swap(alpha_goal: float, E1: float, E2: float, temp: float = 10) -> tuple[float, float]:
    beta = 1 / temp
    q1 = np.exp(-beta * E1)
    q2 = np.exp(-beta * E2)
    z = 1 + q1 + q2
    g2 = E2 - E1
    t = (2 * z * alpha_goal - 1 - 2 * q1) / (1 - q1)
    cost = E1 * alpha_goal + 1 / (2 * z) * (E1 * (-1 - 2 * q1) + g2 * (q1 - q2))
    return t, cost


def scan_gaps(alpha_goal: float, gap_func: Callable[[float, float, float, float], tuple[float, float]], iters: int = 100, temp: float = 10, gap_min=0.01, gap_max=1000) -> list[
    tuple[float, float, float]]:
    energies = []
    for gap in np.geomspace(gap_min, gap_max, iters):
        for gap2 in np.geomspace(gap_min, gap_max, iters):
            t, cost = gap_func(alpha_goal, gap, gap + gap2, temp)
            if t > 1 or t < 0:
                continue
            energies.append((gap, gap2, cost, t))
    return energies
