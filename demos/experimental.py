from polytope_cooling.systems import SingleSystem, CombinedSystem
import numpy as np


def get_step_data_gaps(gaps: list[float], alpha_goal: float = 1, temp: float = 10) -> tuple[float, float, list[CombinedSystem]]:
    """
    Tries to reach a given alpha from a maximally mixed qubit system, and a thermal machine with given gaps and temperature
    :returns: a tuple with: the final alpha value reached, the energy diff of the machine subsystem between the final state of the system and the original one, and the system after each swap done
    """
    system = SingleSystem.maximally_mixed([0, 1])
    machine = SingleSystem.thermal_system_from_temperature_and_gaps(gaps, temp)
    combined = CombinedSystem.from_systems(system, machine)
    steps = list(combined.construct_path(alpha_goal))
    return steps[-1].alpha, steps[-1].trace_out_system().energy - combined.trace_out_system().energy, steps


def get_step_data_levels(levels: list[float], alpha_goal: float = 1, temp: float = 10) -> tuple[float, float, list[CombinedSystem]]:
    """
    Tries to reach a given alpha from a maximally mixed qubit system, and a thermal machine with given levels (not gaps) and temperature
    :returns: a tuple with: the final alpha value reached, the energy diff of the machine subsystem between the final state of the system and the original one, and the system after each swap done
    """
    system = SingleSystem.maximally_mixed([0, 1])
    machine = SingleSystem.thermal_system_from_temperature_and_levels(levels, temp)
    combined = CombinedSystem.from_systems(system, machine)
    steps = list(combined.construct_path(alpha_goal))
    return steps[-1].alpha, steps[-1].trace_out_system().energy - combined.trace_out_system().energy, steps


def _scan_gaps_internal(gaps: list[float], dims: float, alpha_goal: float, iters: int = 100, temp: float = 10, gap_min=0.01, gap_max=1000) -> list[tuple[float, ...]]:
    energies = []
    for gap in np.geomspace(gap_min, gap_max, iters):
        new_gaps = gaps + [gap]
        if dims <= 1:
            alpha, e, steps = get_step_data_gaps(new_gaps, alpha_goal=alpha_goal, temp=temp)
            if alpha < alpha_goal:
                continue
            energies.append((*new_gaps, e, len(steps)))
        else:
            energies.extend(_scan_gaps_internal(new_gaps, dims - 1, alpha_goal, iters, temp, gap_min, gap_max))
    return energies


def scan_gaps(dims, alpha_goal: float, iters: int = 100, temp: float = 10, gap_min=0.01, gap_max=1000) -> list[tuple[float, ...]]:
    """
    Finds the costs for a gibbs machine to reach a given alpha goal, for various gap setups
    :param dims: the dimensions of our machine
    :param alpha_goal: the alpha we want to reach
    :param iters: how many iterations per dimension (i.e. how fine-grained our search is), will be spaced between the min gap and the max gap in a logarithmic space
    :param temp: the temperature of our thermal machine
    :param gap_min: the minimum gap we want to consider for each state
    :param gap_max: the maximum gap we want to consider for each state
    :return: for each gap setup that managed to reach the alpha goal, a tuple (gap1, gap2, ..., gapN, cost, steps needed to reach the target alpha)
    """
    return _scan_gaps_internal([], dims - 1, alpha_goal, iters, temp, gap_min, gap_max)
