from typing import Any
from collections.abc import Generator, Iterable, Sequence

import itertools

from .states import State, CombinedState, StateWithCoeff, CombinedStateWithCoeff
from .helpers import Energy, Coefficient, _is_probability_distribution


class SingleSystem:
    """
    A class representing a quantum system of set dimension, consisting of one or more states and their associated coefficients/populations.
    """
    states: list[State]
    coeffs: list[Coefficient]
    dim: int

    def ground_state(self) -> State:
        return min(self.states)

    @property
    def states_with_coeffs(self) -> list[StateWithCoeff]:
        """
        :return: the individual states with their associated coefficients, as a list of StateWithCoeff objects
        """
        return [StateWithCoeff(state, coeff) for state, coeff in zip(self.states, self.coeffs)]

    @property
    def alpha(self):
        """
        :return: the overall alpha/target value of the system (in our case, how much of the system is in the ground state)
        """
        return sum(state.coeff if state.state.is_ground_state else 0 for state in self.states_with_coeffs)

    @property
    def energy(self):
        """
        :return: How much energy is in the system, i.e. the weighted-by-population average of the energies of all states
        """
        return sum(state.coeff * state.state.energy for state in self.states_with_coeffs)

    @classmethod
    def from_energies_and_coeffs(cls, energies: Sequence[Energy], coeffs: Sequence[Coefficient]) -> SingleSystem:
        """
        Build a SingleSystem from a list of energies and a list of coefficients, inferring the state indices from the positions in the list.

        e.g. passing [1,2,1] and [1,2,3] will give you a system with states and associated energies: |0> (population 1/6, energy 1),  |1> (population 1/3, energy 2),  |2> (population 1/2, energy 1)
        """
        states = [State(i, energy) for i, energy in enumerate(energies)]
        return cls(states=states, coeffs=coeffs)

    @classmethod
    def from_traced_states_and_coeffs(cls, traced_states_and_coeffs: Iterable[StateWithCoeff]) -> SingleSystem:
        """
        Build a SingleSystem from a list of StateWithCoeff, consolidating the populations of states with the same index.

        e.g. passing [StateWithCoeff(idx:0, coeff: 0.2), StateWithCoeff(idx:0, coeff: 0.6), StateWithCoeff(idx:1, coeff: 0.2)]
        will give you a system with states: |0> (population 0.8),  |1> (population 0.2)
        """
        dic: dict[State, Coefficient] = {}
        for traced_state, coeff in traced_states_and_coeffs:
            dic.setdefault(traced_state, 0)
            dic[traced_state] += coeff
        states, coeffs = zip(*dic.items())
        return SingleSystem(states, coeffs)

    @classmethod
    def thermal_system_from_temperature_and_gaps(cls, gaps: Sequence[float], temp: float = 300):
        """
        Creates a system in a Gibbs distribution for a given temperature and list of energy gaps (i.e. energy level difference with the previous level, assuming the energy level of the first state is 0)
        """
        import numpy as np, itertools as it
        return cls.thermal_system_from_temperature_and_levels(levels=np.cumsum(list(it.chain([0.], gaps))), temp=temp)

    @classmethod
    def thermal_system_from_temperature_and_levels(cls, levels: Sequence[float], temp: float = 300):
        """
        Creates a system in a Gibbs distribution for a given temperature and list of energy levels
        """
        import numpy as np
        beta = 1 / temp
        levels = np.array(levels)
        z: float = sum(np.exp(-beta * levels))
        coeffs = np.exp(-beta * levels) / z
        return cls.from_energies_and_coeffs(levels, coeffs)

    @classmethod
    def maximally_mixed(cls, energies: Sequence[Energy]) -> SingleSystem:
        """
        Creates a maximally mixed system with given energy levels
        """
        return cls.from_energies_and_coeffs(energies, coeffs=[1 / len(energies) for _ in range(len(energies))])

    def __init__(self, states: Sequence[State], coeffs: Sequence[Coefficient]):
        assert len(states) == len(coeffs), f"number of states ({len(states)}) do not match number of coeffs ({len(coeffs)})"
        if not _is_probability_distribution(coeffs):
            import numpy as np
            coeffs = [float(x) for x in (np.array(coeffs) / sum(coeffs))]
        assert len(states) == len({state.index for state in states}), f"Indices are not unique: {[state.index for state in states]}"
        self.states = list(states)
        self.coeffs = list(coeffs)
        self.dim = len(self.states)


class CombinedSystem:
    """
    A class representing a combination of two quantum subsystems, one "system" one "machine", in a specific superposition of combined states.
    """
    states_with_coeffs: list[CombinedStateWithCoeff]
    _dims: tuple[int, int]

    @property
    def states(self) -> list[CombinedState]:
        return [state.state for state in self.states_with_coeffs]

    @property
    def coeffs(self) -> list[Coefficient]:
        return [state.coeff for state in self.states_with_coeffs]

    @property
    def dims(self):
        return self._dims

    @property
    def alpha(self):
        """
        :return: the overall alpha/target value of the system (in our case, how much of the system is in the ground state of the "system" subsystem)
        """
        return sum(state.coeff if state.state.is_ground_state else 0 for state in self.states_with_coeffs)

    @property
    def energy(self):
        """
        :return: How much energy is in the system, i.e. the weighted-by-population average of the energies of all combined states
        """
        return sum(state.coeff * state.state.energy for state in self.states_with_coeffs)

    def __init__(self, states_with_coeffs: Sequence[CombinedStateWithCoeff], dims: tuple[int, int]):
        assert len(states_with_coeffs) == dims[0] * dims[1], f"number of states ({len(states_with_coeffs)}) do not match dims {dims}"
        coeffs = [coeff for _, coeff in states_with_coeffs]
        assert _is_probability_distribution(coeffs), f"{coeffs} is not a valid probability distribution"
        self.states_with_coeffs = list(states_with_coeffs)
        self.states_with_coeffs.sort(key=lambda state: (state.coeff, state.state.is_ground_state, -state.state.energy), reverse=True)
        self._dims = dims

    @classmethod
    def from_states_and_coeffs(cls, states: Sequence[CombinedState], coeffs: Sequence[Coefficient], dims: tuple[int, int]) -> CombinedSystem:
        """
        Build a CombinedSystem from a list of combined states and a list of coefficients
        """
        assert len(states) == dims[0] * dims[1], f"number of states ({len(states)}) do not match dims {dims}"
        return cls(states_with_coeffs=[CombinedStateWithCoeff(state, coeff) for state, coeff in zip(states, coeffs)], dims=dims)

    @classmethod
    def from_systems(cls, system: SingleSystem, machine: SingleSystem) -> CombinedSystem:
        """
        Build a CombinedSystem from two subsystems
        """
        states_with_coeffs = [CombinedStateWithCoeff(CombinedState.from_states(system_state=system_state.state, machine_state=machine_state.state), system_state.coeff * machine_state.coeff)
                              for system_state in system.states_with_coeffs for machine_state in machine.states_with_coeffs]
        return cls(states_with_coeffs, (system.dim, machine.dim))

    def ground_states(self) -> Iterable[CombinedStateWithCoeff]:
        """
        :return: a list of all the combined states that are ground states of the "system" subsystem, with associated populations
        """
        return (state_with_coeff for state_with_coeff in self.states_with_coeffs if state_with_coeff.state.is_ground_state)

    def min_alpha(self) -> CombinedSystem:
        """
        :return: The combined state associated with this one that, through only swaps of population values, has the lowest alpha/target value possible (and the lowest energy cost as a tiebreaker)
        """
        #TODO: inplace?
        #TODO: handle degenerate cases
        ground_states_first_then_energy = sorted(self.states, key=lambda state: (-state.is_ground_state, -state.energy))

        return CombinedSystem.from_states_and_coeffs(states=ground_states_first_then_energy, coeffs=sorted(self.coeffs), dims=self.dims)

    def trace_out_system(self) -> SingleSystem:
        """
        :return: the SingleSystem corresponding to the "machine" subsystem
        """
        traced_states = [StateWithCoeff(state.trace_out_system(), coeff) for state, coeff in self.states_with_coeffs]
        return SingleSystem.from_traced_states_and_coeffs(traced_states)

    def trace_out_machine(self) -> SingleSystem:
        """
        :return: the SingleSystem corresponding to the "system" subsystem
        """
        traced_states = [StateWithCoeff(state.trace_out_machine(), coeff) for state, coeff in self.states_with_coeffs]
        return SingleSystem.from_traced_states_and_coeffs(traced_states)

    def swap_state_with_next(self, index: int) -> CombinedSystem:
        """
        :return: the combined system corresponding to this system, but with its state at `index`'s population completely swapped with the next state's
        """
        return self.partial_swap_state_with_next(index, part=1)

    def partial_swap_state_with_next(self, index: int, part: float) -> CombinedSystem:
        """
        :return: the combined system corresponding to this system, but with its state at `index`'s population partially (to a factor of `part`) swapped with the next state's.
        e.g. part=1 corresponds to a full swap, part=0 to doing nothing, part=0.5 to setting both populations to be the average of the two.
        """
        states = self.states
        coeffs = self.coeffs
        a, b = coeffs[index + 1], coeffs[index]
        coeffs[index], coeffs[index + 1] = part * a + (1 - part) * b, part * b + (1 - part) * a
        return CombinedSystem.from_states_and_coeffs(states=states, coeffs=coeffs, dims=self.dims)

    def _next_step(self) -> tuple[CombinedSystem | None, int]:
        import math
        best: tuple[int, Energy] = (-1, math.inf)
        for i, states_with_coeff in enumerate(itertools.pairwise(self.states_with_coeffs)):
            gradient = CombinedStateWithCoeff.gradient(states_with_coeff[1], states_with_coeff[0])
            if gradient < best[1]:
                best = (i, gradient)
        # swap the two best states
        return (None, -1) if best[0] < 0 else (self.swap_state_with_next(best[0]), best[0])

    def next_step(self) -> CombinedSystem | None:
        """
        :return: A combined state with the next swap (as determined by https://arxiv.org/abs/2412.07291) fully done, or None if no more swaps are possible to increase target value
        """
        return self._next_step()[0]

    def next_step_or_goal(self, alpha_goal: float) -> CombinedSystem | None:
        """
        :return: A combined state with the next swap (as determined by https://arxiv.org/abs/2412.07291) fully done if it is not sufficient to reach a target value of `alpha_goal`,
         or A combined state with the next swap partially done such as to reach the goal exactly, or None if no more swaps are possible to increase target value
        """
        next_step, index = self._next_step()
        if next_step is None or next_step.alpha < alpha_goal:
            return next_step
        part = (alpha_goal - self.alpha) / (next_step.alpha - self.alpha)
        return self.partial_swap_state_with_next(index, part)

    def construct_path(self, alpha_goal: float = 1) -> Generator[CombinedSystem, Any, None]:
        """
        :return: A sequence of combined systems corresponding to the intermediary states after each swap used to reach `alpha_goal`/maximum possible alpha
        """
        curr = self
        yield curr
        next_step = curr.next_step()
        while next_step is not None:
            if next_step.alpha >= alpha_goal:
                yield next_step
                return
            curr = next_step
            yield next_step
            next_step = curr.next_step_or_goal(alpha_goal)
