from dataclasses import dataclass, field
from typing import NamedTuple
from .helpers import Energy, Coefficient


@dataclass(order=True)
class State:
    """
    A class representing a possible state of a quantum system, with an index (i.e. which quantum level is it) and an associated energy (i.e. what is the energy of the system if it is at this level/state).
    """
    index: int
    energy: Energy = field(compare=False)

    def __post_init__(self):
        if self.index < 0:
            raise ValueError(f"Index must be positive, got {self.index}")

    @property
    def is_ground_state(self) -> bool:
        """
        :return: whether the state is the ground state for its system
        """
        return self.index == 0

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.index == other.index and self.energy == other.energy

    def __str__(self):
        return f"|{self.index}>, E={self.energy:.2f}"

    def __hash__(self):
        return hash((self.index, self.energy))


@dataclass
class CombinedState:
    """
    A class representing a possible combined state of two quantum systems (one "system" one "machine"), with two indices and a combined energy.
    """
    system_index: int
    machine_index: int
    system_energy: Energy
    machine_energy: Energy

    @property
    def energy(self) -> Energy:
        return self.system_energy + self.machine_energy

    @property
    def is_ground_state(self) -> bool:
        """
        :return: whether the state is a ground state for its "system" subsystem
        """
        return self.system_index == 0

    def trace_out_system(self) -> State:
        """
        :return: The uncombined state for the "machine" subsystem, obtained by tracing out the "system" subsystem's part
        """
        return State(index=self.machine_index, energy=self.machine_energy)

    def trace_out_machine(self) -> State:
        """
        :return: The uncombined state for the "system" subsystem, obtained by tracing out the "machine" subsystem's part
        """
        return State(index=self.system_index, energy=self.system_energy)

    @classmethod
    def from_states(cls, system_state: State, machine_state: State):
        """
        Create a combined state from a system state and a machine state
        :param system_state:
        :param machine_state:
        :return:
        """
        return cls(
            system_index=system_state.index,
            machine_index=machine_state.index,
            system_energy=system_state.energy,
            machine_energy=machine_state.energy
        )

    def __str__(self):
        return f"|{self.system_index}{self.machine_index}>, E={self.energy:.2f}"


class StateWithCoeff(NamedTuple):
    """
    A structure associating a State with a coefficient/population percentage.

    Useful to represent superpositions with specific mixes of states.
    """
    state: State
    coeff: Coefficient

    def __str__(self):
        return f"{self.coeff:.2f} * {self.state}"


class CombinedStateWithCoeff(NamedTuple):
    """
    A structure associating a CombinedState with a coefficient/population percentage.

    Useful to represent superpositions with specific mixes of combined states.
    """
    state: CombinedState
    coeff: Coefficient

    @property
    def target_function(self) -> float:
        """
        :return: The value of the target function for this combined state (in our case, it is simply whether it is a ground state or not)
        """
        return 1 if self.is_ground_state else 0

    @property
    def cost_function(self) -> float:
        """
        :return: The value of the target function for this combined state (in our case, it is simply the energy of the quantum levels)
        """
        return self.state.energy

    @property
    def is_ground_state(self) -> bool:
        return self.state.is_ground_state

    @staticmethod
    def gradient(k: CombinedStateWithCoeff, l: CombinedStateWithCoeff) -> float:
        """
        The gradient between two combined states, as computed in https://arxiv.org/abs/2412.07291, Section II.A
        :param k: A combined state and its population percentage, as a candidate to be swapped with `l`, if doing so would increase the overall target value of the global system
        :param l: A combined state and its population percentage, as a candidate to be swapped with `k`, if doing so would increase the overall target value of the global system
        :return: infinity if it would not be a valid swap (i.e. `k` has less population than `l`, or `k` is a ground state while `l` is not), the gradient otherwise
        """
        if k.target_function > l.target_function and k.coeff < l.coeff:
            return (k.cost_function - l.cost_function) / (k.target_function - l.target_function)
        import math
        return math.inf

    def __str__(self):
        return f"{self.coeff:.2f} * {self.state}"
