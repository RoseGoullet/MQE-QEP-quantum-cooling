import numpy as np
from collections.abc import Iterable

type Coefficient = float
type Energy = float


def _is_probability_distribution(coeffs: Iterable[Coefficient]) -> bool:
    """
    Checks whether the given coefficients form a valid probability distribution (no negative coeffs, and sum to 1).
    :param coeffs: The coefficients to validate.
    :return: True if the coefficients form a probability distribution, False otherwise.
    """
    for coeff in coeffs:
        if coeff < 0 or coeff > 1:
            return False
    return np.isclose(sum(coeffs), 1)
