from collections.abc import Collection
from typing import TypeVar

from exceptions import InvalidParameterError

ChoiceT = TypeVar("ChoiceT")

def require_int(name: str, value: object, minimum: int | None = None, maximum: int | None = None ) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameterError(f"{name} must be an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise InvalidParameterError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise InvalidParameterError(f"{name} must be at most {maximum}, got {value}")
    return value

def _require_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidParameterError(f"{name} must be a number, got {value!r}")
    return float(value)


def require_open_probability(name: str, value: object) -> float:
    number = _require_number(name, value)
    if not 0.0 < number < 1.0:
        raise InvalidParameterError(f"{name} must be strictly between 0 and 1, got {value}")
    return number


def require_fraction(name: str, value: object) -> float:
    number = _require_number(name, value)
    if not 0.0 <= number <= 1.0:
        raise InvalidParameterError(f"{name} must be between 0 and 1, got {value}")
    return number


def require_non_negative_float(name: str, value: object) -> float:
    number = _require_number(name, value)
    if not number >= 0.0:
        raise InvalidParameterError(f"{name} must be zero or greater, got {value}")
    return number


def require_choice(name: str, value: ChoiceT, choices: Collection[ChoiceT]) -> ChoiceT:
    if value not in choices:
        options = ", ".join(str(choice) for choice in choices)
        raise InvalidParameterError(
            f"{name} must be one of {options}, got {value!r}"
        )
    return value