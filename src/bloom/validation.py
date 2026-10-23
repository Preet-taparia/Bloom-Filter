import math
import reprlib
from collections.abc import Collection
from typing import TypeVar

from .Exceptions import ParameterTypeError, ParameterValueError

ChoiceT = TypeVar("ChoiceT")


def _type_error(name: str, expected: str, value: object) -> ParameterTypeError:
    return ParameterTypeError(
        f"{name} must be {expected}, got {reprlib.repr(value)}",
        parameter=name,
        value=value,
    )


def _value_error(name: str, requirement: str, value: object) -> ParameterValueError:
    return ParameterValueError(
        f"{name} must be {requirement}, got {reprlib.repr(value)}",
        parameter=name,
        value=value,
    )


def require_int(
    name: str,
    value: object,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Validate that a value is an integer within optional inclusive bounds.

    Booleans are rejected even though they subclass int, because passing
    True for a size is almost always a mistake.

    Args:
        name: Parameter name used in error messages.
        value: The value to validate.
        minimum: Smallest accepted value, or None for no lower bound.
        maximum: Largest accepted value, or None for no upper bound.

    Returns:
        The validated integer.

    Raises:
        ParameterTypeError: If the value is not an integer.
        ParameterValueError: If the value lies outside the bounds.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _type_error(name, "an integer", value)

    if minimum is not None and value < minimum:
        raise _value_error(name, f"at least {minimum}", value)

    if maximum is not None and value > maximum:
        raise _value_error(name, f"at most {maximum}", value)

    return value


def _require_number(name: str, value: object) -> float:
    """Validate a finite int or float and return it as a float.

    Raises:
        ParameterTypeError: If the value is not an int or float.
        ParameterValueError: If the value is infinite, NaN or too large
            to represent as a float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _type_error(name, "a number", value)

    try:
        number = float(value)
    except OverflowError as exc:
        raise _value_error(name, "a finite number", value) from exc

    if not math.isfinite(number):
        raise _value_error(name, "a finite number", value)

    return number


def require_open_probability(name: str, value: object) -> float:
    """Validate a probability strictly between zero and one.

    Returns:
        The value as a float.

    Raises:
        ParameterTypeError: If the value is not a number.
        ParameterValueError: If the value is not strictly between 0 and 1.
    """
    number = _require_number(name, value)

    if not 0.0 < number < 1.0:
        raise _value_error(name, "strictly between 0 and 1", value)

    return number


def require_fraction(name: str, value: object) -> float:
    """Validate a fraction between zero and one inclusive.

    Returns:
        The value as a float.

    Raises:
        ParameterTypeError: If the value is not a number.
        ParameterValueError: If the value is outside the range 0 to 1.
    """
    number = _require_number(name, value)

    if not 0.0 <= number <= 1.0:
        raise _value_error(name, "between 0 and 1", value)

    return number


def require_choice(
    name: str,
    value: ChoiceT,
    choices: Collection[ChoiceT],
) -> ChoiceT:
    """Validate that a value belongs to a collection of allowed choices.

    Unhashable values checked against a set are treated as not allowed
    instead of leaking a TypeError from the membership test.

    Raises:
        ParameterValueError: If the value is not one of the choices.
    """
    try:
        allowed = value in choices
    except TypeError:
        allowed = False

    if not allowed:
        options = ", ".join(repr(choice) for choice in choices)
        raise _value_error(name, f"one of {options}", value)

    return value
