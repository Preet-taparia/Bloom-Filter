class BloomError(Exception):
    """Base class for every error raised by the bloom package.

    Catch this type to handle any failure that originates in the library
    without also catching unrelated built in exceptions.
    """


class InvalidParameterError(BloomError):
    """Raised when a constructor or function argument is unacceptable.

    Attributes:
        parameter: Name of the offending argument, when known.
        value: The rejected value, when known.
    """

    def __init__(
        self,
        message: str,
        *,
        parameter: str | None = None,
        value: object = None,
    ) -> None:
        super().__init__(message)
        self.parameter = parameter
        self.value = value


class ParameterTypeError(InvalidParameterError, TypeError):
    """Raised when an argument has the wrong type, such as a float size."""


class ParameterValueError(InvalidParameterError, ValueError):
    """Raised when an argument has the right type but an unacceptable value."""


class InvalidItemError(BloomError, TypeError):
    """Raised when an item cannot be converted into bytes for hashing."""


class IncompatibleFilterError(BloomError, ValueError):
    """Raised when filters are combined in a way their configuration forbids."""


class AllocationError(BloomError, MemoryError):
    """Raised when the storage for a filter cannot be allocated."""
