class BloomError(Exception):
    """Base exception for Bloom filter errors."""


class InvalidParameterError(BloomError, ValueError):
    """Raised when a parameter has an invalid type or value."""


class UnsupportedOperationError(BloomError):
    """Raised when an operation is not supported."""


class IncompatibleFilterError(BloomError):
    """Raised when Bloom filters are incompatible."""


class SerializationError(BloomError):
    """Raised when serialization or deserialization fails."""
