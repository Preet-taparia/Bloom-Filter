class BloomError(Exception):
    pass

class InvalidParameterError(BloomError, ValueError):
    pass


class UnsupportedOperationError(BloomError):
    pass


class IncompatibleFilterError(BloomError):
    pass


class SerializationError(BloomError):
    pass
