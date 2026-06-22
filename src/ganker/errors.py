"""Domain errors mapped by gRPC servicers."""


class GankerError(Exception):
    """Base class for expected service errors."""


class InvalidRequestError(GankerError):
    """The request is malformed or unsupported."""


class NotFoundError(GankerError):
    """The requested run, artifact, or resource does not exist."""


class BackendUnavailableError(GankerError):
    """A configured backend cannot be used in this environment."""
