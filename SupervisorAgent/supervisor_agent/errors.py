class SupervisorError(Exception):
    pass


class ValidationError(SupervisorError):
    pass


class ToolTimeoutError(SupervisorError):
    pass


class InconsistencyError(SupervisorError):
    pass


class AuditFailureError(SupervisorError):
    pass


class UnknownSupervisorError(SupervisorError):
    pass
