from enum import Enum


class TaskStatusEnum(str, Enum):
    """brief:
        Represent TaskStatusEnum state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
