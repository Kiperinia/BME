from typing import Generic, TypeVar

from pydantic import BaseModel


ResponseDataT = TypeVar("ResponseDataT")


class ApiResponse(BaseModel, Generic[ResponseDataT]):
    """brief:
        Represent ApiResponse state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    code: int = 200
    message: str = "success"
    data: ResponseDataT
