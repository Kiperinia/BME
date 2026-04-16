from typing import Generic, TypeVar

from pydantic import BaseModel


ResponseDataT = TypeVar("ResponseDataT")


class ApiResponse(BaseModel, Generic[ResponseDataT]):
    code: int = 200
    message: str = "success"
    data: ResponseDataT
