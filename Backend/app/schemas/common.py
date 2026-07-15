from pydantic import BaseModel, Field


class AuthenticatedUserSchema(BaseModel):
    """brief:
        Represent AuthenticatedUserSchema state and behavior.

    parameter:
        - None.

    retrival:
        - Provides instances used by the surrounding workflow.
    """
    user_id: str = Field(min_length=1, max_length=64)
    is_authenticated: bool = True
    role: str = Field(default="developer", max_length=32)
