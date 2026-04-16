from pydantic import BaseModel, Field


class AuthenticatedUserSchema(BaseModel):
    user_id: str = Field(min_length=1, max_length=64)
    is_authenticated: bool = True
    role: str = Field(default="developer", max_length=32)