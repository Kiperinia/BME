from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class LlmProfileSchema(BaseModel):
    profileId: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._-]+$")
    providerKind: Literal["openai_compatible", "modelscope"] = "openai_compatible"
    defaultProvider: str = Field(default="openai", min_length=1, max_length=64)
    defaultModel: str = Field(default="gpt-4o-mini", min_length=1, max_length=128)
    apiKey: str = Field(default="")
    baseUrl: str = Field(default="")
    timeout: int = Field(default=60, ge=1, le=600)


class LlmSettingsSchema(BaseModel):
    activeProfile: str = Field(default="openai_compatible", min_length=1, max_length=128)
    profiles: list[LlmProfileSchema] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_profiles(self) -> "LlmSettingsSchema":
        if not self.profiles:
            raise ValueError("at least one LLM profile is required")

        profile_ids = [profile.profileId for profile in self.profiles]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("LLM profile ids must be unique")

        if self.activeProfile not in profile_ids:
            raise ValueError("activeProfile must match an existing LLM profile")

        return self


class AgentSettingsSchema(BaseModel):
    enableLlm: bool = True
    enableLlmReport: bool = True
    pixelSizeMm: float = Field(default=0.15, gt=0.0, le=10.0)


class Sam3SettingsSchema(BaseModel):
    loadMode: Literal["mock", "sam3"] = "mock"
    device: str = Field(default="cuda", min_length=1, max_length=32)
    checkpointPath: str = Field(default="")
    inputSize: int = Field(default=1024, ge=256, le=4096)
    keepAspectRatio: bool = False
    warmupEnabled: bool = True
    loraEnabled: bool = False
    loraPath: str = Field(default="")


class RuntimeSettingsSchema(BaseModel):
    inferenceTimeoutSeconds: int = Field(default=20, ge=1, le=300)
    maxUploadSizeMb: int = Field(default=20, ge=1, le=200)
    mockDelayMs: int = Field(default=0, ge=0, le=10000)


class SystemSettingsPayloadSchema(BaseModel):
    llm: LlmSettingsSchema = Field(default_factory=LlmSettingsSchema)
    agent: AgentSettingsSchema = Field(default_factory=AgentSettingsSchema)
    sam3: Sam3SettingsSchema = Field(default_factory=Sam3SettingsSchema)
    runtime: RuntimeSettingsSchema = Field(default_factory=RuntimeSettingsSchema)


class SystemSettingsStatusSchema(BaseModel):
    llmReady: bool
    sam3Ready: bool
    sam3RuntimeMode: Literal["mock", "sam3"]
    loraLoaded: bool
    llmConfigPath: str
    runtimeSettingsPath: str
    warnings: list[str] = Field(default_factory=list)


class SystemSettingsResponseSchema(BaseModel):
    settings: SystemSettingsPayloadSchema
    status: SystemSettingsStatusSchema