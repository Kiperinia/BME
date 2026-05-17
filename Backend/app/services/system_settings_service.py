from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import (
    Settings,
    WORKSPACE_DIR,
    get_runtime_settings_path,
    get_settings,
    load_settings_overrides,
    refresh_settings_cache,
    save_settings_overrides,
)
from app.core.exceptions import AppException
from app.schemas.system_settings import (
    AgentSettingsSchema,
    LlmProfileSchema,
    LlmSettingsSchema,
    RuntimeSettingsSchema,
    Sam3SettingsSchema,
    SystemSettingsPayloadSchema,
    SystemSettingsResponseSchema,
    SystemSettingsStatusSchema,
)
from app.services.sam3_runtime import SAM3RuntimeSingleton


class SystemSettingsService:
    def __init__(self) -> None:
        self.llm_config_path = (WORKSPACE_DIR / "agent" / "config" / "llm_profiles.json").resolve()
        self.runtime_settings_path = get_runtime_settings_path().resolve()

    def get_system_settings(self) -> SystemSettingsResponseSchema:
        settings, settings_warnings = self._load_settings_with_fallback()
        llm_config, llm_warnings = self._load_llm_config_with_fallback()
        payload = self._build_payload(settings=settings, llm_config=llm_config)
        status = self._build_status(settings=settings, payload=payload)
        status.warnings = [*settings_warnings, *llm_warnings, *status.warnings]
        return SystemSettingsResponseSchema(settings=payload, status=status)

    def update_system_settings(
        self,
        payload: SystemSettingsPayloadSchema,
    ) -> SystemSettingsResponseSchema:
        self._validate_payload(payload)

        previous_llm_config_snapshot = self._read_optional_text(self.llm_config_path)
        previous_runtime_snapshot = self._read_optional_text(self.runtime_settings_path)
        previous_overrides = self._load_runtime_overrides_with_fallback()
        next_llm_config = self._serialize_llm_config(payload.llm)
        next_overrides = self._serialize_runtime_overrides(payload)
        runtime_changed = next_overrides != previous_overrides

        try:
            self._write_llm_config(next_llm_config)
            save_settings_overrides(next_overrides)
            refresh_settings_cache()
            if runtime_changed:
                SAM3RuntimeSingleton.reload_instance(settings=get_settings())
        except AppException:
            self._rollback(
                previous_llm_config_snapshot=previous_llm_config_snapshot,
                previous_runtime_snapshot=previous_runtime_snapshot,
            )
            raise
        except Exception as exc:
            self._rollback(
                previous_llm_config_snapshot=previous_llm_config_snapshot,
                previous_runtime_snapshot=previous_runtime_snapshot,
            )
            raise AppException(500, 50031, f"failed to apply system settings: {exc}") from exc

        return self.get_system_settings()

    def _rollback(
        self,
        previous_llm_config_snapshot: str | None,
        previous_runtime_snapshot: str | None,
    ) -> None:
        self._restore_file_snapshot(self.llm_config_path, previous_llm_config_snapshot)
        self._restore_file_snapshot(self.runtime_settings_path, previous_runtime_snapshot)
        refresh_settings_cache()
        try:
            SAM3RuntimeSingleton.reload_instance(settings=get_settings())
        except Exception:
            return

    def _load_settings_with_fallback(self) -> tuple[Settings, list[str]]:
        try:
            return get_settings(), []
        except Exception as exc:
            return Settings(), [f"运行时配置文件读取失败，已回退到默认设置：{exc}"]

    def _load_llm_config_with_fallback(self) -> tuple[dict[str, Any], list[str]]:
        try:
            return self._load_llm_config(), []
        except AppException as exc:
            return self._default_llm_config(), [f"LLM 配置文件读取失败，已回退到默认配置：{exc.message}"]
        except Exception as exc:
            return self._default_llm_config(), [f"LLM 配置文件读取失败，已回退到默认配置：{exc}"]

    def _load_runtime_overrides_with_fallback(self) -> dict[str, Any]:
        try:
            return load_settings_overrides()
        except Exception:
            return {}

    @staticmethod
    def _read_optional_text(path: Path) -> str | None:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _restore_file_snapshot(path: Path, content: str | None) -> None:
        if content is None:
            if path.exists():
                path.unlink()
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def _validate_payload(self, payload: SystemSettingsPayloadSchema) -> None:
        if payload.sam3.loadMode == "sam3":
            checkpoint_path = Path(payload.sam3.checkpointPath).expanduser()
            if not checkpoint_path.exists():
                raise AppException(400, 40041, "SAM3 checkpoint path does not exist")

            if payload.sam3.loraEnabled:
                if not payload.sam3.loraPath.strip():
                    raise AppException(400, 40042, "SAM3 LoRA is enabled but no adapter path was provided")

                lora_path = Path(payload.sam3.loraPath).expanduser()
                if not lora_path.exists():
                    raise AppException(400, 40043, "SAM3 LoRA checkpoint path does not exist")

    def _build_payload(
        self,
        *,
        settings: Any,
        llm_config: dict[str, Any],
    ) -> SystemSettingsPayloadSchema:
        profiles = llm_config.get("profiles", {})
        profile_schemas: list[LlmProfileSchema] = []

        for profile_id, profile_data in profiles.items():
            if not isinstance(profile_data, dict):
                continue

            provider_kind = self._detect_provider_kind(profile_data)
            api_key_field = "modelscope_api_key" if provider_kind == "modelscope" else "api_key"
            base_url_field = "modelscope_base_url" if provider_kind == "modelscope" else "base_url"
            default_provider = "modelscope" if provider_kind == "modelscope" else "openai"
            default_model = (
                "Qwen/Qwen2.5-VL-72B-Instruct"
                if provider_kind == "modelscope"
                else "gpt-4o-mini"
            )
            default_base_url = (
                "https://api-inference.modelscope.cn/v1/"
                if provider_kind == "modelscope"
                else self._resolve_openai_compatible_base_url(
                    provider=str(profile_data.get("default_provider", default_provider)),
                    base_url=str(profile_data.get(base_url_field, "")),
                )
            )

            profile_schemas.append(
                LlmProfileSchema(
                    profileId=profile_id,
                    providerKind=provider_kind,
                    defaultProvider=profile_data.get("default_provider", default_provider),
                    defaultModel=profile_data.get("default_model", default_model),
                    apiKey=profile_data.get(api_key_field, ""),
                    baseUrl=profile_data.get(base_url_field, default_base_url),
                    timeout=profile_data.get("timeout", 60),
                )
            )

        if not profile_schemas:
            default_profiles = self._default_llm_config().get("profiles", {})
            for profile_id, profile_data in default_profiles.items():
                profile_schemas.append(
                    LlmProfileSchema(
                        profileId=profile_id,
                        providerKind=self._detect_provider_kind(profile_data),
                        defaultProvider=profile_data.get("default_provider", "openai"),
                        defaultModel=profile_data.get("default_model", "gpt-4o-mini"),
                        apiKey=profile_data.get("api_key", profile_data.get("modelscope_api_key", "")),
                        baseUrl=profile_data.get("base_url", profile_data.get("modelscope_base_url", "")),
                        timeout=profile_data.get("timeout", 60),
                    )
                )

        active_profile = llm_config.get("active_profile") or profile_schemas[0].profileId

        return SystemSettingsPayloadSchema(
            llm=LlmSettingsSchema(
                activeProfile=active_profile,
                profiles=profile_schemas,
            ),
            agent=AgentSettingsSchema(
                enableLlm=settings.agent_use_llm,
                enableLlmReport=settings.agent_use_llm_report,
                pixelSizeMm=settings.agent_pixel_size_mm,
                useLlmReport=settings.report_use_llm,
                enableReportReflection=settings.report_enable_reflection,
                reflectionMaxIterations=settings.report_reflection_max_iterations,
                reflectionQualityThreshold=settings.report_reflection_quality_threshold,
            ),
            sam3=Sam3SettingsSchema(
                loadMode=settings.model_load_mode,
                device=settings.model_device,
                checkpointPath=settings.model_checkpoint_path,
                inputSize=settings.model_input_size,
                keepAspectRatio=settings.model_keep_aspect_ratio,
                warmupEnabled=settings.model_warmup_enabled,
                loraEnabled=settings.model_lora_enabled,
                loraPath=settings.model_lora_path,
                loraStage=settings.model_lora_stage,
            ),
            runtime=RuntimeSettingsSchema(
                inferenceTimeoutSeconds=settings.model_inference_timeout_seconds,
                maxUploadSizeMb=settings.max_upload_size_mb,
                mockDelayMs=settings.model_mock_delay_ms,
            ),
        )

    def _build_status(
        self,
        *,
        settings: Any,
        payload: SystemSettingsPayloadSchema,
    ) -> SystemSettingsStatusSchema:
        warnings: list[str] = []
        llm_ready = self._is_llm_ready(payload.llm)

        if not llm_ready and (payload.agent.enableLlm or payload.agent.enableLlmReport):
            warnings.append("当前激活的 LLM profile 仍缺少必要凭据，Agent 会自动回退到规则模式。")

        sam3_ready = settings.model_load_mode == "mock"
        runtime_instance = SAM3RuntimeSingleton.peek_instance()
        if settings.model_load_mode == "sam3":
            sam3_ready = bool(runtime_instance and runtime_instance.engine.model is not None)

        if settings.model_load_mode == "mock":
            warnings.append("SAM3 当前运行在 mock 模式，适合联调但不会执行真实模型推理。")

        if settings.model_lora_enabled and settings.model_load_mode != "sam3":
            warnings.append("已保存 LoRA 路径，但当前是 mock 模式，LoRA 不会实际加载。")

        last_reload_error = SAM3RuntimeSingleton.get_last_reload_error()
        if last_reload_error:
            warnings.append(f"最近一次 SAM3 重载失败：{last_reload_error}")

        return SystemSettingsStatusSchema(
            llmReady=llm_ready,
            sam3Ready=sam3_ready,
            sam3RuntimeMode=settings.model_load_mode,
            loraLoaded=bool(settings.model_lora_enabled and settings.model_lora_path and sam3_ready),
            llmConfigPath=str(self.llm_config_path),
            runtimeSettingsPath=str(self.runtime_settings_path),
            warnings=warnings,
        )

    def _load_llm_config(self) -> dict[str, Any]:
        if not self.llm_config_path.exists():
            return self._default_llm_config()

        raw = json.loads(self.llm_config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise AppException(500, 50032, "LLM config file must be a JSON object")
        return raw

    def _write_llm_config(self, config_data: dict[str, Any]) -> None:
        self.llm_config_path.parent.mkdir(parents=True, exist_ok=True)
        self.llm_config_path.write_text(
            json.dumps(config_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _default_llm_config() -> dict[str, Any]:
        return {
            "active_profile": "deepseek_chat",
            "profiles": {
                "openai_compatible": {
                    "default_provider": "openai",
                    "default_model": "gpt-4o-mini",
                    "api_key": "",
                    "base_url": "",
                    "timeout": 60,
                },
                "deepseek_chat": {
                    "default_provider": "deepseek",
                    "default_model": "deepseek-chat",
                    "api_key": "",
                    "base_url": "https://api.deepseek.com/v1",
                    "timeout": 60,
                },
                "modelscope_qwen": {
                    "default_provider": "modelscope",
                    "default_model": "Qwen/Qwen2.5-VL-72B-Instruct",
                    "modelscope_api_key": "",
                    "modelscope_base_url": "https://api-inference.modelscope.cn/v1/",
                    "timeout": 60,
                },
            },
        }

    @staticmethod
    def _serialize_llm_config(payload: LlmSettingsSchema) -> dict[str, Any]:
        profiles: dict[str, dict[str, Any]] = {}

        for profile in payload.profiles:
            if profile.providerKind == "modelscope":
                profiles[profile.profileId] = {
                    "default_provider": profile.defaultProvider,
                    "default_model": profile.defaultModel,
                    "modelscope_api_key": profile.apiKey,
                    "modelscope_base_url": profile.baseUrl,
                    "timeout": profile.timeout,
                }
                continue

            resolved_base_url = profile.baseUrl
            if not resolved_base_url.strip() and profile.defaultProvider.strip().lower() == "deepseek":
                resolved_base_url = "https://api.deepseek.com/v1"

            profiles[profile.profileId] = {
                "default_provider": profile.defaultProvider,
                "default_model": profile.defaultModel,
                "api_key": profile.apiKey,
                "base_url": resolved_base_url,
                "timeout": profile.timeout,
            }

        return {
            "active_profile": payload.activeProfile,
            "profiles": profiles,
        }

    @staticmethod
    def _serialize_runtime_overrides(payload: SystemSettingsPayloadSchema) -> dict[str, Any]:
        return {
            "model_load_mode": payload.sam3.loadMode,
            "model_device": payload.sam3.device,
            "model_checkpoint_path": payload.sam3.checkpointPath,
            "model_lora_enabled": payload.sam3.loraEnabled,
            "model_lora_path": payload.sam3.loraPath,
            "model_lora_stage": payload.sam3.loraStage,
            "model_input_size": payload.sam3.inputSize,
            "model_keep_aspect_ratio": payload.sam3.keepAspectRatio,
            "model_warmup_enabled": payload.sam3.warmupEnabled,
            "model_inference_timeout_seconds": payload.runtime.inferenceTimeoutSeconds,
            "model_mock_delay_ms": payload.runtime.mockDelayMs,
            "max_upload_size_mb": payload.runtime.maxUploadSizeMb,
            "agent_use_llm": payload.agent.enableLlm,
            "agent_use_llm_report": payload.agent.enableLlmReport,
            "agent_pixel_size_mm": payload.agent.pixelSizeMm,
            "report_use_llm": payload.agent.useLlmReport,
            "report_enable_reflection": payload.agent.enableReportReflection,
            "report_reflection_max_iterations": payload.agent.reflectionMaxIterations,
            "report_reflection_quality_threshold": payload.agent.reflectionQualityThreshold,
        }

    @staticmethod
    def _is_llm_ready(payload: LlmSettingsSchema) -> bool:
        active_profile = next(
            (profile for profile in payload.profiles if profile.profileId == payload.activeProfile),
            None,
        )
        if active_profile is None:
            return False

        if active_profile.providerKind == "openai_compatible":
            provider = active_profile.defaultProvider.strip().lower()
            if provider == "deepseek":
                return bool(active_profile.apiKey.strip())
            return bool(active_profile.apiKey.strip() and active_profile.baseUrl.strip())

        return bool(active_profile.apiKey.strip())

    @staticmethod
    def _detect_provider_kind(profile_data: dict[str, Any]) -> str:
        default_provider = str(profile_data.get("default_provider", "")).strip().lower()
        if "modelscope_api_key" in profile_data or default_provider == "modelscope":
            return "modelscope"
        return "openai_compatible"

    @staticmethod
    def _resolve_openai_compatible_base_url(*, provider: str, base_url: str) -> str:
        resolved_provider = provider.strip().lower()
        if base_url.strip():
            return base_url
        if resolved_provider == "deepseek":
            return "https://api.deepseek.com/v1"
        return ""
