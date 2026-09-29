from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .env_bootstrap import load_repo_env

from .core.schemas import NavigationMode


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_APP_ENV_FILE = APP_ROOT / ".env"


class ConfigValidationError(RuntimeError):
    pass


class STTProvider(StrEnum):
    DISABLED = "disabled"
    MISTRAL_VOXTRAL = "mistral_voxtral"
    DEEPGRAM = "deepgram"
    ASSEMBLYAI = "assemblyai"


class LLMProvider(StrEnum):
    DISABLED = "disabled"
    MISTRAL = "mistral"
    GEMINI = "gemini"


class TTSProvider(StrEnum):
    DISABLED = "disabled"
    CARTESIA = "cartesia"
    VOXTRAL = "voxtral"


class LocationRecoveryProviderKind(StrEnum):
    OVERPASS = "overpass"
    NOMINATIM = "nominatim"


@dataclass(frozen=True, slots=True)
class AppSettings:
    host: str
    port: int
    log_level: str
    session_id: str
    language: str
    default_route_mode: NavigationMode
    max_clarification_turns: int
    location_recovery_radius_meters: int
    nearby_landmark_radius_meters: int


@dataclass(frozen=True, slots=True)
class TwilioSettings:
    account_sid: str
    auth_token: str
    phone_number: str
    webhook_base_url: str
    stream_path: str


@dataclass(frozen=True, slots=True)
class STTSettings:
    provider: STTProvider
    mistral_api_key: str
    deepgram_api_key: str
    assemblyai_api_key: str
    voxtral_realtime_url: str
    voxtral_model: str


@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: LLMProvider
    mistral_api_key: str
    gemini_api_key: str
    mistral_model: str
    gemini_model: str
    max_tokens: int
    temperature: float


@dataclass(frozen=True, slots=True)
class TTSSettings:
    provider: TTSProvider
    mistral_api_key: str
    cartesia_api_key: str
    cartesia_voice_id: str
    cartesia_version: str
    cartesia_language: str
    cartesia_model_id: str
    voxtral_tts_model: str
    voxtral_tts_voice_id: str


@dataclass(frozen=True, slots=True)
class MapSettings:
    location_recovery_provider: LocationRecoveryProviderKind
    nominatim_user_agent: str
    overpass_user_agent: str
    overpass_base_url: str
    valhalla_base_url: str


@dataclass(frozen=True, slots=True)
class CmiyglConfig:
    app: AppSettings
    twilio: TwilioSettings
    stt: STTSettings
    llm: LLMSettings
    tts: TTSSettings
    maps: MapSettings


@dataclass(frozen=True, slots=True)
class CapabilityReport:
    twilio_ready: bool
    stt_ready: bool
    llm_ready: bool
    tts_ready: bool
    maps_ready: bool
    phone_runtime_ready: bool
    issues: tuple[str, ...]


def load_cmiygl_config(*, app_env_file: str | Path | None = None) -> CmiyglConfig:
    load_repo_env(app_env_file=app_env_file or DEFAULT_APP_ENV_FILE)

    cfg = CmiyglConfig(
        app=AppSettings(
            host=_env_string("CMIYGL_HOST", "127.0.0.1"),
            port=_env_int("CMIYGL_PORT", 8770),
            log_level=_env_string("CMIYGL_LOG_LEVEL", "INFO"),
            session_id=_env_string("CMIYGL_SESSION_ID", "cmiygl-session"),
            language=_env_string("CMIYGL_LANGUAGE", "en"),
            default_route_mode=_env_enum(
                "CMIYGL_DEFAULT_ROUTE_MODE",
                NavigationMode,
                NavigationMode.WALKING,
            ),
            max_clarification_turns=_env_int("CMIYGL_MAX_CLARIFICATION_TURNS", 3),
            location_recovery_radius_meters=_env_int(
                "CMIYGL_LOCATION_RECOVERY_RADIUS_METERS",
                1000,
            ),
            nearby_landmark_radius_meters=_env_int(
                "CMIYGL_NEARBY_LANDMARK_RADIUS_METERS",
                250,
            ),
        ),
        twilio=TwilioSettings(
            account_sid=_env_string(
                "CMIYGL_TWILIO_ACCOUNT_SID",
                os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
            ),
            auth_token=_env_string(
                "CMIYGL_TWILIO_AUTH_TOKEN",
                os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
            ),
            phone_number=_env_string(
                "CMIYGL_TWILIO_PHONE_NUMBER",
                os.getenv("TWILIO_PHONE_NUMBER", "").strip(),
            ),
            webhook_base_url=_env_string("CMIYGL_TWILIO_WEBHOOK_BASE_URL", ""),
            stream_path=_env_string("CMIYGL_TWILIO_STREAM_PATH", "/ws/cmiygl"),
        ),
        stt=STTSettings(
            provider=_env_enum(
                "CMIYGL_STT_PROVIDER",
                STTProvider,
                STTProvider.MISTRAL_VOXTRAL,
            ),
            mistral_api_key=_env_string(
                "CMIYGL_MISTRAL_API_KEY",
                os.getenv("MISTRAL_API_KEY", "").strip(),
            ),
            deepgram_api_key=_env_string(
                "CMIYGL_DEEPGRAM_API_KEY",
                os.getenv("DEEPGRAM_API_KEY", "").strip(),
            ),
            assemblyai_api_key=_env_string(
                "CMIYGL_ASSEMBLYAI_API_KEY",
                os.getenv("ASSEMBLYAI_API_KEY", "").strip(),
            ),
            voxtral_realtime_url=_env_string(
                "CMIYGL_VOXTRAL_REALTIME_URL",
                os.getenv("VOXTRAL_REALTIME_URL", "").strip(),
            ),
            voxtral_model=_env_string(
                "CMIYGL_VOXTRAL_MODEL",
                _env_string("VOXTRAL_MODEL", "voxtral-mini-transcribe-realtime-2602"),
            ),
        ),
        llm=LLMSettings(
            provider=_env_enum(
                "CMIYGL_LLM_PROVIDER",
                LLMProvider,
                LLMProvider.MISTRAL,
            ),
            mistral_api_key=_env_string(
                "CMIYGL_MISTRAL_API_KEY",
                os.getenv("MISTRAL_API_KEY", "").strip(),
            ),
            gemini_api_key=_env_string(
                "CMIYGL_GEMINI_API_KEY",
                os.getenv("GEMINI_API_KEY", "").strip(),
            ),
            mistral_model=_env_string(
                "CMIYGL_MISTRAL_LLM_MODEL",
                _env_string("MISTRAL_LLM_MODEL", "mistral-small-latest"),
            ),
            gemini_model=_env_string(
                "CMIYGL_GEMINI_LLM_MODEL",
                _env_string("GEMINI_LLM_MODEL", "gemini-2.5-flash"),
            ),
            max_tokens=_env_int("CMIYGL_MAX_TOKENS", 220),
            temperature=_env_float("CMIYGL_TEMPERATURE", 0.2),
        ),
        tts=TTSSettings(
            provider=_env_enum(
                "CMIYGL_TTS_PROVIDER",
                TTSProvider,
                TTSProvider.CARTESIA,
            ),
            mistral_api_key=_env_string(
                "CMIYGL_MISTRAL_API_KEY",
                os.getenv("MISTRAL_API_KEY", "").strip(),
            ),
            cartesia_api_key=_env_string(
                "CMIYGL_CARTESIA_API_KEY",
                os.getenv("CARTESIA_API_KEY", "").strip(),
            ),
            cartesia_voice_id=_env_string(
                "CMIYGL_CARTESIA_VOICE_ID",
                os.getenv("CARTESIA_VOICE_ID", "").strip(),
            ),
            cartesia_version=_env_string(
                "CMIYGL_CARTESIA_VERSION",
                _env_string("CARTESIA_VERSION", "2025-04-16"),
            ),
            cartesia_language=_env_string(
                "CMIYGL_CARTESIA_LANGUAGE",
                _env_string("CARTESIA_LANGUAGE", "en"),
            ),
            cartesia_model_id=_env_string(
                "CMIYGL_CARTESIA_MODEL_ID",
                _env_string("CARTESIA_MODEL_ID", "sonic-3"),
            ),
            voxtral_tts_model=_env_string(
                "CMIYGL_VOXTRAL_TTS_MODEL",
                _env_string("VOXTRAL_TTS_MODEL", "voxtral-mini-tts-2603"),
            ),
            voxtral_tts_voice_id=_env_string(
                "CMIYGL_VOXTRAL_TTS_VOICE_ID",
                os.getenv("VOXTRAL_TTS_VOICE_ID", "").strip(),
            ),
        ),
        maps=MapSettings(
            location_recovery_provider=_env_enum(
                "CMIYGL_LOCATION_RECOVERY_PROVIDER",
                LocationRecoveryProviderKind,
                LocationRecoveryProviderKind.OVERPASS,
            ),
            nominatim_user_agent=_env_string("CMIYGL_NOMINATIM_USER_AGENT", ""),
            overpass_user_agent=_env_string("CMIYGL_OVERPASS_USER_AGENT", "cmiygl/0.1"),
            overpass_base_url=_env_string(
                "CMIYGL_OVERPASS_BASE_URL",
                "https://overpass-api.de/api/interpreter",
            ),
            valhalla_base_url=_env_string(
                "CMIYGL_VALHALLA_BASE_URL",
                "http://localhost:8002",
            ),
        ),
    )

    _validate_structural_config(cfg)
    return cfg


def build_capability_report(cfg: CmiyglConfig) -> CapabilityReport:
    issues: list[str] = []

    twilio_ready = all(
        (
            cfg.twilio.account_sid,
            cfg.twilio.auth_token,
            cfg.twilio.phone_number,
            cfg.twilio.webhook_base_url,
        )
    )
    if not twilio_ready:
        issues.append(
            "Twilio is not fully configured. Set CMIYGL_TWILIO_ACCOUNT_SID, "
            "CMIYGL_TWILIO_AUTH_TOKEN, CMIYGL_TWILIO_PHONE_NUMBER, and "
            "CMIYGL_TWILIO_WEBHOOK_BASE_URL."
        )

    stt_ready = _stt_ready(cfg.stt)
    if not stt_ready:
        issues.append(_stt_issue(cfg.stt))

    llm_ready = _llm_ready(cfg.llm)
    if not llm_ready:
        issues.append(_llm_issue(cfg.llm))

    tts_ready = _tts_ready(cfg.tts)
    if not tts_ready:
        issues.append(_tts_issue(cfg.tts))

    maps_ready = bool(
        cfg.maps.nominatim_user_agent
        and cfg.maps.valhalla_base_url
        and _location_recovery_provider_ready(cfg.maps)
    )
    if not maps_ready:
        issues.append(
            "Maps are not fully configured. Set CMIYGL_NOMINATIM_USER_AGENT, confirm "
            "CMIYGL_VALHALLA_BASE_URL points at a running Valhalla service, and set "
            "the required location recovery provider configuration."
        )

    phone_runtime_ready = all((twilio_ready, stt_ready, llm_ready, tts_ready, maps_ready))

    return CapabilityReport(
        twilio_ready=twilio_ready,
        stt_ready=stt_ready,
        llm_ready=llm_ready,
        tts_ready=tts_ready,
        maps_ready=maps_ready,
        phone_runtime_ready=phone_runtime_ready,
        issues=tuple(issues),
    )


def require_phone_runtime_config(cfg: CmiyglConfig) -> None:
    report = build_capability_report(cfg)
    if report.phone_runtime_ready:
        return
    raise ConfigValidationError("CMIYGL runtime config incomplete:\n- " + "\n- ".join(report.issues))


def _validate_structural_config(cfg: CmiyglConfig) -> None:
    if cfg.app.port <= 0:
        raise ConfigValidationError("CMIYGL_PORT must be greater than 0.")
    if cfg.app.max_clarification_turns <= 0:
        raise ConfigValidationError("CMIYGL_MAX_CLARIFICATION_TURNS must be greater than 0.")
    if cfg.app.location_recovery_radius_meters <= 0:
        raise ConfigValidationError("CMIYGL_LOCATION_RECOVERY_RADIUS_METERS must be greater than 0.")
    if cfg.app.nearby_landmark_radius_meters <= 0:
        raise ConfigValidationError("CMIYGL_NEARBY_LANDMARK_RADIUS_METERS must be greater than 0.")
    if cfg.tts.cartesia_model_id.strip() == "":
        raise ConfigValidationError("CMIYGL_CARTESIA_MODEL_ID must be non-empty.")
    if cfg.twilio.stream_path and not cfg.twilio.stream_path.startswith("/"):
        raise ConfigValidationError("CMIYGL_TWILIO_STREAM_PATH must start with '/'.")


def _stt_ready(stt: STTSettings) -> bool:
    if stt.provider is STTProvider.DISABLED:
        return True
    if stt.provider is STTProvider.MISTRAL_VOXTRAL:
        return bool(stt.mistral_api_key)
    if stt.provider is STTProvider.DEEPGRAM:
        return bool(stt.deepgram_api_key)
    if stt.provider is STTProvider.ASSEMBLYAI:
        return bool(stt.assemblyai_api_key)
    return False


def _llm_ready(llm: LLMSettings) -> bool:
    if llm.provider is LLMProvider.DISABLED:
        return True
    if llm.provider is LLMProvider.MISTRAL:
        return bool(llm.mistral_api_key)
    if llm.provider is LLMProvider.GEMINI:
        return bool(llm.gemini_api_key)
    return False


def _tts_ready(tts: TTSSettings) -> bool:
    if tts.provider is TTSProvider.DISABLED:
        return True
    if tts.provider is TTSProvider.CARTESIA:
        return bool(tts.cartesia_api_key and tts.cartesia_voice_id)
    if tts.provider is TTSProvider.VOXTRAL:
        return bool(tts.mistral_api_key and tts.voxtral_tts_voice_id)
    return False


def _stt_issue(stt: STTSettings) -> str:
    if stt.provider is STTProvider.MISTRAL_VOXTRAL:
        return "STT provider mistral_voxtral requires CMIYGL_MISTRAL_API_KEY or MISTRAL_API_KEY."
    if stt.provider is STTProvider.DEEPGRAM:
        return "STT provider deepgram requires CMIYGL_DEEPGRAM_API_KEY or DEEPGRAM_API_KEY."
    if stt.provider is STTProvider.ASSEMBLYAI:
        return "STT provider assemblyai requires CMIYGL_ASSEMBLYAI_API_KEY or ASSEMBLYAI_API_KEY."
    return ""


def _llm_issue(llm: LLMSettings) -> str:
    if llm.provider is LLMProvider.MISTRAL:
        return "LLM provider mistral requires CMIYGL_MISTRAL_API_KEY or MISTRAL_API_KEY."
    if llm.provider is LLMProvider.GEMINI:
        return "LLM provider gemini requires CMIYGL_GEMINI_API_KEY or GEMINI_API_KEY."
    return ""


def _tts_issue(tts: TTSSettings) -> str:
    if tts.provider is TTSProvider.CARTESIA:
        return (
            "TTS provider cartesia requires CMIYGL_CARTESIA_API_KEY/CARTESIA_API_KEY "
            "and CMIYGL_CARTESIA_VOICE_ID/CARTESIA_VOICE_ID."
        )
    if tts.provider is TTSProvider.VOXTRAL:
        return (
            "TTS provider voxtral requires CMIYGL_MISTRAL_API_KEY/MISTRAL_API_KEY "
            "and CMIYGL_VOXTRAL_TTS_VOICE_ID/VOXTRAL_TTS_VOICE_ID."
        )
    return ""


def _location_recovery_provider_ready(maps: MapSettings) -> bool:
    if maps.location_recovery_provider is LocationRecoveryProviderKind.OVERPASS:
        return bool(maps.overpass_user_agent and maps.overpass_base_url)
    if maps.location_recovery_provider is LocationRecoveryProviderKind.NOMINATIM:
        return bool(maps.nominatim_user_agent)
    return False


def _env_string(key: str, fallback: str) -> str:
    value = os.getenv(key, "").strip()
    return value or fallback


def _env_int(key: str, fallback: int) -> int:
    value = os.getenv(key, "").strip()
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigValidationError(f"{key} must be an integer.") from exc


def _env_float(key: str, fallback: float) -> float:
    value = os.getenv(key, "").strip()
    if not value:
        return fallback
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigValidationError(f"{key} must be a float.") from exc


def _env_enum[T: StrEnum](key: str, enum_type: type[T], fallback: T) -> T:
    value = os.getenv(key, "").strip()
    if not value:
        return fallback
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise ConfigValidationError(f"{key} must be one of: {allowed}.") from exc
