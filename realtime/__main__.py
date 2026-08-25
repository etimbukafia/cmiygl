from __future__ import annotations

import asyncio

from ..config import build_capability_report, load_cmiygl_config
from .orchestrator import CmiyglSessionOrchestrator


async def _run_smoke_test() -> None:
    config = load_cmiygl_config()
    report = build_capability_report(config)

    session = CmiyglSessionOrchestrator(config.app.session_id)
    print(f"cmiygl realtime boot ok: state={session.record.state.value}")
    print(
        "providers: "
        f"stt={config.stt.provider.value} "
        f"llm={config.llm.provider.value} "
        f"tts={config.tts.provider.value} "
        f"route_mode={config.app.default_route_mode.value}"
    )
    print(
        "readiness: "
        f"twilio={'ok' if report.twilio_ready else 'missing'} "
        f"stt={'ok' if report.stt_ready else 'missing'} "
        f"llm={'ok' if report.llm_ready else 'missing'} "
        f"tts={'ok' if report.tts_ready else 'missing'} "
        f"maps={'ok' if report.maps_ready else 'missing'} "
        f"phone_runtime={'ok' if report.phone_runtime_ready else 'not_ready'}"
    )
    for issue in report.issues:
        print(f"config issue: {issue}")
    await session.close(reason="phase0_smoke_test")


def main() -> int:
    asyncio.run(_run_smoke_test())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
