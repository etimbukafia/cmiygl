from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv


def load_repo_env(*, app_env_file: str | Path | None = None) -> None:
    """Load the repository .env, then optional app overrides.

    Values already present in the process environment always win. This keeps
    CMIYGL independent of the upstream voice repo's top-level env helper, which
    is not included in that project's installed package.
    """
    project_env = Path(__file__).resolve().parent / ".env"
    original_keys = set(os.environ)

    if project_env.is_file():
        load_dotenv(project_env, override=False)

    if app_env_file is None:
        return

    app_env = Path(app_env_file)
    if not app_env.is_file() or app_env.resolve() == project_env.resolve():
        return

    for key, value in dotenv_values(app_env).items():
        if key and value is not None and key not in original_keys:
            os.environ[key] = value
