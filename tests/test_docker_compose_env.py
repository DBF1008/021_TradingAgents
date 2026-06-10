"""Regression tests for docker-compose.yml environment configuration.

Ensures the ollama profile uses the correct TRADINGAGENTS_* variable names
and points at the compose-internal ollama service, not localhost.
"""

from __future__ import annotations

import pathlib

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML required to parse docker-compose.yml")

from tradingagents.default_config import _ENV_OVERRIDES

COMPOSE_PATH = pathlib.Path(__file__).resolve().parents[1] / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load(COMPOSE_PATH.read_text())


@pytest.fixture(scope="module")
def ollama_app_env(compose) -> dict[str, str]:
    """Parse the tradingagents-ollama environment list into a dict."""
    raw = compose["services"]["tradingagents-ollama"]["environment"]
    env = {}
    for entry in raw:
        key, _, value = entry.partition("=")
        env[key] = value
    return env


# ---- variable naming --------------------------------------------------------


def test_no_bare_llm_provider(ollama_app_env):
    """LLM_PROVIDER (without prefix) is ignored by the code — must not appear."""
    assert "LLM_PROVIDER" not in ollama_app_env, (
        "Use TRADINGAGENTS_LLM_PROVIDER, not LLM_PROVIDER"
    )


def test_tradingagents_vars_are_recognised(ollama_app_env):
    """Every TRADINGAGENTS_* var in compose must exist in _ENV_OVERRIDES."""
    for key in ollama_app_env:
        if key.startswith("TRADINGAGENTS_"):
            assert key in _ENV_OVERRIDES, (
                f"{key} is not in _ENV_OVERRIDES — the app will ignore it"
            )


# ---- provider ---------------------------------------------------------------


def test_provider_is_ollama(ollama_app_env):
    assert ollama_app_env.get("TRADINGAGENTS_LLM_PROVIDER") == "ollama"


# ---- base URL ---------------------------------------------------------------


def test_ollama_base_url_points_at_service(ollama_app_env):
    """Inside compose, OLLAMA_BASE_URL must use the 'ollama' service name."""
    url = ollama_app_env.get("OLLAMA_BASE_URL", "")
    assert "localhost" not in url, (
        "localhost won't reach the ollama container — use http://ollama:11434/v1"
    )
    assert "ollama" in url


# ---- model defaults ---------------------------------------------------------


def test_model_defaults_are_set(ollama_app_env):
    """Ollama profile must ship with model defaults to avoid openai fallback."""
    assert ollama_app_env.get("TRADINGAGENTS_DEEP_THINK_LLM"), (
        "TRADINGAGENTS_DEEP_THINK_LLM must be set to an ollama-compatible model"
    )
    assert ollama_app_env.get("TRADINGAGENTS_QUICK_THINK_LLM"), (
        "TRADINGAGENTS_QUICK_THINK_LLM must be set to an ollama-compatible model"
    )


# ---- ollama service health ---------------------------------------------------


def test_ollama_service_has_healthcheck(compose):
    """The ollama service needs a healthcheck so depends_on can wait for it."""
    ollama_svc = compose["services"]["ollama"]
    assert "healthcheck" in ollama_svc, "ollama service must define a healthcheck"


def test_app_waits_for_healthy_ollama(compose):
    """tradingagents-ollama should wait for ollama to be healthy, not just started."""
    deps = compose["services"]["tradingagents-ollama"]["depends_on"]
    # depends_on can be a list (just started) or dict (with condition)
    assert isinstance(deps, dict), (
        "depends_on should be a dict with condition, not a bare list"
    )
    assert deps["ollama"]["condition"] == "service_healthy"
