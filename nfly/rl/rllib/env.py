"""Register nfly suites as RLlib environments: env id "nfly/<suite>/<game>"."""

from __future__ import annotations

from ray.tune.registry import register_env

from ...suite import get_suite

PREFIX = "nfly/"


def env_id(suite: str, game: str) -> str:
    return f"{PREFIX}{suite}/{game}"


def register_nfly_env(suite: str, game: str, **suite_kw) -> str:
    """Make `nfly/<suite>/<game>` known to RLlib and return the id."""
    name = env_id(suite, game)
    register_env(name, lambda cfg: get_suite(suite, **suite_kw).make(game, **cfg))
    return name
