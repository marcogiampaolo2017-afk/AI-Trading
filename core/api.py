# -*- coding: utf-8 -*-
"""
core/api.py — Base44 REST helper
=================================
SECURITY FIXES applied:
  1. API key read from environment variable BASE44_API_KEY (never hardcoded).
  2. Module-level make_api_request() call removed: it executed on every
     `import api`, causing a live HTTP request during any import — including
     during training on machines without internet access.
"""
import os
import requests

# ── Read credentials from environment ─────────────────────────────────────────
_BASE44_URL = "https://app.base44.com/api"
_APP_ID     = "696fe84f14c617992088dd7d"
_API_KEY    = os.environ.get("BASE44_API_KEY", "")   # Set in OS env, never here


def _headers() -> dict:
    """Build auth headers; raises if the key is missing."""
    if not _API_KEY:
        raise EnvironmentError(
            "BASE44_API_KEY environment variable is not set. "
            "Export it before running the bot."
        )
    return {
        "api_key": _API_KEY,
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def make_api_request(api_path: str, method: str = "GET", data=None):
    """Generic Base44 REST call."""
    url = f"{_BASE44_URL}/{api_path}"
    hdrs = _headers()
    if method.upper() == "GET":
        response = requests.request(method, url, headers=hdrs, params=data, timeout=15)
    else:
        response = requests.request(method, url, headers=hdrs, json=data, timeout=15)
    response.raise_for_status()
    return response.json()


def update_entity(entity_id: str, update_data: dict):
    """Update a BotStatus entity via PUT."""
    url = f"{_BASE44_URL}/apps/{_APP_ID}/entities/BotStatus/{entity_id}"
    response = requests.put(url, headers=_headers(), json=update_data, timeout=15)
    response.raise_for_status()
    return response.json()
