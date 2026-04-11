"""Exchange an Upstox OAuth authorization code for an access token.

Official flow (browser + one POST): register an app in the Upstox developer console,
send the user through the Authorize URL, then POST the returned ``code`` to the
Get Token API. See: https://upstox.com/developer/api-documentation/get-token/

The access token is short-lived (Upstox documents validity until ~03:30 IST the next day).
Set ``KUBERA_UPSTOX_ACCESS_TOKEN`` to the ``access_token`` string for Kubera ingest.
"""

from __future__ import annotations

from typing import Any

import requests

UPSTOX_TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"


def exchange_authorization_code_for_access_token(
    *,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    timeout_seconds: float = 60.0,
) -> str:
    """
    POST ``grant_type=authorization_code`` to Upstox and return ``access_token``.

    Parameters must match the app you created in the Upstox developer console
    (``client_id``, ``client_secret``, ``redirect_uri``). The ``code`` is the
    one-time value from the user's redirect query string after login.
    """

    response = requests.post(
        UPSTOX_TOKEN_URL,
        headers={
            "accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "code": code.strip(),
            "client_id": client_id.strip(),
            "client_secret": client_secret.strip(),
            "redirect_uri": redirect_uri.strip(),
            "grant_type": "authorization_code",
        },
        timeout=timeout_seconds,
    )
    payload: dict[str, Any] = {}
    try:
        payload = response.json()
    except ValueError:
        payload = {}

    if response.status_code != 200:
        raise RuntimeError(
            f"Upstox token exchange failed ({response.status_code}): {response.text[:800]}"
        )

    token = payload.get("access_token")
    if not token or not isinstance(token, str):
        raise RuntimeError(f"Upstox token response missing access_token: {payload!r}")

    return token.strip()
