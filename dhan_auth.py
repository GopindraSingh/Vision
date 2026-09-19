"""
dhan_auth.py

DhanHQ authentication layer.

Current Dhan TOTP authentication endpoint:
    POST https://auth.dhan.co/app/generateAccessToken

Dhan currently requires:
    dhanClientId
    pin
    totp

The PIN is intentionally NOT placed in config.env because the requested
config.env was required to contain exactly the specified variables.

Usage:
    from dhan_auth import get_daily_access_token

    token = get_daily_access_token(dhan_pin="123456")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyotp
import requests


LOGGER = logging.getLogger(__name__)

AUTH_URL = "https://auth.dhan.co/app/generateAccessToken"


def _load_env_file(path: str = "config.env") -> None:
    """
    Minimal dotenv loader.

    This deliberately avoids requiring python-dotenv.
    Existing environment variables take precedence.
    """
    env_path = Path(path)

    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            # Remove inline comments only when outside a quoted value.
            value = value.strip()

            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]

            os.environ.setdefault(key, value)


_load_env_file()


@dataclass(frozen=True)
class DhanAccessToken:
    """
    Immutable representation of a Dhan access token.
    """

    client_id: str
    access_token: str
    expiry_time: Optional[str]

    @property
    def is_present(self) -> bool:
        return bool(self.access_token)


class DhanAuthenticationError(RuntimeError):
    """Raised when Dhan authentication fails."""


def _required_env(name: str) -> str:
    value = os.getenv(name)

    if not value:
        raise DhanAuthenticationError(
            f"Required environment variable {name!r} is missing."
        )

    return value.strip()


def generate_totp(secret: str) -> str:
    """
    Generate the current six-digit RFC-6238 TOTP.
    """
    try:
        totp = pyotp.TOTP(secret)
        code = totp.now()
    except Exception as exc:
        raise DhanAuthenticationError(
            "Unable to generate TOTP. Verify DHAN_TOTP_SECRET."
        ) from exc

    if len(code) != 6 or not code.isdigit():
        raise DhanAuthenticationError("Generated TOTP is not a valid 6-digit code.")

    return code


def get_daily_access_token(
    dhan_pin: str,
    timeout: float = 15.0,
) -> DhanAccessToken:
    """
    Generate a fresh 24-hour Dhan access token using TOTP.

    Parameters
    ----------
    dhan_pin:
        Dhan's six-digit account PIN.

    timeout:
        HTTP request timeout.

    Returns
    -------
    DhanAccessToken
        Immutable access-token object.
    """

    client_id = _required_env("DHAN_CLIENT_ID")
    totp_secret = _required_env("DHAN_TOTP_SECRET")

    if not dhan_pin:
        raise DhanAuthenticationError("Dhan PIN is required.")

    dhan_pin = str(dhan_pin).strip()

    if len(dhan_pin) != 6 or not dhan_pin.isdigit():
        raise DhanAuthenticationError(
            "Dhan PIN must be exactly six numeric digits."
        )

    totp_code = generate_totp(totp_secret)

    params = {
        "dhanClientId": client_id,
        "pin": dhan_pin,
        "totp": totp_code,
    }

    LOGGER.info("Generating daily Dhan access token.")

    try:
        response = requests.post(
            AUTH_URL,
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DhanAuthenticationError(
            f"Dhan authentication request failed: {exc}"
        ) from exc

    if response.status_code != 200:
        body = response.text[:500]
        raise DhanAuthenticationError(
            f"Dhan authentication failed with HTTP "
            f"{response.status_code}: {body}"
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise DhanAuthenticationError(
            "Dhan authentication returned invalid JSON."
        ) from exc

    access_token = payload.get("accessToken")

    if not access_token:
        message = (
            payload.get("errorMessage")
            or payload.get("message")
            or "No accessToken returned."
        )
        raise DhanAuthenticationError(
            f"Dhan authentication failed: {message}"
        )

    expiry_time = payload.get("expiryTime")

    LOGGER.info(
        "Dhan access token generated successfully. "
        "Expiry=%s",
        expiry_time or "unknown",
    )

    return DhanAccessToken(
        client_id=client_id,
        access_token=access_token,
        expiry_time=expiry_time,
    )


def initialize_rest_auth(
    rest_client,
    token: DhanAccessToken,
) -> None:
    """
    Securely initialize a REST client with the generated token.

    The token is passed directly to the REST client and is never logged.
    """
    if not token.is_present:
        raise DhanAuthenticationError("Cannot initialize REST client without token.")

    rest_client.set_access_token(
        access_token=token.access_token,
        client_id=token.client_id,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    # Deliberately no hard-coded PIN.
    # For fully unattended AWS operation, retrieve the PIN securely
    # from your secret-management mechanism and pass it here.
    dhan_pin = os.getenv("DHAN_PIN")

    if not dhan_pin:
        raise SystemExit(
            "DHAN_PIN was not supplied. "
            "Pass the six-digit Dhan PIN through your secure runtime "
            "secret mechanism and call get_daily_access_token()."
        )

    token = get_daily_access_token(dhan_pin=dhan_pin)

    print(token.access_token)
