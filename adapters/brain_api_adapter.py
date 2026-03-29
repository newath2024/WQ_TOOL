from __future__ import annotations

import getpass
import json
import os
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urljoin

import requests

from adapters.simulation_adapter import SimulationAdapter
from services.email_service import EmailService, SmtpNotificationConfig


class SessionProtocol(Protocol):
    def get(self, url: str, **kwargs): ...

    def post(self, url: str, **kwargs): ...

    @property
    def cookies(self): ...


@dataclass(slots=True)
class ApiEndpointConfig:
    authentication_path: str = "/authentication"
    submit_path: str = "/simulations"
    status_path_template: str = "/simulations/{job_id}"
    alpha_path_template: str = "/alphas/{alpha_id}"
    recordsets_path_template: str = "/alphas/{alpha_id}/recordsets"


class PersonaVerificationRequired(RuntimeError):
    def __init__(self, persona_url: str) -> None:
        super().__init__("BRAIN Persona verification is required before API work can continue.")
        self.persona_url = persona_url


class BrainApiAdapter(SimulationAdapter):
    """
    BRAIN API adapter with interactive authentication support.

    The adapter can:
    - reuse a saved session cookie
    - prompt for email/password in the terminal
    - pause for Persona/face-scan authentication when required
    - submit simulations and poll their progress

    It does not persist the password. Only session cookies may be stored locally.
    """

    def __init__(
        self,
        *,
        base_url: str = "https://api.worldquantbrain.com",
        auth_env: str = "BRAIN_API_TOKEN",
        auth_token: str | None = None,
        email_env: str = "BRAIN_API_EMAIL",
        password_env: str = "BRAIN_API_PASSWORD",
        credentials_file: str = "secrets/brain_credentials.json",
        session_path: str = "outputs/brain_api_session.json",
        auth_expiry_seconds: int = 14400,
        open_browser_for_persona: bool = True,
        persona_poll_interval_seconds: int = 15,
        persona_timeout_seconds: int = 1800,
        endpoints: ApiEndpointConfig | None = None,
        session: SessionProtocol | None = None,
        email_service: EmailService | None = None,
        max_retries: int = 3,
        rate_limit_per_minute: int = 60,
        verify_ssl: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.auth_env = auth_env
        self.auth_token = auth_token
        self.email_env = email_env
        self.password_env = password_env
        self.credentials_file = Path(credentials_file).expanduser().resolve()
        self.session_path = Path(session_path).expanduser().resolve()
        self.auth_expiry_seconds = auth_expiry_seconds
        self.open_browser_for_persona = open_browser_for_persona
        self.persona_poll_interval_seconds = persona_poll_interval_seconds
        self.persona_timeout_seconds = persona_timeout_seconds
        self.endpoints = endpoints or ApiEndpointConfig()
        self.session = session or requests.Session()
        self.email_service = email_service or EmailService()
        self.max_retries = max_retries
        self.rate_limit_per_minute = rate_limit_per_minute
        self.verify_ssl = verify_ssl
        self._min_request_gap_seconds = 60.0 / float(rate_limit_per_minute)
        self._last_request_at = 0.0
        self._loaded_session = False
        self._credentials_payload: dict | None = None

    def submit_simulation(self, expression: str, sim_config: dict) -> dict:
        self.ensure_authenticated()
        response = self.with_retry(
            lambda: self._request(
                "POST",
                self._resolve_submit_url(),
                json_payload=self.build_submit_payload(expression, sim_config),
            )
        )
        payload = self._response_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("Expected a JSON object from BRAIN submit response.")
        return self.parse_submit_response(response=response, payload=payload, expression=expression)

    def get_simulation_status(self, job_id: str) -> dict:
        self.ensure_authenticated()
        response = self.with_retry(
            lambda: self._request("GET", self._resolve_status_url(job_id), json_payload=None)
        )
        payload = self._response_json(response)
        if not isinstance(payload, dict):
            raise RuntimeError("Expected a JSON object from BRAIN status response.")
        return self.parse_status_response(response=response, payload=payload, job_id=job_id)

    def get_simulation_result(self, job_id: str) -> dict:
        self.ensure_authenticated()
        simulation_response = self.with_retry(
            lambda: self._request("GET", self._resolve_status_url(job_id), json_payload=None)
        )
        simulation_payload = self._response_json(simulation_response)
        if not isinstance(simulation_payload, dict):
            raise RuntimeError("Expected a JSON object from BRAIN simulation status response.")
        alpha_id = simulation_payload.get("alpha")
        alpha_payload: dict = {}
        recordsets_payload: dict = {}
        if alpha_id:
            alpha_response = self.with_retry(
                lambda: self._request("GET", self._resolve_alpha_url(str(alpha_id)), json_payload=None)
            )
            alpha_payload_raw = self._response_json(alpha_response)
            alpha_payload = alpha_payload_raw if isinstance(alpha_payload_raw, dict) else {}
            try:
                recordsets_response = self.with_retry(
                    lambda: self._request("GET", self._resolve_recordsets_url(str(alpha_id)), json_payload=None)
                )
                recordsets_payload_raw = self._response_json(recordsets_response)
                recordsets_payload = recordsets_payload_raw if isinstance(recordsets_payload_raw, dict) else {}
            except Exception:  # noqa: BLE001
                recordsets_payload = {}
        return self.parse_result_response(
            simulation_payload=simulation_payload,
            alpha_payload=alpha_payload,
            recordsets_payload=recordsets_payload,
            job_id=job_id,
        )

    def batch_submit(self, expressions: list[str], sim_config: dict) -> list[dict]:
        # Submit one-by-one for now. The official API supports multi-simulation, but its parent/child
        # tracking shape is different enough that we keep the production flow explicit until we wire
        # a dedicated parent multi-simulation handler.
        return [self.submit_simulation(expression, sim_config) for expression in expressions]

    def ensure_authenticated(
        self,
        *,
        force: bool = False,
        show_password: bool = False,
        interactive: bool = True,
    ) -> dict:
        token = self.auth_token or os.getenv(self.auth_env)
        if token:
            return {"mode": "bearer_token", "status": "ready"}

        if force:
            self._clear_session_file()

        if not self._loaded_session:
            self._load_session_from_disk()
            self._loaded_session = True

        if not force and self._authentication_state().get("authenticated"):
            return {"mode": "session_cookie", "status": "ready", "session_path": str(self.session_path)}

        email = os.getenv(self.email_env) or self._credential_value("brain", "email")
        if not email:
            if not interactive:
                raise RuntimeError(
                    "BRAIN non-interactive authentication requires credentials in env vars or credentials_file."
                )
            email = input("BRAIN email: ").strip()
        if not email:
            raise ValueError("BRAIN email is required for interactive authentication.")

        password = os.getenv(self.password_env) or self._credential_value("brain", "password")
        if not password:
            if not interactive:
                raise RuntimeError(
                    "BRAIN non-interactive authentication requires credentials in env vars or credentials_file."
                )
            password = self._prompt_password_interactive(show_password=show_password)
        if not password:
            raise ValueError("BRAIN password is required for interactive authentication.")

        self.authenticate_with_credentials(email=email, password=password, interactive=interactive)
        state = self._authentication_state()
        if not state.get("authenticated"):
            raise RuntimeError("BRAIN authentication did not complete successfully.")
        self._save_session_to_disk()
        return {
            "mode": "interactive" if interactive else "non_interactive",
            "status": "ready",
            "session_path": str(self.session_path),
        }

    def authenticate_with_credentials(self, *, email: str, password: str, interactive: bool = True) -> None:
        response = self._request(
            "POST",
            self._resolve_authentication_url(),
            json_payload={"expiry": self.auth_expiry_seconds},
            auth=(email, password),
            allow_retry=False,
        )
        if response.status_code in {200, 201}:
            return
        if response.status_code == 401 and response.headers.get("WWW-Authenticate", "").lower() == "persona":
            persona_url = urljoin(response.url, response.headers.get("Location", ""))
            if not persona_url:
                raise RuntimeError("BRAIN requested Persona authentication but did not provide a Location URL.")
            if not interactive:
                raise PersonaVerificationRequired(persona_url)
            print("BRAIN requires biometric authentication.")
            print(f"Open this URL to complete face scan: {persona_url}")
            if self.open_browser_for_persona:
                try:
                    webbrowser.open(persona_url)
                except Exception:  # noqa: BLE001
                    pass
            self._notify_persona_required(persona_url)
            persona_response = self._wait_for_persona_completion(persona_url)
            if persona_response.status_code not in {200, 201, 204}:
                raise RuntimeError(
                    f"Persona authentication failed with status {persona_response.status_code}: {persona_response.text}"
                )
            return
        raise RuntimeError(
            f"BRAIN authentication failed with status {response.status_code}: {response.text}"
        )

    def send_persona_notification(self, persona_url: str) -> bool:
        smtp_config = self._smtp_notification_config()
        if smtp_config is None:
            return False
        self.email_service.send_persona_link(persona_url=persona_url, smtp_config=smtp_config)
        return True

    @staticmethod
    def _prompt_password_interactive(show_password: bool = False) -> str:
        if show_password:
            return input("BRAIN password (visible): ").strip()
        print("Password input is hidden. Type normally, then press Enter.")
        try:
            password = getpass.getpass("BRAIN password: ")
        except (EOFError, KeyboardInterrupt):
            password = ""
        if password:
            return password
        print("Hidden password input was empty or unavailable.")
        return input("BRAIN password (visible fallback): ").strip()

    def build_submit_payload(self, expression: str, sim_config: dict) -> dict:
        settings = {
            "instrumentType": str(sim_config.get("instrument_type", "EQUITY")).upper(),
            "region": str(sim_config["region"]).upper(),
            "universe": str(sim_config["universe"]).upper(),
            "delay": int(sim_config["delay"]),
            "decay": int(sim_config["decay"]),
            "neutralization": str(sim_config["neutralization"]).upper(),
            "truncation": float(sim_config["truncation"]),
            "pasteurization": "ON" if bool(sim_config.get("pasteurization", True)) else "OFF",
            "unitHandling": str(sim_config.get("unit_handling", "VERIFY")).upper(),
            "nanHandling": str(sim_config.get("nan_handling", "OFF")).upper(),
            "language": str(sim_config.get("language", "FASTEXPR")).upper(),
            "visualization": bool(sim_config.get("visualization", False)),
        }
        test_period = sim_config.get("test_period")
        if test_period:
            settings["testPeriod"] = str(test_period)
        max_trade = sim_config.get("max_trade")
        if max_trade is not None:
            settings["maxTrade"] = str(max_trade).upper()
        return {
            "type": str(sim_config.get("simulation_type", "REGULAR")).upper(),
            "settings": settings,
            "regular": expression,
        }

    def parse_submit_response(self, *, response, payload: dict, expression: str) -> dict:
        job_id = payload.get("id") or payload.get("simulation_id")
        if not job_id:
            location = response.headers.get("Location", "")
            if location:
                job_id = location.rstrip("/").split("/")[-1]
        if not job_id:
            raise ValueError("BRAIN submit response did not contain a simulation id.")
        return {
            "job_id": str(job_id),
            "expression": expression,
            "status": self._map_simulation_status(payload.get("status")),
            "raw_submission": payload,
        }

    def parse_status_response(self, *, response, payload: dict, job_id: str) -> dict:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            return {
                "job_id": job_id,
                "status": "running",
                "retry_after": float(retry_after),
                "raw_status": payload,
            }
        return {
            "job_id": job_id,
            "status": self._map_simulation_status(payload.get("status")),
            "raw_status": payload,
        }

    def parse_result_response(
        self,
        *,
        simulation_payload: dict,
        alpha_payload: dict,
        recordsets_payload: dict,
        job_id: str,
    ) -> dict:
        status = self._map_simulation_status(simulation_payload.get("status"))
        raw_result = {
            "simulation": simulation_payload,
            "alpha": alpha_payload,
            "recordsets": recordsets_payload,
        }
        metrics = self._extract_metrics(alpha_payload, recordsets_payload)
        rejection_reason = simulation_payload.get("message")
        return {
            "job_id": job_id,
            "status": status,
            "metrics": metrics,
            "raw_result": raw_result,
            "rejection_reason": rejection_reason,
            "submission_eligible": None if status != "completed" else None,
            "alpha_id": simulation_payload.get("alpha"),
        }

    def with_retry(self, operation):
        attempt = 0
        while True:
            try:
                return operation()
            except Exception:  # noqa: BLE001
                attempt += 1
                if attempt > self.max_retries:
                    raise
                time.sleep(min(2**attempt, 5))

    def handle_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        remaining = self._min_request_gap_seconds - elapsed
        if remaining > 0:
            time.sleep(remaining)
        self._last_request_at = time.monotonic()

    def _request(
        self,
        method: str,
        url: str,
        *,
        json_payload: dict | list | None,
        auth: tuple[str, str] | None = None,
        allow_retry: bool = True,
    ):
        if allow_retry:
            self.handle_rate_limit()
        request_fn = self.session.post if method.upper() == "POST" else self.session.get
        response = request_fn(
            url,
            headers=self._build_headers(),
            json=json_payload,
            auth=auth,
            verify=self.verify_ssl,
        )
        return response

    def _build_headers(self) -> dict[str, str]:
        token = self.auth_token or os.getenv(self.auth_env)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "wq-tool/0.2.0",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _response_json(self, response) -> dict | list:
        if response.status_code >= 400:
            raise RuntimeError(f"BRAIN API request failed with status {response.status_code}: {response.text}")
        if not response.text.strip():
            return {}
        try:
            payload = response.json()
        except ValueError:
            return {}
        return payload

    def _authentication_state(self) -> dict:
        response = self.session.get(
            self._resolve_authentication_url(),
            headers=self._build_headers(),
            verify=self.verify_ssl,
        )
        if response.status_code == 200:
            payload_raw = self._response_json(response)
            payload = payload_raw if isinstance(payload_raw, dict) else {}
            return {"authenticated": True, "payload": payload}
        if response.status_code == 204:
            return {"authenticated": False, "payload": {}}
        return {"authenticated": False, "payload": {"status_code": response.status_code, "text": response.text}}

    def _save_session_to_disk(self) -> None:
        if self.auth_token:
            return
        self.session_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_url": self.base_url,
            "cookies": [
                {
                    "name": cookie.name,
                    "value": cookie.value,
                    "domain": cookie.domain,
                    "path": cookie.path,
                    "secure": cookie.secure,
                    "expires": cookie.expires,
                }
                for cookie in self.session.cookies
            ],
        }
        self.session_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_session_from_disk(self) -> None:
        if not self.session_path.exists():
            return
        try:
            payload = json.loads(self.session_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        if payload.get("base_url") and str(payload["base_url"]).rstrip("/") != self.base_url:
            return
        for cookie in payload.get("cookies", []):
            self.session.cookies.set(
                cookie["name"],
                cookie["value"],
                domain=cookie.get("domain"),
                path=cookie.get("path", "/"),
            )

    def _clear_session_file(self) -> None:
        if self.session_path.exists():
            self.session_path.unlink()
        self.session.cookies.clear()

    def _notify_persona_required(self, persona_url: str) -> None:
        smtp_config = self._smtp_notification_config()
        if smtp_config is None:
            print("Persona email notification is not configured; skipping email alert.")
            return
        try:
            self.email_service.send_persona_link(persona_url=persona_url, smtp_config=smtp_config)
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to send Persona notification email: {exc}")
            return
        print(f"Sent Persona verification link to {smtp_config.to_email}.")

    def _wait_for_persona_completion(self, persona_url: str):
        print(
            f"Waiting for Persona verification to complete "
            f"(timeout={self.persona_timeout_seconds}s, poll={self.persona_poll_interval_seconds}s)..."
        )
        deadline = time.monotonic() + float(self.persona_timeout_seconds)
        last_response = None
        while time.monotonic() < deadline:
            response = self._request(
                "POST",
                persona_url,
                json_payload=None,
                allow_retry=False,
            )
            if response.status_code in {200, 201, 204}:
                return response
            last_response = response
            time.sleep(float(self.persona_poll_interval_seconds))
        last_status = getattr(last_response, "status_code", "unknown")
        last_text = getattr(last_response, "text", "")
        raise RuntimeError(
            f"Timed out waiting for Persona verification after {self.persona_timeout_seconds}s "
            f"(last status {last_status}: {last_text})"
        )

    def _load_credentials_payload(self) -> dict:
        if self._credentials_payload is not None:
            return self._credentials_payload
        if not self.credentials_file.exists():
            self._credentials_payload = {}
            return self._credentials_payload
        try:
            payload = json.loads(self.credentials_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in credentials file: {self.credentials_file}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Credentials file must contain a JSON object: {self.credentials_file}")
        self._credentials_payload = payload
        return self._credentials_payload

    def _credential_value(self, section: str, key: str) -> str:
        payload = self._load_credentials_payload()
        section_payload = payload.get(section)
        if not isinstance(section_payload, dict):
            return ""
        value = section_payload.get(key)
        return str(value).strip() if value not in (None, "") else ""

    def _smtp_notification_config(self) -> SmtpNotificationConfig | None:
        payload = self._load_credentials_payload()
        section_payload = payload.get("persona_notification")
        if not isinstance(section_payload, dict):
            return None
        from_email = str(
            section_payload.get("from_email")
            or section_payload.get("smtp_username")
            or self._credential_value("brain", "email")
            or ""
        ).strip()
        to_email = str(
            section_payload.get("to_email")
            or self._credential_value("brain", "email")
            or ""
        ).strip()
        try:
            port = int(section_payload.get("smtp_port") or 587)
        except (TypeError, ValueError):
            port = 587
        smtp_config = SmtpNotificationConfig(
            host=str(section_payload.get("smtp_host") or "").strip(),
            port=port,
            username=str(section_payload.get("smtp_username") or "").strip(),
            password=str(section_payload.get("smtp_password") or "").strip(),
            from_email=from_email,
            to_email=to_email,
            use_tls=bool(section_payload.get("use_tls", True)),
        )
        return smtp_config if smtp_config.is_configured() else None

    def _resolve_authentication_url(self) -> str:
        return f"{self.base_url}{self.endpoints.authentication_path}"

    def _resolve_submit_url(self) -> str:
        return f"{self.base_url}{self.endpoints.submit_path}"

    def _resolve_status_url(self, job_id: str) -> str:
        return f"{self.base_url}{self.endpoints.status_path_template.format(job_id=job_id)}"

    def _resolve_alpha_url(self, alpha_id: str) -> str:
        return f"{self.base_url}{self.endpoints.alpha_path_template.format(alpha_id=alpha_id)}"

    def _resolve_recordsets_url(self, alpha_id: str) -> str:
        return f"{self.base_url}{self.endpoints.recordsets_path_template.format(alpha_id=alpha_id)}"

    @staticmethod
    def _map_simulation_status(value: object) -> str:
        normalized = str(value or "").strip().upper()
        mapping = {
            "WAITING": "submitted",
            "SIMULATING": "running",
            "COMPLETE": "completed",
            "WARNING": "completed",
            "ERROR": "failed",
            "FAIL": "failed",
            "TIMEOUT": "timeout",
            "CANCELLED": "failed",
        }
        return mapping.get(normalized, normalized.lower() if normalized else "submitted")

    @staticmethod
    def _extract_metrics(alpha_payload: dict, recordsets_payload: dict) -> dict[str, float | None]:
        metrics = {
            "sharpe": None,
            "fitness": None,
            "turnover": None,
            "drawdown": None,
            "returns": None,
            "margin": None,
        }
        search_space = [alpha_payload]
        if isinstance(recordsets_payload.get("results"), list):
            search_space.extend(item for item in recordsets_payload["results"] if isinstance(item, dict))

        aliases = {
            "sharpe": ("sharpe",),
            "fitness": ("fitness",),
            "turnover": ("turnover",),
            "drawdown": ("drawdown", "maxDrawdown", "max_drawdown"),
            "returns": ("returns", "return", "pnl"),
            "margin": ("margin",),
        }
        for target, keys in aliases.items():
            metrics[target] = _find_first_numeric(search_space, keys)
        return metrics


def _find_first_numeric(payloads: list[dict], keys: tuple[str, ...]) -> float | None:
    for payload in payloads:
        value = _recursive_find_numeric(payload, set(keys))
        if value is not None:
            return value
    return None


def _recursive_find_numeric(value: object, keys: set[str]) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in keys:
                parsed = _optional_float(item)
                if parsed is not None:
                    return parsed
            nested = _recursive_find_numeric(item, keys)
            if nested is not None:
                return nested
    elif isinstance(value, list):
        for item in value:
            nested = _recursive_find_numeric(item, keys)
            if nested is not None:
                return nested
    return None


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
