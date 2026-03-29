from __future__ import annotations

import builtins
import getpass
import pytest
import json
from pathlib import Path

import requests

from adapters.brain_api_adapter import BiometricsThrottled, BrainApiAdapter
from services.email_service import SmtpNotificationConfig, TelegramNotificationConfig


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        *,
        json_data: dict | None = None,
        headers: dict[str, str] | None = None,
        text: str = "",
        url: str = "https://api.worldquantbrain.com/authentication",
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.headers = headers or {}
        self.text = text or ("" if json_data is None else str(json_data))
        self.url = url

    def json(self):
        return self._json_data


class FakeSession:
    def __init__(self, *, get_queue: list, post_queue: list) -> None:
        self.cookies = requests.cookies.RequestsCookieJar()
        self._get_queue = list(get_queue)
        self._post_queue = list(post_queue)
        self.calls: list[tuple[str, str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append(("GET", url, kwargs))
        item = self._get_queue.pop(0)
        if callable(item):
            return item(self, url, kwargs)
        item.url = url
        return item

    def post(self, url: str, **kwargs):
        self.calls.append(("POST", url, kwargs))
        item = self._post_queue.pop(0)
        if callable(item):
            return item(self, url, kwargs)
        item.url = url
        return item


class FakeEmailService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SmtpNotificationConfig]] = []

    def send_persona_link(self, *, persona_url: str, smtp_config: SmtpNotificationConfig) -> None:
        self.calls.append((persona_url, smtp_config))


class FakeTelegramService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, TelegramNotificationConfig]] = []

    def send_persona_link(self, *, persona_url: str, telegram_config: TelegramNotificationConfig) -> None:
        self.calls.append((persona_url, telegram_config))


def test_brain_api_adapter_prompts_and_saves_session(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-1"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[_success_login_response],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        open_browser_for_persona=False,
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": "user@example.com")
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "secret")

    result = adapter.ensure_authenticated()

    assert result["mode"] == "interactive"
    assert session_path.exists()
    assert fake_session.cookies.get("t") == "jwt-cookie"


def test_brain_api_adapter_handles_persona_flow(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-2"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[
            FakeResponse(
                401,
                json_data={},
                headers={
                    "WWW-Authenticate": "persona",
                    "Location": "/authentication/persona?inquiry=inq_123",
                },
                url="https://api.worldquantbrain.com/authentication",
            ),
            _success_login_response,
        ],
    )
    opened_urls: list[str] = []
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        open_browser_for_persona=True,
        persona_poll_interval_seconds=0.01,
        persona_timeout_seconds=1,
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": "user@example.com")
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "secret")
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = adapter.ensure_authenticated()

    assert result["mode"] == "interactive"
    assert opened_urls == ["https://api.worldquantbrain.com/authentication/persona?inquiry=inq_123"]
    assert fake_session.cookies.get("t") == "jwt-cookie"


def test_brain_api_adapter_uses_visible_password_fallback(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-3"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[_success_login_response],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        open_browser_for_persona=False,
    )
    inputs = iter(["user@example.com", "visible-secret"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")

    result = adapter.ensure_authenticated()

    assert result["mode"] == "interactive"
    assert fake_session.cookies.get("t") == "jwt-cookie"


def test_brain_api_adapter_accepts_visible_password_prompt(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-4"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[_success_login_response],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        open_browser_for_persona=False,
    )
    inputs = iter(["user@example.com", "visible-secret"])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "should-not-be-used")

    result = adapter.ensure_authenticated(show_password=True)

    assert result["mode"] == "interactive"
    assert fake_session.cookies.get("t") == "jwt-cookie"


def test_brain_api_adapter_reads_credentials_file_and_sends_persona_email(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    credentials_path = tmp_path / "brain_credentials.json"
    credentials_path.write_text(
        """
        {
          "brain": {
            "email": "user@example.com",
            "password": "secret"
          },
          "persona_notification": {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "notify@example.com",
            "smtp_password": "app-password",
            "from_email": "notify@example.com",
            "to_email": "owner@example.com",
            "use_tls": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-5"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[
            FakeResponse(
                401,
                json_data={},
                headers={
                    "WWW-Authenticate": "persona",
                    "Location": "/authentication/persona?inquiry=inq_999",
                },
                url="https://api.worldquantbrain.com/authentication",
            ),
            _success_login_response,
        ],
    )
    fake_email_service = FakeEmailService()
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(credentials_path),
        open_browser_for_persona=False,
        persona_poll_interval_seconds=0.01,
        persona_timeout_seconds=1,
        email_service=fake_email_service,
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(AssertionError("input not expected")))
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("getpass not expected")),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = adapter.ensure_authenticated()

    assert result["mode"] == "interactive"
    assert fake_session.cookies.get("t") == "jwt-cookie"
    assert len(fake_email_service.calls) == 1
    persona_url, smtp_config = fake_email_service.calls[0]
    assert persona_url == "https://api.worldquantbrain.com/authentication/persona?inquiry=inq_999"
    assert smtp_config.to_email == "owner@example.com"


def test_brain_api_adapter_prefers_telegram_notification_when_configured(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    credentials_path = tmp_path / "brain_credentials.json"
    credentials_path.write_text(
        """
        {
          "brain": {
            "email": "user@example.com",
            "password": "secret"
          },
          "persona_notification": {
            "telegram_bot_token": "bot-token-123",
            "telegram_chat_id": "99887766",
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_username": "notify@example.com",
            "smtp_password": "app-password",
            "from_email": "notify@example.com",
            "to_email": "owner@example.com",
            "use_tls": true
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(200, json_data={"user": {"id": "user-telegram"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[
            FakeResponse(
                401,
                json_data={},
                headers={
                    "WWW-Authenticate": "persona",
                    "Location": "/authentication/persona?inquiry=inq_tg",
                },
                url="https://api.worldquantbrain.com/authentication",
            ),
            _success_login_response,
        ],
    )
    fake_email_service = FakeEmailService()
    fake_telegram_service = FakeTelegramService()
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(credentials_path),
        open_browser_for_persona=False,
        persona_poll_interval_seconds=0.01,
        persona_timeout_seconds=1,
        email_service=fake_email_service,
        telegram_service=fake_telegram_service,
    )
    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(AssertionError("input not expected")))
    monkeypatch.setattr(
        getpass,
        "getpass",
        lambda prompt="": (_ for _ in ()).throw(AssertionError("getpass not expected")),
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = adapter.ensure_authenticated()

    assert result["mode"] == "interactive"
    assert fake_email_service.calls == []
    assert len(fake_telegram_service.calls) == 1
    persona_url, telegram_config = fake_telegram_service.calls[0]
    assert persona_url == "https://api.worldquantbrain.com/authentication/persona?inquiry=inq_tg"
    assert telegram_config.chat_id == "99887766"


def test_brain_api_adapter_reloads_session_file_when_updated_externally(tmp_path: Path) -> None:
    session_path = tmp_path / "brain_session.json"
    _write_session_file(session_path, cookie_value="stale-cookie")

    def auth_state_from_cookie(session: FakeSession, url: str, kwargs: dict) -> FakeResponse:
        del url, kwargs
        cookie_value = session.cookies.get("t")
        if cookie_value == "fresh-cookie-v2":
            return FakeResponse(200, json_data={"user": {"id": "user-6"}, "token": {"expiry": 1000}})
        return FakeResponse(204, json_data={})

    fake_session = FakeSession(
        get_queue=[
            auth_state_from_cookie,
            auth_state_from_cookie,
        ],
        post_queue=[],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        email_env="TEST_BRAIN_API_EMAIL",
        password_env="TEST_BRAIN_API_PASSWORD",
        credentials_file=str(tmp_path / "missing.json"),
        open_browser_for_persona=False,
    )

    with pytest.raises(RuntimeError, match="non-interactive authentication requires credentials"):
        adapter.ensure_authenticated(interactive=False)

    _write_session_file(session_path, cookie_value="fresh-cookie-v2")

    result = adapter.ensure_authenticated(interactive=False)

    assert result["mode"] == "session_cookie"
    assert fake_session.cookies.get("t") == "fresh-cookie-v2"


def test_brain_api_adapter_retries_transient_auth_probe_disconnect(tmp_path: Path, monkeypatch) -> None:
    session_path = tmp_path / "brain_session.json"
    _write_session_file(session_path, cookie_value="fresh-cookie")

    fake_session = FakeSession(
        get_queue=[
            lambda session, url, kwargs: (_ for _ in ()).throw(
                requests.ConnectionError("Remote end closed connection without response")
            ),
            FakeResponse(200, json_data={"user": {"id": "user-7"}, "token": {"expiry": 1000}}),
        ],
        post_queue=[],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(tmp_path / "missing.json"),
        open_browser_for_persona=False,
        max_retries=1,
    )
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    result = adapter.ensure_authenticated(interactive=False)

    assert result["mode"] == "session_cookie"
    assert fake_session.cookies.get("t") == "fresh-cookie"


def test_brain_api_adapter_surfaces_biometrics_throttle(tmp_path: Path) -> None:
    session_path = tmp_path / "brain_session.json"
    fake_session = FakeSession(
        get_queue=[
            FakeResponse(204, json_data={}),
            FakeResponse(204, json_data={}),
        ],
        post_queue=[
            FakeResponse(
                429,
                json_data={"detail": "BIOMETRICS_THROTTLED"},
                headers={"Retry-After": "45"},
            )
        ],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(tmp_path / "missing.json"),
        open_browser_for_persona=False,
    )
    adapter._credentials_payload = {"brain": {"email": "user@example.com", "password": "secret"}}

    with pytest.raises(BiometricsThrottled) as exc_info:
        adapter.ensure_authenticated(interactive=False)

    assert exc_info.value.retry_after_seconds == 45


def _success_login_response(session: FakeSession, url: str, kwargs: dict) -> FakeResponse:
    session.cookies.set("t", "jwt-cookie", domain="api.worldquantbrain.com", path="/")
    return FakeResponse(201, json_data={"user": {"id": "user-1"}, "token": {"expiry": 1000}}, url=url)


def _write_session_file(path: Path, *, cookie_value: str) -> None:
    path.write_text(
        json.dumps(
            {
                "base_url": "https://api.worldquantbrain.com",
                "cookies": [
                    {
                        "name": "t",
                        "value": cookie_value,
                        "domain": "api.worldquantbrain.com",
                        "path": "/",
                        "secure": True,
                        "expires": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
