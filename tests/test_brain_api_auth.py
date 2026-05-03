from __future__ import annotations

import builtins
import getpass
import pytest
import json
from pathlib import Path

import requests

from adapters.brain_api_adapter import BiometricsThrottled, BrainApiAdapter, ConcurrentSimulationLimitExceeded
from services.email_service import TelegramNotificationConfig, TelegramService


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


class FakeTelegramService:
    def __init__(self) -> None:
        self.calls: list[tuple[str, TelegramNotificationConfig]] = []

    def send_persona_link(self, *, persona_url: str, telegram_config: TelegramNotificationConfig) -> None:
        self.calls.append((persona_url, telegram_config))


def test_telegram_service_accepts_ready_callback_even_if_ack_is_expired() -> None:
    telegram = TelegramService(
        session=FakeSession(
            get_queue=[
                FakeResponse(
                    200,
                    json_data={
                        "ok": True,
                        "result": [
                            {
                                "update_id": 123,
                                "callback_query": {
                                    "id": "cbq-1",
                                    "data": "persona_ready:abc123",
                                    "message": {"chat": {"id": "99887766"}},
                                },
                            }
                        ],
                    },
                )
            ],
            post_queue=[
                FakeResponse(
                    400,
                    json_data={"ok": False, "error_code": 400},
                    text='{"ok":false,"error_code":400,"description":"Bad Request: query is too old and response timeout expired or query ID is invalid"}',
                )
            ],
        )
    )

    result = telegram.poll_persona_confirmation(
        prompt_token="abc123",
        telegram_config=TelegramNotificationConfig(bot_token="bot-token", chat_id="99887766"),
    )

    assert result.approved is True
    assert result.declined is False
    assert result.last_update_id == 123


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


def test_brain_api_adapter_skips_persona_notification_when_telegram_is_not_configured(
    tmp_path: Path, monkeypatch
) -> None:
    session_path = tmp_path / "brain_session.json"
    credentials_path = tmp_path / "brain_credentials.json"
    credentials_path.write_text(
        """
        {
          "brain": {
            "email": "user@example.com",
            "password": "secret"
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
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(credentials_path),
        open_browser_for_persona=False,
        persona_poll_interval_seconds=0.01,
        persona_timeout_seconds=1,
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
    assert adapter.send_persona_notification("https://persona.example/manual") is False


def test_brain_api_adapter_sends_persona_notification_via_telegram(tmp_path: Path, monkeypatch) -> None:
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
            "telegram_chat_id": "99887766"
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
    fake_telegram_service = FakeTelegramService()
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(session_path),
        credentials_file=str(credentials_path),
        open_browser_for_persona=False,
        persona_poll_interval_seconds=0.01,
        persona_timeout_seconds=1,
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


def test_brain_api_adapter_surfaces_concurrent_simulation_limit(tmp_path: Path, monkeypatch) -> None:
    fake_session = FakeSession(
        get_queue=[],
        post_queue=[
            FakeResponse(
                429,
                json_data={"detail": "CONCURRENT_SIMULATION_LIMIT_EXCEEDED"},
                text='{"detail":"CONCURRENT_SIMULATION_LIMIT_EXCEEDED"}',
                url="https://api.worldquantbrain.com/simulations",
            )
        ],
    )
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=fake_session,
        session_path=str(tmp_path / "brain_session.json"),
        open_browser_for_persona=False,
    )
    monkeypatch.setattr(adapter, "ensure_authenticated", lambda **kwargs: {"status": "ready"})

    with pytest.raises(ConcurrentSimulationLimitExceeded) as exc_info:
        adapter.submit_simulation(
            "rank(close)",
            {
                "region": "USA",
                "universe": "TOP3000",
                "delay": 1,
                "decay": 4,
                "neutralization": "sector",
                "truncation": 0.08,
            },
        )

    assert exc_info.value.cooldown_seconds == 180


def test_brain_api_adapter_parse_result_preserves_checks_and_submission_eligible(tmp_path: Path) -> None:
    adapter = BrainApiAdapter(
        base_url="https://api.worldquantbrain.com",
        session=FakeSession(get_queue=[], post_queue=[]),
        session_path=str(tmp_path / "brain_session.json"),
        open_browser_for_persona=False,
    )

    parsed = adapter.parse_result_response(
        simulation_payload={"id": "sim-1", "status": "COMPLETE"},
        alpha_payload={
            "sharpe": 0.9,
            "fitness": 0.4,
            "is": {
                "submissionEligible": False,
                "checks": [
                    {"name": "REVERSION_COMPONENT", "result": "WARNING", "message": "reversion"},
                ],
            },
        },
        recordsets_payload={},
        job_id="job-1",
    )

    assert parsed["submission_eligible"] is False
    assert parsed["derived_submit_ready"] is False
    assert parsed["blocking_warning_checks"] == ["REVERSION_COMPONENT"]
    assert parsed["rejection_reason"] == "reversion"


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
