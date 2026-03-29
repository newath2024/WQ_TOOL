from __future__ import annotations

import builtins
import getpass
from pathlib import Path

import requests

from adapters.brain_api_adapter import BrainApiAdapter


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
    )
    inputs = iter(["user@example.com", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(inputs))
    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "secret")
    monkeypatch.setattr("webbrowser.open", lambda url: opened_urls.append(url))

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


def _success_login_response(session: FakeSession, url: str, kwargs: dict) -> FakeResponse:
    session.cookies.set("t", "jwt-cookie", domain="api.worldquantbrain.com", path="/")
    return FakeResponse(201, json_data={"user": {"id": "user-1"}, "token": {"expiry": 1000}}, url=url)
