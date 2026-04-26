from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests


class HttpSessionProtocol(Protocol):
    def get(self, url: str, **kwargs): ...

    def post(self, url: str, **kwargs): ...


@dataclass(slots=True, frozen=True)
class TelegramNotificationConfig:
    bot_token: str
    chat_id: str
    api_base_url: str = "https://api.telegram.org"
    disable_web_page_preview: bool = True

    def is_configured(self) -> bool:
        required = (self.bot_token, self.chat_id, self.api_base_url)
        return all(str(value).strip() for value in required)


@dataclass(slots=True, frozen=True)
class TelegramConfirmationPollResult:
    approved: bool
    declined: bool
    last_update_id: int | None


class TelegramService:
    def __init__(self, session: HttpSessionProtocol | None = None) -> None:
        self.session = session or requests.Session()

    def send_persona_link(self, *, persona_url: str, telegram_config: TelegramNotificationConfig) -> None:
        if not telegram_config.is_configured():
            raise ValueError("Telegram configuration is incomplete; cannot send Persona notification.")

        message = "\n".join(
            [
                "BRAIN yeu cau xac thuc Persona de tiep tuc session API.",
                "",
                "Mo link duoi day de quet mat:",
                persona_url,
                "",
                "Sau khi quet mat xong, tool se tu dong polling de hoan tat dang nhap.",
            ]
        )
        response = self.session.post(
            f"{telegram_config.api_base_url.rstrip('/')}/bot{telegram_config.bot_token}/sendMessage",
            json={
                "chat_id": telegram_config.chat_id,
                "text": message,
                "disable_web_page_preview": telegram_config.disable_web_page_preview,
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Telegram notification failed with status {response.status_code}: {response.text}"
            )

    def send_persona_confirmation_prompt(
        self,
        *,
        prompt_token: str,
        service_name: str,
        telegram_config: TelegramNotificationConfig,
    ) -> None:
        if not telegram_config.is_configured():
            raise ValueError("Telegram configuration is incomplete; cannot send Persona readiness prompt.")

        message = "\n".join(
            [
                f"{service_name}: session BRAIN can het han.",
                "Neu ban SAN SANG xac thuc ngay bay gio thi bam nut ben duoi.",
                "Tool chi xin Persona link sau khi ban xac nhan, de tranh ton quota hang ngay.",
                "",
                f"Neu nut inline khong hien, reply: /ready {prompt_token}",
                f"Neu chua san sang, co the bo qua hoac reply: /later {prompt_token}",
            ]
        )
        response = self.session.post(
            f"{telegram_config.api_base_url.rstrip('/')}/bot{telegram_config.bot_token}/sendMessage",
            json={
                "chat_id": telegram_config.chat_id,
                "text": message,
                "disable_web_page_preview": telegram_config.disable_web_page_preview,
                "reply_markup": {
                    "inline_keyboard": [
                        [{"text": "San sang xac thuc", "callback_data": f"persona_ready:{prompt_token}"}],
                        [{"text": "De sau", "callback_data": f"persona_later:{prompt_token}"}],
                    ]
                },
            },
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Telegram readiness prompt failed with status {response.status_code}: {response.text}"
            )

    def poll_persona_confirmation(
        self,
        *,
        prompt_token: str,
        telegram_config: TelegramNotificationConfig,
        offset: int | None = None,
    ) -> TelegramConfirmationPollResult:
        if not telegram_config.is_configured():
            raise ValueError("Telegram configuration is incomplete; cannot poll Persona readiness prompt.")

        response = self.session.get(
            f"{telegram_config.api_base_url.rstrip('/')}/bot{telegram_config.bot_token}/getUpdates",
            params={"offset": offset} if offset is not None else None,
            timeout=30,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Telegram readiness poll failed with status {response.status_code}: {response.text}"
            )
        payload = response.json()
        updates = payload.get("result") if isinstance(payload, dict) else []
        if not isinstance(updates, list):
            updates = []

        approved = False
        declined = False
        last_update_id: int | None = None
        expected_chat_id = str(telegram_config.chat_id)
        ready_text = f"/ready {prompt_token}".lower()
        later_text = f"/later {prompt_token}".lower()

        for update in updates:
            if not isinstance(update, dict):
                continue
            update_id_raw = update.get("update_id")
            try:
                update_id = int(update_id_raw)
            except (TypeError, ValueError):
                update_id = None
            if update_id is not None and (last_update_id is None or update_id > last_update_id):
                last_update_id = update_id

            callback_query = update.get("callback_query")
            if isinstance(callback_query, dict):
                data = str(callback_query.get("data") or "").strip()
                callback_message = callback_query.get("message") or {}
                callback_chat = callback_message.get("chat") or {}
                if str(callback_chat.get("id") or "") != expected_chat_id:
                    continue
                if data == f"persona_ready:{prompt_token}":
                    approved = True
                    try:
                        self._answer_callback_query(
                            callback_query_id=str(callback_query.get("id") or ""),
                            text="Da nhan xac nhan. Dang xin Persona link...",
                            telegram_config=telegram_config,
                        )
                    except Exception:
                        pass
                elif data == f"persona_later:{prompt_token}":
                    declined = True
                    try:
                        self._answer_callback_query(
                            callback_query_id=str(callback_query.get("id") or ""),
                            text="OK, tool se cho ban san sang roi moi xin link.",
                            telegram_config=telegram_config,
                        )
                    except Exception:
                        pass
                continue

            message = update.get("message")
            if not isinstance(message, dict):
                continue
            chat = message.get("chat") or {}
            if str(chat.get("id") or "") != expected_chat_id:
                continue
            text = str(message.get("text") or "").strip().lower()
            if text == ready_text:
                approved = True
            elif text == later_text:
                declined = True

        return TelegramConfirmationPollResult(
            approved=approved,
            declined=declined,
            last_update_id=last_update_id,
        )

    def _answer_callback_query(
        self,
        *,
        callback_query_id: str,
        text: str,
        telegram_config: TelegramNotificationConfig,
    ) -> None:
        if not callback_query_id:
            return
        response = self.session.post(
            f"{telegram_config.api_base_url.rstrip('/')}/bot{telegram_config.bot_token}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=30,
        )
        if response.status_code >= 400:
            if self._is_expired_callback_query_error(response):
                return
            raise RuntimeError(
                f"Telegram callback acknowledgement failed with status {response.status_code}: {response.text}"
            )

    @staticmethod
    def _is_expired_callback_query_error(response) -> bool:
        if int(getattr(response, "status_code", 0) or 0) != 400:
            return False
        text = str(getattr(response, "text", "") or "").lower()
        return (
            "query is too old" in text
            or "query id is invalid" in text
            or "response timeout expired" in text
        )
