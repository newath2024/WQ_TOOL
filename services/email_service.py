from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol

import requests


class SmtpClientProtocol(Protocol):
    def ehlo(self) -> None: ...

    def starttls(self) -> None: ...

    def login(self, username: str, password: str) -> None: ...

    def send_message(self, message: EmailMessage) -> None: ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc, tb) -> None: ...


class HttpSessionProtocol(Protocol):
    def post(self, url: str, **kwargs): ...


@dataclass(slots=True, frozen=True)
class SmtpNotificationConfig:
    host: str
    port: int
    username: str
    password: str
    from_email: str
    to_email: str
    use_tls: bool = True

    def is_configured(self) -> bool:
        required = (self.host, self.username, self.password, self.from_email, self.to_email)
        return self.port > 0 and all(str(value).strip() for value in required)


@dataclass(slots=True, frozen=True)
class TelegramNotificationConfig:
    bot_token: str
    chat_id: str
    api_base_url: str = "https://api.telegram.org"
    disable_web_page_preview: bool = True

    def is_configured(self) -> bool:
        required = (self.bot_token, self.chat_id, self.api_base_url)
        return all(str(value).strip() for value in required)


class EmailService:
    def __init__(self, smtp_factory=None) -> None:
        self.smtp_factory = smtp_factory or smtplib.SMTP

    def send_persona_link(self, *, persona_url: str, smtp_config: SmtpNotificationConfig) -> None:
        if not smtp_config.is_configured():
            raise ValueError("SMTP configuration is incomplete; cannot send Persona notification email.")

        message = EmailMessage()
        message["Subject"] = "BRAIN Persona verification required"
        message["From"] = smtp_config.from_email
        message["To"] = smtp_config.to_email
        message.set_content(
            "\n".join(
                [
                    "BRAIN yeu cau xac thuc Persona de tiep tuc session API.",
                    "",
                    "Mo link duoi day de quet mat:",
                    persona_url,
                    "",
                    "Sau khi quet mat xong, tool se tu dong polling de hoan tat dang nhap.",
                ]
            )
        )

        with self.smtp_factory(smtp_config.host, smtp_config.port, timeout=30) as client:
            client.ehlo()
            if smtp_config.use_tls:
                client.starttls()
                client.ehlo()
            client.login(smtp_config.username, smtp_config.password)
            client.send_message(message)


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
