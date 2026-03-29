from __future__ import annotations

import smtplib
from dataclasses import dataclass
from email.message import EmailMessage
from typing import Protocol


class SmtpClientProtocol(Protocol):
    def ehlo(self) -> None: ...

    def starttls(self) -> None: ...

    def login(self, username: str, password: str) -> None: ...

    def send_message(self, message: EmailMessage) -> None: ...

    def __enter__(self): ...

    def __exit__(self, exc_type, exc, tb) -> None: ...


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
