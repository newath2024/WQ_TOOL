from __future__ import annotations

from infrastructure.notifications.telegram import (
    HttpSessionProtocol,
    TelegramConfirmationPollResult,
    TelegramNotificationConfig,
    TelegramService,
)

# The module name is kept for backward-compatible imports, but Persona
# notifications are Telegram-only now and the implementation lives below services.

__all__ = [
    "HttpSessionProtocol",
    "TelegramConfirmationPollResult",
    "TelegramNotificationConfig",
    "TelegramService",
]
