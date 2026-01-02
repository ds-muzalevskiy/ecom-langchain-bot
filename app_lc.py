from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.brand_chain import AssistantReply, BASE, STYLE, ask


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class UsageCounters:
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def add(self, p: int, c: int) -> None:
        self.prompt += int(p)
        self.completion += int(c)
        self.total += int(p) + int(c)


class JsonlLogger:
    def __init__(self, log_dir: Path) -> None:
        log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = uuid.uuid4().hex
        self.path = log_dir / f"session_{self.session_id}.jsonl"

    def write(self, role: str, content: Any, **extra: Any) -> None:
        entry: Dict[str, Any] = {
            "timestamp": utc_now_iso(),
            "role": role,
            "content": content,
        }
        entry.update(extra)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _get_openai_callback():
    try:
        from langchain_community.callbacks import get_openai_callback  # type: ignore

        return get_openai_callback
    except Exception:
        try:
            from langchain.callbacks import get_openai_callback  # type: ignore

            return get_openai_callback
        except Exception:
            return None


HELP = """Доступные команды:
  /help              справка
  /order <id>         статус заказа (пример: /order 12345)
  /reset             очистить историю
  /exit              выйти

Подсказка: можно задавать вопросы про доставку, возврат, оплату, промокоды.
"""


def render(reply: AssistantReply) -> str:
    # Пользователю показываем только answer (+ actions); поле tone остаётся служебным
    out = [reply.answer.strip()]
    if reply.actions:
        out.append("\nСледующие шаги:")
        out.extend([f"- {a}" for a in reply.actions[:3]])
    return "\n".join(out).strip()


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv(".env.example", override=True)
    brand = os.getenv("BRAND_NAME", STYLE.get("brand", "Shoply"))
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY не задан.")
        return 2

    logger = JsonlLogger(BASE / "logs")
    usage_total = UsageCounters()
    get_cb = _get_openai_callback()

    history = []  # type: List[Any]

    print(f"Поддержка {brand}")
    print("/help — команды, /exit — выход\n")

    logger.write("system", "session_start", brand=brand, model=model)

    while True:
        try:
            text = input("Вы: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            break

        if not text:
            continue

        if text.lower() in {"/exit", "exit", "quit", "/quit", "выход"}:
            break

        if text.lower() == "/help":
            print("Бот:\n" + HELP)
            logger.write("assistant", HELP)
            continue

        if text.lower() == "/reset":
            history.clear()
            msg = "История очищена. Чем помочь?"
            print("Бот:", msg)
            logger.write("assistant", msg)
            continue

        logger.write("user", text)

        # Запрос + учёт токенов
        if get_cb is not None:
            with get_cb() as cb:
                reply, history, _ = ask(text, model, history)
            usage = {
                "prompt": int(getattr(cb, "prompt_tokens", 0)),
                "completion": int(getattr(cb, "completion_tokens", 0)),
                "total": int(getattr(cb, "total_tokens", 0)),
            }
        else:
            reply, history, _ = ask(text, history)
            usage = {"prompt": 0, "completion": 0, "total": 0}

        usage_total.add(usage["prompt"], usage["completion"])

        shown = render(reply)
        print("Бот:\n" + shown + "\n")

        logger.write(
            "assistant",
            reply.model_dump(),
            usage=usage,
        )

    logger.write(
        "system",
        "session_end",
        usage={"prompt": usage_total.prompt, "completion": usage_total.completion, "total": usage_total.total},
    )

    print("\nСессия завершена.")
    print("Лог:", logger.path)
    print("Токены:", {"prompt": usage_total.prompt, "completion": usage_total.completion, "total": usage_total.total})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
