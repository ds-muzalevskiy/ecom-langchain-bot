from __future__ import annotations

import json
import os
import pathlib
import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

try:
    from .brand_chain import BASE, STYLE, ask
except ImportError:  # allow `python src/style_eval.py`
    from brand_chain import BASE, STYLE, ask

load_dotenv(".env.example", override=True)

REPORTS = BASE / "reports"
REPORTS.mkdir(exist_ok=True)


# ------------------------
# Rule-checks
# ------------------------

_EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF]")
_INFORMAL_RE = re.compile(r"\b(ты|тебе|тебя|твой|твоя|твои|твоё)\b", re.IGNORECASE)
_SLANG_WORDS = ["чё", "чо", "влеп", "конечн", "давай"]


@dataclass
class RuleResult:
    score: int
    issues: Dict[str, bool]


def rule_checks(text: str) -> RuleResult:
    score = 100
    issues: Dict[str, bool] = {}

    # 1) Без эмодзи
    emoji = bool(_EMOJI_RE.search(text))
    issues["emoji_detected"] = emoji
    if emoji:
        score -= 20

    # 2) Без крика!!! (и без множественных !)
    shout = "!!!" in text or bool(re.search(r"!{2,}", text))
    issues["exclamation_burst"] = shout
    if shout:
        score -= 10

    # 3) На "Вы" (не на "ты")
    informal = bool(_INFORMAL_RE.search(text))
    issues["second_person_informal"] = informal
    if informal:
        score -= 20

    # 4) Сленг (ассистент сам не должен использовать)
    slang = any(w in text.lower() for w in _SLANG_WORDS)
    issues["slang_detected"] = slang
    if slang:
        score -= 10

    # 5) Ограничение по предложениям
    max_sent = int(STYLE.get("tone", {}).get("sentences_max", 3) or 3)
    # очень простое разбиение
    sent_count = len([s for s in re.split(r"[.!?]+", text) if s.strip()])
    too_many = sent_count > max_sent
    issues["sentences_over_limit"] = too_many
    if too_many:
        score -= 10

    # 6) Ограничение по длине
    too_long = len(text) > 600
    issues["too_long_chars"] = too_long
    if too_long:
        score -= 10

    return RuleResult(score=max(score, 0), issues=issues)


# ------------------------
# LLM-оценка
# ------------------------

class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str


def _make_grader() -> ChatOpenAI:
    model = os.getenv("OPENAI_EVAL_MODEL") or "gpt-4o"
    return ChatOpenAI(model=model, temperature=0)


GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", "Ты — строгий ревьюер соответствия голосу бренда {brand}."),
        (
            "system",
            "Персона: {persona}. Ограничения: максимум {sentences_max} предложений. "
            "Нельзя: эмодзи и крик (много !). Всегда обращение на 'Вы'. "
            "Если нет данных из FAQ/заказов — корректный отказ по fallback.",
        ),
        (
            "human",
            "Ответ ассистента:\n{answer}\n\nОцени соответствие голосу бренда по шкале 0..100 "
            "и кратко (2–4 пункта) объясни почему.",
        ),
    ]
)


def llm_grade(text: str) -> Grade:
    if os.getenv("SKIP_LLM_GRADE") == "1" or not os.getenv("OPENAI_API_KEY"):
        # offline / без ключа
        return Grade(score=100, notes="LLM-оценка пропущена (SKIP_LLM_GRADE=1 или нет ключа).")

    grader = _make_grader()
    parser = grader.with_structured_output(Grade)
    chain = GRADE_PROMPT | parser
    return chain.invoke(
        {
            "brand": STYLE.get("brand", "Shoply"),
            "persona": STYLE.get("tone", {}).get("persona", ""),
            "sentences_max": STYLE.get("tone", {}).get("sentences_max", 3),
            "answer": text,
        }
    )


# ------------------------
# Batch eval + report
# ------------------------

def eval_batch(prompts: List[str]) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    history = []
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    
    for p in prompts:
        reply, history, _ = ask(p, model, history=None)  # без истории — чистая проверка
        r = rule_checks(reply.answer)
        g = llm_grade(reply.answer)
        final = int(0.4 * r.score + 0.6 * g.score)

        cases.append(
            {
                "user": p,
                "assistant": reply.answer,
                "actions": reply.actions,
                "tone_model": reply.tone,
                "rule_issues": r.issues,
                "rule_score": r.score,
                "llm_score": g.score,
                "final": final,
                "llm_evaluation": g.notes,
            }
        )

    mean_final = round(statistics.mean(c["final"] for c in cases), 2) if cases else 0.0

    # сводка по нарушениям
    violations: Dict[str, int] = {}
    for c in cases:
        for k, v in c["rule_issues"].items():
            if v:
                violations[k] = violations.get(k, 0) + 1

    pass_count = sum(1 for c in cases if c["final"] >= 80)
    out: Dict[str, Any] = {
        "generated_at": pathlib.Path().resolve().as_posix(),
        "mean_final": mean_final,
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "passed": pass_count,
            "pass_rate": round(pass_count / len(cases), 3) if cases else 0.0,
            "violations": violations,
        },
    }

    (REPORTS / "style_eval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out


if __name__ == "__main__":
    prompts = (BASE / "data" / "eval_prompts.txt").read_text(encoding="utf-8").strip().splitlines()
    report = eval_batch(prompts)
    print("Средний балл:", report["mean_final"])
    print("Отчёт:", REPORTS / "style_eval.json")
