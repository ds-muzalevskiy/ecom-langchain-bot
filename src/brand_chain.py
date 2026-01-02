from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data"

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


STYLE: Dict[str, Any] = yaml.safe_load((DATA / "style_guide.yaml").read_text(encoding="utf-8"))
FAQ: List[Dict[str, str]] = _read_json(DATA / "faq.json")
ORDERS: Dict[str, Dict[str, Any]] = _read_json(DATA / "orders.json")
FEW_SHOTS: List[Dict[str, str]] = _read_jsonl(DATA / "few_shots.jsonl")


class AssistantReply(BaseModel):
    answer: str = Field(..., description="Краткий ответ пользователю")
    tone: str = Field(..., description="да/нет + одна фраза почему")
    actions: List[str] = Field(default_factory=list, description="0–3 следующих шага")


# --- простая нормализация для FAQ ---
_PUNCT_RE = re.compile(r"[^\w\sа-яА-ЯёЁ-]+", re.UNICODE)


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _tokens(text: str) -> List[str]:
    return [w for w in _normalize(text).split(" ") if w]




def _faq_best(question: str, threshold: float = 0.2) -> Optional[Tuple[str, str]]:
    """Возвращает (faq_q, faq_a) если есть уверенное совпадение.

    1) Сначала — простой роутинг по ключевым словам (для живых формулировок).
    2) Затем — Jaccard-похожесть по токенам.
    """

    q_norm = _normalize(question)
    if not q_norm:
        return None

    # --- роутинг по ключевым словам ---
    routes = [
        ("промокод", "промокод"),
        ("достав", "доставка"),
        ("возврат", "возврат"),
        ("оплат", "оплата"),
        ("адрес", "адрес"),
    ]
    for needle, key in routes:
        if needle in q_norm:
            for item in FAQ:
                q = str(item.get("q", "")).strip()
                a = str(item.get("a", "")).strip()
                if not q or not a:
                    continue
                if key in _normalize(q):
                    return (q, a)

    # --- похожесть по токенам (Jaccard) ---
    q_tokens = set(_tokens(question))
    if not q_tokens:
        return None

    best_score = 0.0
    best_pair: Optional[Tuple[str, str]] = None
    for item in FAQ:
        q = str(item.get("q", "")).strip()
        a = str(item.get("a", "")).strip()
        if not q or not a:
            continue
        faq_tokens = set(_tokens(q))
        inter = len(q_tokens & faq_tokens)
        union = len(q_tokens | faq_tokens)
        score = inter / union if union else 0.0
        if score > best_score:
            best_score = score
            best_pair = (q, a)
    if best_score >= threshold:
        return best_pair
    return None


_ORDER_CMD_RE = re.compile(r"^/order\s+(\d+)$", re.IGNORECASE)
_ORDER_IN_TEXT_RE = re.compile(r"(?:заказ\s*#?|order\s*#?)\s*(\d{3,})", re.IGNORECASE)


def _order_id_from_text(text: str) -> Optional[str]:
    m = _ORDER_CMD_RE.match(text.strip())
    if m:
        return m.group(1)
    m2 = _ORDER_IN_TEXT_RE.search(text)
    if m2:
        return m2.group(1)
    return None


def _order_to_text(order_id: str, order: Dict[str, Any]) -> str:
    status = order.get("status")
    if status == "in_transit":
        carrier = order.get("carrier", "служба доставки")
        eta = order.get("eta_days", "?")
        return f"Заказ {order_id} в пути ({carrier}). Ожидаемый срок: {eta} дн."
    if status == "delivered":
        delivered_at = order.get("delivered_at", "ранее")
        return f"Заказ {order_id} доставлен {delivered_at}."
    if status == "processing":
        note = order.get("note", "Ожидает обработки")
        return f"Заказ {order_id} обрабатывается. {note}"
    return f"Статус заказа {order_id}: {status}"


def _few_shot_to_structured(text: str) -> AssistantReply:
    """Преобразуем текстовый few-shot (строка) в структурированный объект."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return AssistantReply(answer="", tone="да — соответствует гайду.", actions=[])
    answer = lines[0]
    actions: List[str] = []
    for l in lines[1:]:
        if l.startswith("-"):
            actions.append(re.sub(r"^-\s*", "", l).strip())
    return AssistantReply(answer=answer, tone="да — кратко и по делу.", actions=actions[:3])


def _build_prompt() -> ChatPromptTemplate:
    brand = STYLE.get("brand", "Shoply")
    persona = STYLE.get("tone", {}).get("persona", "вежливый, деловой")
    sentences_max = int(STYLE.get("tone", {}).get("sentences_max", 3))
    avoid = STYLE.get("tone", {}).get("avoid", [])
    must_include = STYLE.get("tone", {}).get("must_include", [])
    no_data = STYLE.get("fallback", {}).get("no_data", "У меня нет точной информации. Могу подключить оператора или оформить запрос.")

    system_rules = (
        f"Вы — бренд-ассистент магазина {brand}. Тон: {persona}.\n"
        f"Ограничения: максимум {sentences_max} предложения в поле answer.\n"
        "Всегда на 'Вы'. Не используйте эмодзи и многократные восклицания.\n"
        "Не придумывайте факты и правила. Используйте ТОЛЬКО данные из контекста (FAQ/заказы).\n"
        "Если в контексте есть ORDER=NOT_FOUND — скажите, что заказ не найден, и предложите проверить номер/подключить оператора.\n"
        f"Если данных нет, верните fallback дословно: '{no_data}'.\n\n"
        "Формат ответа: строго JSON, поля:\n"
        "- answer: короткий ответ\n"
        "- tone: 'да/нет — ...' одна фраза почему\n"
        "- actions: список 0–3 следующих шагов\n"
    )

    style_hints = (
        f"Избегайте: {', '.join(avoid) if avoid else '—'}.\n"
        f"Обязательно: {', '.join(must_include) if must_include else '—'}."
    )

    messages: List[Any] = [
        ("system", system_rules),
        ("system", style_hints),
    ]

    for ex in FEW_SHOTS:
        u = str(ex.get("user", "")).strip()
        a = str(ex.get("assistant", "")).strip()
        if not u or not a:
            continue
        structured = _few_shot_to_structured(a).model_dump()
        messages.append(HumanMessage(content=u))
        messages.append(AIMessage(content=json.dumps(structured, ensure_ascii=False)))

    prompt = ChatPromptTemplate.from_messages(
        [
            *messages,
            MessagesPlaceholder("history"),
            (
                "human",
                "Запрос пользователя:\n{input}\n\nКонтекст (единственный источник фактов):\n{context}",
            ),
        ]
    )
    return prompt


PROMPT = _build_prompt()


def _make_llm(model: str) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=0)


def _context_for(text: str) -> str:
    """Собирает контекст (единственный источник фактов) для промпта.

    Ответы должны генерироваться LLM в едином голосе бренда.
    Поэтому мы НЕ возвращаем детерминированные ответы по FAQ/заказам — вместо этого
    передаём данные в контекст и просим модель оформить ответ строго по гайду.
    """

    # 1) /order или упоминание номера в тексте
    order_id = _order_id_from_text(text)
    if order_id:
        order = ORDERS.get(order_id)
        no_data = STYLE.get("fallback", {}).get(
            "no_data", "У меня нет точной информации. Могу подключить оператора или оформить запрос."
        )
        if not order:
            return f"ORDER_ID={order_id}\nORDER=NOT_FOUND\nFALLBACK={no_data}"
        return f"ORDER_ID={order_id}\nORDER={json.dumps(order, ensure_ascii=False)}"

    # 2) FAQ (если нашли уверенное совпадение — даём паре q/a статус единственного источника фактов)
    best = _faq_best(text)
    if best:
        faq_q, faq_a = best
        return f"FAQ_MATCH_Q={faq_q}\nFAQ_MATCH_A={faq_a}"

    # 3) нет данных -> модель должна вернуть fallback (из системного промпта) без домыслов
    return "FAQ_MATCH=NONE\nORDER_MATCH=NONE"



def ask(
     user_text: str, model:str, history: Optional[List[BaseMessage]] = None
) -> Tuple[AssistantReply, List[BaseMessage], Dict[str, int]]:
    """Отвечает на запрос с учётом истории.

    Возвращает: (reply, new_history, usage_tokens)
    usage_tokens: dict prompt/completion/total (здесь best-effort нули;
    точный учёт делается в app_lc.py через callback)
    """

    history = list(history or [])
    context = _context_for(user_text)
    
    llm = _make_llm(model)

    chain = PROMPT | llm.with_structured_output(AssistantReply)
    reply: AssistantReply = chain.invoke({"input": user_text, "context": context, "history": history[-12:]})

    # --- лёгкая нормализация/защита от выхода за формат ---
    reply.answer = reply.answer.strip()
    reply.tone = reply.tone.strip()
    reply.actions = [a.strip() for a in (reply.actions or []) if a and a.strip()][:3]

    # Если контекст говорит, что данных нет, то требуем fallback (без домыслов)
    if "FAQ_MATCH=NONE" in context and "ORDER_MATCH=NONE" in context:
        no_data = STYLE.get("fallback", {}).get(
            "no_data", "У меня нет точной информации. Могу подключить оператора или оформить запрос."
        )
        # Если модель не вернула fallback дословно
        if no_data not in reply.answer:
            reply.answer = no_data
            if not reply.actions:
                reply.actions = ["Опишите вопрос подробнее", "Я подключу оператора при необходимости"]
            if not reply.tone:
                reply.tone = "да — вежливо и без догадок."

    # Если заказ не найден — не допускаем домыслов
    if "ORDER=NOT_FOUND" in context:
        if "не найден" not in reply.answer.lower():
            # мягко приводим к ожидаемому сообщению
            m = re.search(r"ORDER_ID=(\d+)", context)
            order_id = m.group(1) if m else ""
            reply.answer = f"Заказ {order_id} не найден.".strip()
        if not reply.actions:
            reply.actions = ["Проверьте номер заказа", "Если номер верный — я подключу оператора"]
        if not reply.tone:
            reply.tone = "да — вежливо и без догадок."

    # usage считаем в app_lc.py через callback; тут оставляем best-effort нули
    usage = {"prompt": 0, "completion": 0, "total": 0}

    history.append(HumanMessage(content=user_text))
    history.append(AIMessage(content=json.dumps(reply.model_dump(), ensure_ascii=False)))
    return reply, history, usage