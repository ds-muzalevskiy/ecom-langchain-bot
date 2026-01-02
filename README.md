# Shoply ecom-bot (LangChain)

Консольный бот поддержки магазина **Shoply** с единым бренд-голосом и автооценкой стиля.

## Структура

```
ecom-bot/
  app_lc.py
  src/
    brand_chain.py
    style_eval.py
  data/
    style_guide.yaml
    few_shots.jsonl
    eval_prompts.txt
    faq.json
    orders.json
  reports/
    style_eval.json
  logs/
    session_*.jsonl
  .env.example
  requirements.txt
```

## Установка

```bash
pip install -r requirements.txt
# впишите OPENAI_API_KEY (и при желании OPENAI_MODEL / OPENAI_EVAL_MODEL) в .env.example
```

## Запуск бренд-бота

```bash
python app_lc.py
```

Команды:
- `/help` — справка
- `/order <id>` — статус заказа (пример: `/order 98765`)
- `/reset` — очистить историю
- `/exit` — выход

### Логи
- `logs/session_*.jsonl` — по каждой реплике пишется JSON с ролью, контентом и `usage` (prompt/completion/total).

## Автооценка стиля

```bash
python src/style_eval.py
```

Результат: `reports/style_eval.json`.

### Примечание про отчёт в репозитории
В репозитории лежит пример `reports/style_eval.json`, который можно пересоздать командой выше.

## Данные и правила
- `data/style_guide.yaml` — голос бренда (персона, ограничения, taboo, fallback).
- `data/few_shots.jsonl` — 3 примера «как надо» (подмешиваются в system prompt).
- `data/faq.json` и `data/orders.json` — единственные источники фактов.

## Антигаллюцинации
Если вопрос не покрыт FAQ/заказами, бот отвечает fallback-сообщением из `style_guide.yaml` и предлагает 1–3 следующих шага.
