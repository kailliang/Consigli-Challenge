from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import Any, Sequence
import logging

try:  # pragma: no cover - optional sync client
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .parsers.documents import TableSummary

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def build_schema_summary(table: TableSummary, *, max_length: int = 100) -> str:
    headers = _infer_headers(table)
    label_header = headers[0] if headers else None
    value_headers = headers[1:] if len(headers) > 1 else []

    parts: list[str] = []

    if label_header:
        parts.append(f"{label_header} breakdown")

    if value_headers:
        year_headers = [header for header in value_headers if _YEAR_RE.search(header)]
        if year_headers:
            parts.append(f"years {', '.join(year_headers[:3])}")
        else:
            parts.append(", ".join(value_headers[:2]))

    categories = []
    if table.rows and label_header:
        seen: set[str] = set()
        for row in table.rows:
            label = str(row.get(label_header, "")).strip()
            if not label:
                continue
            if label.lower() in seen:
                continue
            seen.add(label.lower())
            categories.append(label)
            if len(categories) >= 3:
                break

    if categories:
        parts.append(f"examples: {', '.join(categories)}")

    if not parts:
        parts.append("Table overview")

    summary = "; ".join(parts)
    return truncate_text(summary, max_length=max_length)


def truncate_text(value: str, *, max_length: int) -> str:
    cleaned = " ".join(value.strip().split())
    if len(cleaned) <= max_length:
        return cleaned

    tokens = cleaned.split()
    result: list[str] = []
    total = 0

    for token in tokens:
        addition = len(token) if not result else len(token) + 1
        if total + addition > max_length:
            break
        result.append(token)
        total += addition

    return " ".join(result).strip()


def tokenize(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value)}


def _infer_headers(table: TableSummary) -> list[str]:
    if table.rows:
        first_row = table.rows[0]
        return list(first_row.keys())
    return []


@dataclass(slots=True)
class TableContextGenerator:
    api_key: str | None
    api_base: str | None
    model: str
    max_tokens: int = 80
    temperature: float = 0.2
    max_chars: int = 100

    def generate(
        self,
        *,
        table: TableSummary,
        company: str | None,
        year: int | None,
        document_name: str | None,
        heuristics_sentence: str | None,
        surrounding_text: str | None,
    ) -> str:
        schema_hint = build_schema_summary(table, max_length=self.max_chars)
        fallback_candidates = [heuristics_sentence, surrounding_text, schema_hint]
        fallback = next((candidate for candidate in fallback_candidates if candidate), schema_hint)

        headers = _infer_headers(table)
        rows_preview = table.rows[:5] if table.rows else []

        logger.debug(
            "table_context.generate.start table_id=%s heuristics_sentence=%s schema_hint=%s",
            table.table_id,
            heuristics_sentence,
            schema_hint,
        )

        summary = self._summarize_with_llm(
            headers=headers,
            rows_preview=rows_preview,
            company=company,
            year=year,
            document_name=document_name,
            schema_hint=schema_hint,
            nearby_text=heuristics_sentence,
            surrounding_text=surrounding_text,
            table_id=table.table_id,
        )

        if summary:
            summary = truncate_text(summary, max_length=self.max_chars)
            if self._is_valid_summary(summary, headers):
                logger.debug("table_context.generate.success table_id=%s summary=%s", table.table_id, summary)
                return summary
            logger.debug(
                "table_context.generate.invalid table_id=%s summary=%s headers=%s",
                table.table_id,
                summary,
                headers,
            )

        logger.debug("table_context.generate.fallback table_id=%s fallback=%s", table.table_id, fallback)
        return truncate_text(fallback, max_length=self.max_chars)

    def _summarize_with_llm(
        self,
        *,
        headers: Sequence[str],
        rows_preview: Sequence[dict[str, Any]],
        company: str | None,
        year: int | None,
        document_name: str | None,
        schema_hint: str,
        nearby_text: str | None,
        surrounding_text: str | None,
        table_id: str | None,
    ) -> str | None:
        if not self.api_key or not self.model:
            logger.debug("table_context.llm.disabled reason=no_api_key_or_model table_id=%s", table_id)
            return None

        prompt_lines: list[str] = []

        if company:
            prompt_lines.append(f"Company: {company}")
        if year is not None:
            prompt_lines.append(f"Fiscal year: {year}")
        if document_name:
            prompt_lines.append(f"Document: {document_name}")

        prompt_lines.append(f"Headers: {', '.join(headers) or 'unknown'}")

        if rows_preview:
            prompt_lines.append("Rows:")
            for index, row in enumerate(rows_preview, start=1):
                row_text = ", ".join(f"{key}: {row.get(key, '')}" for key in headers if key in row)
                if not row_text:
                    row_text = ", ".join(f"{key}: {value}" for key, value in row.items())
                prompt_lines.append(f"- {index}. {row_text}")

        if nearby_text:
            prompt_lines.append(f"Nearby text: {nearby_text}")

        if surrounding_text and surrounding_text != nearby_text:
            prompt_lines.append(f"Surrounding details: {surrounding_text}")

        prompt_lines.append(f"Schema hint: {schema_hint}")

        prompt_lines.append(
            (
                "Write one English sentence (max 100 characters) describing the table's subject, "
                "key metrics, dimensions, and years or units. "
                "Only use information present in the table or provided context. "
                "Do not introduce new facts or assumptions. Return only the sentence."
            )
        )

        prompt = "\n".join(prompt_lines)

        if OpenAI is None:
            return None

        try:
            response_text = self._summarize_with_responses(prompt=prompt, table_id=table_id)
            if response_text:
                logger.debug("table_context.llm.responses_success table_id=%s", table_id)
                return response_text
        except Exception as exc:  # pragma: no cover
            logger.error(
                "table_context.llm.responses_error table_id=%s model=%s error=%s",
                table_id,
                self.model,
                _summarize_exception(exc),
            )

        try:
            fallback_text = self._summarize_with_chat_completions(prompt=prompt, table_id=table_id)
            if fallback_text:
                logger.debug("table_context.llm.chat_success table_id=%s", table_id)
                return fallback_text
        except Exception as exc:  # pragma: no cover
            logger.error(
                "table_context.llm.chat_error table_id=%s model=%s error=%s",
                table_id,
                self.model,
                _summarize_exception(exc),
            )

        return None

    def _summarize_with_responses(self, *, prompt: str, table_id: str | None) -> str | None:
        if OpenAI is None:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.responses.create(
            model=self.model,
            input=prompt,
            instructions="You are a precise financial data summarizer.",
            max_output_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        text = getattr(response, "output_text", None)
        if isinstance(text, str):
            return text.strip()

        output = getattr(response, "output", None)
        if output:
            for item in output:
                message = getattr(item, "message", None)
                if not message:
                    continue
                contents = getattr(message, "content", [])
                for content in contents:
                    text_value = getattr(content, "text", None)
                    if text_value:
                        return text_value.strip()

        return None

    def _summarize_with_chat_completions(self, *, prompt: str, table_id: str | None) -> str | None:
        if OpenAI is None:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise financial data summarizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        choices = getattr(response, "choices", [])
        if not choices:
            return None
        message = choices[0].message
        content = getattr(message, "content", None)
        return content.strip() if isinstance(content, str) else None

    def _is_valid_summary(self, summary: str, headers: Sequence[str]) -> bool:
        tokens = tokenize(summary)
        header_tokens: set[str] = set()
        for header in headers:
            header_tokens.update(tokenize(header))

        has_overlap = bool(tokens & header_tokens)
        has_year = bool(_YEAR_RE.search(summary))

        return has_overlap or has_year


def _summarize_exception(exc: Exception) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        status = getattr(response, "status_code", "?")
        message: str | None = None
        try:
            data = response.json()
            if isinstance(data, dict):
                error = data.get("error")
                if isinstance(error, dict):
                    message = error.get("message") or error.get("code")
        except Exception:
            try:
                message = response.text
            except Exception:
                message = None

        snippet = (message or "").strip()
        if snippet:
            snippet = " ".join(snippet.split())[:200]
            return f"{status} {snippet}"
        return f"{status}"

    return str(exc)
