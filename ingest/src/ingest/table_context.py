from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence

try:  # pragma: no cover - optional clients
    from openai import AsyncOpenAI, OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]
    AsyncOpenAI = None  # type: ignore[assignment]

from .parsers.documents import TableSummary

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_YEAR_RE = re.compile(r"(19|20)\d{2}")


def build_schema_summary(table: TableSummary, *, max_words: int = 100) -> str:
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
    return truncate_text(summary, max_words=max_words)


def truncate_text(value: str, *, max_words: int) -> str:
    tokens = [token for token in value.strip().split() if token]
    if len(tokens) <= max_words:
        return " ".join(tokens)

    truncated = tokens[:max_words]
    return " ".join(truncated).strip()


def tokenize(value: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(value)}


def _infer_headers(table: TableSummary) -> list[str]:
    if table.rows:
        first_row = table.rows[0]
        return list(first_row.keys())
    return []


@dataclass(slots=True)
class TableContextRequest:
    table: TableSummary
    company: str | None
    year: int | None
    document_name: str | None
    heuristics_sentence: str | None
    surrounding_text: str | None


@dataclass(slots=True)
class TableContextGenerator:
    api_key: str | None
    api_base: str | None
    model: str
    max_tokens: int = 200
    max_words: int = 100
    max_concurrency: int = 4

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
        request = TableContextRequest(
            table=table,
            company=company,
            year=year,
            document_name=document_name,
            heuristics_sentence=heuristics_sentence,
            surrounding_text=surrounding_text,
        )
        results = self.generate_bulk([request])
        if results:
            return results[0]

        _, fallback, _ = self._prepare_bundle(request)
        return truncate_text(fallback, max_words=self.max_words)

    def generate_bulk(self, requests: Sequence[TableContextRequest]) -> list[str]:
        if not requests:
            return []

        if AsyncOpenAI is None or not self.api_key or not self.model:
            return [self._generate_sync(request) for request in requests]

        try:
            return asyncio.run(self._generate_bulk_async(requests))
        except RuntimeError:
            logger.debug("table_context.async.unavailable_falling_back")
            return [self._generate_sync(request) for request in requests]

    def _generate_sync(self, request: TableContextRequest) -> str:
        prompt, fallback, headers = self._prepare_bundle(request)
        summary = self._call_llm_sync(prompt=prompt, table_id=request.table.table_id)
        return self._finalize_summary(
            summary=summary,
            headers=headers,
            fallback=fallback,
            table_id=request.table.table_id,
        )

    async def _generate_bulk_async(self, requests: Sequence[TableContextRequest]) -> list[str]:
        if AsyncOpenAI is None:
            return [self._generate_sync(request) for request in requests]

        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async with AsyncOpenAI(api_key=self.api_key, base_url=self.api_base) as client:
            async def worker(request: TableContextRequest) -> str:
                prompt, fallback, headers = self._prepare_bundle(request)
                async with semaphore:
                    summary = await self._call_llm_async(
                        client=client,
                        prompt=prompt,
                        table_id=request.table.table_id,
                    )
                return self._finalize_summary(
                    summary=summary,
                    headers=headers,
                    fallback=fallback,
                    table_id=request.table.table_id,
                )

            tasks = [asyncio.create_task(worker(request)) for request in requests]
            return await asyncio.gather(*tasks)

    def _prepare_bundle(self, request: TableContextRequest) -> tuple[str, str, list[str]]:
        table = request.table
        schema_hint = build_schema_summary(table, max_words=self.max_words)
        fallback_candidates = [request.heuristics_sentence, request.surrounding_text, schema_hint]
        fallback = next((candidate for candidate in fallback_candidates if candidate), schema_hint)

        headers = _infer_headers(table)
        rows_preview = table.rows[:5] if table.rows else []
        raw_table = getattr(table, "raw_table", None)

        logger.debug(
            "table_context.generate.start table_id=%s heuristics_sentence=%s schema_hint=%s",
            table.table_id,
            request.heuristics_sentence,
            schema_hint,
        )

        prompt_lines: list[str] = []

        if request.company:
            prompt_lines.append(f"Company: {request.company}")
        if request.year is not None:
            prompt_lines.append(f"Fiscal year: {request.year}")
        if request.document_name:
            prompt_lines.append(f"Document: {request.document_name}")

        prompt_lines.append(f"Headers: {', '.join(headers) or 'unknown'}")

        if raw_table:
            prompt_lines.append("Table data:")
            prompt_lines.append("```markdown")
            prompt_lines.append(raw_table.strip())
            prompt_lines.append("```")
        elif rows_preview:
            prompt_lines.append("Rows:")
            for index, row in enumerate(rows_preview, start=1):
                row_text = ", ".join(f"{key}: {row.get(key, '')}" for key in headers if key in row)
                if not row_text:
                    row_text = ", ".join(f"{key}: {value}" for key, value in row.items())
                prompt_lines.append(f"- {index}. {row_text}")

        if request.heuristics_sentence:
            prompt_lines.append(f"Nearby text: {request.heuristics_sentence}")

        if request.surrounding_text and request.surrounding_text != request.heuristics_sentence:
            prompt_lines.append(f"Surrounding details: {request.surrounding_text}")

        prompt_lines.append(f"Schema hint: {schema_hint}")
        prompt_lines.append(
            (
                "Write one English sentence (max 100 words) describing the table's subject, "
                "Only use information present in the table or provided context. "
                "Do not introduce new facts or assumptions. Return only the sentence."
                "Focus on revenue, profit, growth trends, dimensions, and years or units. "
            )
        )

        prompt = "\n".join(prompt_lines)
        return prompt, fallback, headers

    def _call_llm_sync(self, *, prompt: str, table_id: str | None) -> str | None:
        if not self.api_key or not self.model:
            logger.debug("table_context.llm.disabled reason=no_api_key_or_model table_id=%s", table_id)
            return None

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
            print(
                f"[TableContext][Error] table_id={table_id} source=responses error={_summarize_exception(exc)}"
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
            print(
                f"[TableContext][Error] table_id={table_id} source=chat_completions error={_summarize_exception(exc)}"
            )

        return None

    async def _call_llm_async(
        self,
        *,
        client: AsyncOpenAI,
        prompt: str,
        table_id: str | None,
    ) -> str | None:
        if not self.api_key or not self.model:
            logger.debug("table_context.llm.disabled reason=no_api_key_or_model table_id=%s", table_id)
            return None

        try:
            response_text = await self._summarize_with_responses_async(
                client=client,
                prompt=prompt,
                table_id=table_id,
            )
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
            print(
                f"[TableContext][Error] table_id={table_id} source=responses error={_summarize_exception(exc)}"
            )

        try:
            fallback_text = await self._summarize_with_chat_completions_async(
                client=client,
                prompt=prompt,
                table_id=table_id,
            )
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
            print(
                f"[TableContext][Error] table_id={table_id} source=chat_completions error={_summarize_exception(exc)}"
            )

        return None

    def _finalize_summary(
        self,
        *,
        summary: str | None,
        headers: Sequence[str],
        fallback: str,
        table_id: str | None,
    ) -> str:
        if summary:
            summary = truncate_text(summary, max_words=self.max_words)
            logger.debug("table_context.generate.success table_id=%s summary=%s", table_id, summary)
            print(f"[TableContext] table_id={table_id} summary={summary}")
            return summary

        truncated_fallback = truncate_text(fallback, max_words=self.max_words)
        logger.debug("table_context.generate.fallback table_id=%s fallback=%s", table_id, truncated_fallback)
        print(f"[TableContext][Fallback] table_id={table_id} summary={truncated_fallback}")
        return truncated_fallback

    def _summarize_with_responses(self, *, prompt: str, table_id: str | None) -> str | None:
        if OpenAI is None:
            return None

        client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        response = client.responses.create(
            model=self.model,
            input=prompt,
            instructions="You are a precise financial data summarizer.",
            max_output_tokens=self.max_tokens,
        )
        print(f"[TableContext][ResponsesRaw] table_id={table_id} status={getattr(response, 'usage', None)}")

        return _extract_responses_text(response)

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
        )
        message_preview = None
        choices = getattr(response, "choices", [])
        if choices:
            first_message = getattr(choices[0], "message", None)
            message_preview = _extract_chat_message_text(first_message)
        print(f"[TableContext][Chat] table_id={table_id} preview={message_preview}")

        choices = getattr(response, "choices", [])
        if not choices:
            return None
        message = choices[0].message
        return _extract_chat_message_text(message)

    async def _summarize_with_responses_async(
        self,
        *,
        client: AsyncOpenAI,
        prompt: str,
        table_id: str | None,
    ) -> str | None:
        response = await client.responses.create(
            model=self.model,
            input=prompt,
            instructions="You are a precise financial data summarizer.",
            max_output_tokens=self.max_tokens,
        )
        print(f"[TableContext][ResponsesRaw] table_id={table_id} status={getattr(response, 'usage', None)}")
        return _extract_responses_text(response)

    async def _summarize_with_chat_completions_async(
        self,
        *,
        client: AsyncOpenAI,
        prompt: str,
        table_id: str | None,
    ) -> str | None:
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a precise financial data summarizer."},
                {"role": "user", "content": prompt},
            ],
        )
        choices = getattr(response, "choices", [])
        message_preview = None
        if choices:
            message_preview = _extract_chat_message_text(getattr(choices[0], "message", None))
        print(f"[TableContext][Chat] table_id={table_id} preview={message_preview}")

        choices = getattr(response, "choices", [])
        if not choices:
            return None
        message = choices[0].message
        return _extract_chat_message_text(message)

    def _is_valid_summary(self, summary: str, headers: Sequence[str]) -> bool:
        tokens = tokenize(summary)
        header_tokens: set[str] = set()
        for header in headers:
            header_tokens.update(tokenize(header))

        has_overlap = bool(tokens & header_tokens)
        has_year = bool(_YEAR_RE.search(summary))

        return has_overlap or has_year


def _extract_responses_text(response: Any) -> str | None:
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


def _extract_chat_message_text(message: Any) -> str | None:
    content = getattr(message, "content", None)

    if isinstance(content, str):
        stripped = content.strip()
        return stripped or None

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            text = None
            if isinstance(item, dict):
                text = item.get("text")
            else:
                text = getattr(item, "text", None)
            if text and text.strip():
                parts.append(text.strip())
        joined = " ".join(parts).strip()
        if joined:
            return joined

    return None


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
