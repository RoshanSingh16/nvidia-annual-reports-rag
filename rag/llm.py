"""Grounded answer generation over an OpenAI-compatible endpoint (NVIDIA NIM by default)."""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from .config import Settings
from .retrieve import Hit

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about NVIDIA annual reports and proxy statements using ONLY the numbered sources provided.
Rules:
1. Use only facts that appear in the sources. Never use outside knowledge, even if you know the answer.
2. Cite every factual claim with the source tag, e.g. [S1] or [S2][S4]. Never cite a tag that was not provided.
3. For numbers, always state the metric name, value, unit (e.g. millions of USD), and the reporting period / fiscal year exactly as the source states. If sources give different periods, do not mix them.
4. If the sources do not contain enough evidence, reply exactly: "I can't answer this from the retrieved documents." and briefly say what is missing. Do not guess.
5. ANSWER STRUCTURE AND TABLES:
- Choose the answer format based on the question.
- Whenever the question asks to compare multiple fiscal years, financial metrics, business segments, periods, or categories, use a Markdown table.
- For financial comparisons, put metrics in rows and fiscal years in separate columns whenever practical.
- Include all relevant values available in the retrieved sources.
- Every factual claim and every financial value in the table must include the appropriate source citation, such as [S1] or [S2].
- Never invent missing values. Use "Not available in retrieved sources" for individual missing cells when appropriate.
- After the table, briefly explain important trends or differences supported by the sources.
- For simple factual questions, use a concise paragraph.
- For multi-part questions, use clear headings and answer each part separately.
- Preserve exact financial figures, units, and reporting periods.
6. Do arithmetic only on numbers present in the sources and show the calculation."""

ABSTAIN_TEXT = "I can't answer this from the retrieved documents."
_CITE = re.compile(r"\[S(\d+)\]")


class LLMUnavailable(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


@dataclass
class Generation:
    text: str = ""
    cited: list[int] = field(default_factory=list)       # valid 1-based source numbers actually cited
    invalid_cites: list[int] = field(default_factory=list)
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: float = 0.0
    first_token_ms: Optional[float] = None
    context_sources: int = 0
    abstained: bool = False


def build_context(hits: list[Hit], max_chars: int) -> tuple[str, int]:
    """Number sources in rank order; stop before exceeding the character budget."""
    parts, used, n = [], 0, 0
    for h in hits:
        label = f"[S{n + 1}] {h.doc} | page {h.page} | {h.doc_type} {h.year or ''} | {h.kind}"
        block = f"{label}\n{h.text}\n"
        if used + len(block) > max_chars and n > 0:
            break
        parts.append(block[: max_chars - used] if n == 0 else block)
        used += len(parts[-1])
        n += 1
    return "\n".join(parts), n


def validate_citations(text: str, n_sources: int) -> tuple[list[int], list[int]]:
    found = sorted({int(m) for m in _CITE.findall(text)})
    return [i for i in found if 1 <= i <= n_sources], [i for i in found if i < 1 or i > n_sources]


class LLM:
    def __init__(self, settings: Settings):
        self.s = settings
        self._client = None

    @property
    def available(self) -> bool:
        return bool(self.s.llm_api_key)

    def _get_client(self):
        if not self.available:
            raise LLMUnavailable("No API key configured. Set NVIDIA_API_KEY in .env to enable answer generation.")
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.s.llm_base_url, api_key=self.s.llm_api_key,
                                  timeout=self.s.llm_timeout_s, max_retries=self.s.llm_max_retries)
        return self._client

    def _messages(self, question: str, hits: list[Hit], detailed: bool, history: Optional[list[dict]]):
        ctx, n = build_context(hits, self.s.max_context_chars)
        style = "Give a detailed, step-by-step explanation." if detailed else "Keep the answer short and direct."
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in (history or [])[-4:]:  # small rolling window of prior turns (questions only, no old contexts)
            if m["role"] == "user":
                msgs.append({"role": "user", "content": m["content"]})
        msgs.append({"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}\n{style}"})
        return msgs, n

    def stream(self, question: str, hits: list[Hit], detailed: bool = False,
               history: Optional[list[dict]] = None, out: Optional[Generation] = None) -> Iterator[str]:
        """Yield text deltas; fills `out` (usage, latency, citations) when exhausted."""
        gen = out if out is not None else Generation()
        client = self._get_client()
        msgs, n = self._messages(question, hits, detailed, history)
        gen.context_sources = n
        t0 = time.perf_counter()
        try:
            kwargs = dict(model=self.s.llm_model, messages=msgs, temperature=0.1, max_tokens=1200, stream=True)
            if self.s.llm_reasoning_effort:  # e.g. "low"/"none" makes Gemini/OpenAI reasoning models answer faster
                kwargs["reasoning_effort"] = self.s.llm_reasoning_effort
            try:
                resp = client.chat.completions.create(**kwargs, stream_options={"include_usage": True})
            except Exception as exc:  # some OpenAI-compatible servers reject stream_options; retry without token usage
                if "stream_options" in str(exc).lower() or type(exc).__name__ == "BadRequestError":
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise
            for chunk in resp:
                if getattr(chunk, "usage", None):
                    gen.prompt_tokens, gen.completion_tokens = chunk.usage.prompt_tokens, chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    if gen.first_token_ms is None:
                        gen.first_token_ms = (time.perf_counter() - t0) * 1e3
                    gen.text += delta
                    yield delta
        except Exception as exc:
            name = type(exc).__name__
            if "RateLimit" in name:
                raise LLMError("The LLM API rate limit was hit. Wait a few seconds and try again.") from exc
            if "Timeout" in name or "Connection" in name:
                raise LLMError("The LLM API timed out or is unreachable. Retrieval results are still shown below.") from exc
            if "Authentication" in name or "PermissionDenied" in name:
                raise LLMError("The API key was rejected. Check NVIDIA_API_KEY in .env.") from exc
            raise LLMError(f"LLM request failed ({name}): {exc}") from exc
        finally:
            gen.latency_ms = (time.perf_counter() - t0) * 1e3
        gen.cited, gen.invalid_cites = validate_citations(gen.text, n)
        gen.abstained = ABSTAIN_TEXT.lower() in gen.text.lower()
        if gen.prompt_tokens is not None and (self.s.price_in_per_m or self.s.price_out_per_m):
            gen.cost_usd = (gen.prompt_tokens * self.s.price_in_per_m + (gen.completion_tokens or 0) * self.s.price_out_per_m) / 1e6
