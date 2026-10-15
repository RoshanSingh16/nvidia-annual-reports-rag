"""Facade tying index, retrieval, generation and logging together."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Iterator, Optional

from .config import Settings, load_settings
from .llm import ABSTAIN_TEXT, Generation, LLM, LLMError, LLMUnavailable
from .metrics import log_query
from .retrieve import Retrieval, Retriever
from .store import IndexStore

log = logging.getLogger(__name__)


class Engine:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or load_settings()
        self.store = IndexStore(self.s)
        self.retriever = Retriever(self.store, self.s)
        self.llm = LLM(self.s)
        self.ready = self.store.load()

    def sync(self, progress=None):
        rep = self.store.sync(progress)
        self.retriever._rcache.clear()
        self.ready = bool(self.store.chunks)
        return rep

    # ---- evidence gate: abstain before spending tokens ----
    def evidence_ok(self, r: Retrieval) -> bool:
        return bool(r.hits) and r.best_dense >= self.s.min_dense_score

    def retrieve(self, question: str, **kw) -> Retrieval:
        return self.retriever.search(question, **kw)

    def answer_stream(self, question: str, r: Retrieval, detailed: bool = False,
                    history: Optional[list[dict]] = None, gen: Optional[Generation] = None) -> Iterator[str]:
        gen = gen if gen is not None else Generation()
        if not self.evidence_ok(r):
            gen.abstained = True
            gen.text = (f"{ABSTAIN_TEXT} The best match in the index was weak "
                        f"(similarity {r.best_dense:.2f} < threshold {self.s.min_dense_score:.2f}), so no answer was generated.")
            yield gen.text
            return
        yield from self.llm.stream(question, r.hits, detailed, history, gen)

    def record(self, question: str, r: Retrieval, gen: Optional[Generation], error: Optional[str] = None) -> None:
        try:
            log_query(self.s.log_dir, {
                "ts": time.time(), "question": question, "mode": r.mode, "top_k": len(r.hits), "cache_hit": r.cache_hit,
                "retrieval_ms": r.total_ms, "timings_ms": r.timings_ms,
                "llm_ms": gen.latency_ms if gen else None, "first_token_ms": gen.first_token_ms if gen else None,
                "prompt_tokens": gen.prompt_tokens if gen else None, "completion_tokens": gen.completion_tokens if gen else None,
                "cost_usd": gen.cost_usd if gen else None, "abstained": gen.abstained if gen else None,
                "invalid_cites": gen.invalid_cites if gen else None, "error": error,
                "hits": [{"doc": h.doc, "page": h.page, "dense": h.dense, "bm25": h.bm25, "score": h.score} for h in r.hits],
            })
        except OSError as exc:
            log.warning("Could not write query log: %s", exc)
