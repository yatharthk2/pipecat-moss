#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Moss retrieval service implementation.

Provides functionality for retrieval and vector-store services that augment
LLM context with relevant documents retrieved from memory or vector databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from pydantic import BaseModel, Field, field_validator

from pipecat.frames.frames import ErrorFrame, Frame, LLMContextFrame, LLMMessagesFrame, MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from .client import MossClient

try:
    from pipecat.processors.aggregators.openai_llm_context import (
        OpenAILLMContext,
        OpenAILLMContextFrame,
    )
except ImportError:  # pragma: no cover - optional dependency is always present in prod
    OpenAILLMContext = LLMContext  # type: ignore
    OpenAILLMContextFrame = LLMContextFrame  # type: ignore

__all__ = ["RetrievedDocument", "RetrievalService", "MossRetrievalService"]


class RetrievedDocument(BaseModel):
    """Container for normalized retrieval results."""

    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalService(FrameProcessor, ABC):
    """Base class for retrieval/vector-store services that augment LLM context."""

    class Params(BaseModel):
        """Configuration parameters for retrieval services."""

        system_prompt: str = Field(
            default="Here is additional context retrieved from memory:\n\n"
        )
        add_as_system_message: bool = True
        deduplicate_queries: bool = True
        max_documents: int = Field(default=5, ge=1)
        max_document_chars: Optional[int] = Field(default=2000, ge=1)

    def __init__(
        self,
        *,
        params: Optional[Params] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self._params = params or RetrievalService.Params()
        self._last_query: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        return True

    @abstractmethod
    async def retrieve_documents(
        self, query: str, *, limit: int
    ) -> Sequence[RetrievedDocument]:
        """Return normalized documents for the supplied query."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to extract queries and augment LLM context with retrieved documents."""
        await super().process_frame(frame, direction)

        context = None
        messages = None

        if isinstance(frame, (LLMContextFrame, OpenAILLMContextFrame)):
            context = frame.context
        elif isinstance(frame, LLMMessagesFrame):
            messages = frame.messages
            context = LLMContext(messages)

        if not context:
            await self.push_frame(frame, direction)
            return

        try:
            context_messages = context.get_messages()
            latest_user_message = self._get_latest_user_text(context_messages)

            if (
                latest_user_message
                and not self._should_skip_query(latest_user_message)
            ):
                documents = await self.retrieve_documents(
                    latest_user_message, limit=self._params.max_documents
                )
                self._set_last_query(latest_user_message)
                
                if documents:
                    content = self._format_documents(documents)
                    role = "system" if self._params.add_as_system_message else "user"
                    context.add_message({"role": role, "content": content})

            await self._emit_context(frame, messages, context)

        except Exception as exc:
            logger.exception(f"{self}: error while running retrieval: {exc}")
            await self.push_error(ErrorFrame(error=f"{self} retrieval error: {exc}"))

    def _set_last_query(self, query: str):
        if self._params.deduplicate_queries:
            self._last_query = query

    def _should_skip_query(self, query: str) -> bool:
        return bool(self._params.deduplicate_queries and self._last_query == query)

    async def _emit_context(
        self,
        original_frame: Frame,
        original_messages: Optional[List[Dict[str, Any]]],
        context: LLMContext | OpenAILLMContext,
    ):
        if original_messages is not None:
            await self.push_frame(LLMMessagesFrame(context.get_messages()))
        elif isinstance(original_frame, (LLMContextFrame, OpenAILLMContextFrame)):
            await self.push_frame(type(original_frame)(context=context))  # type: ignore[arg-type]
        else:
            await self.push_frame(original_frame)

    @staticmethod
    def _get_latest_user_text(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
        for message in reversed(messages):
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip() or None
                if isinstance(content, list):
                    parts = [
                        c["text"]
                        for c in content
                        if isinstance(c, dict)
                        and c.get("type") == "text"
                        and isinstance(c.get("text"), str)
                    ]
                    if parts:
                        return "\n".join(parts).strip() or None
        return None

    def _format_documents(self, documents: Sequence[RetrievedDocument]) -> str:
        lines = [self._params.system_prompt.rstrip(), ""]
        for idx, document in enumerate(documents, start=1):
            suffix = ""
            if document.metadata:
                extras = []
                if source := (document.metadata.get("source") or document.metadata.get("origin")):
                    extras.append(f"source={source}")
                if (score := document.metadata.get("score")) is not None:
                    extras.append(f"score={score}")
                if extras:
                    suffix = f" ({', '.join(extras)})"
            
            text = document.text
            limit = self._params.max_document_chars
            if limit and len(text) > limit:
                text = f"{text[:limit].rstrip()}…"
                
            lines.append(f"{idx}. {text}{suffix}")
        return "\n".join(line for line in lines if line).strip()


class MossRetrievalService(RetrievalService):
    """Retrieval service backed by InferEdge Moss vector indexes."""

    class Config(BaseModel):
        index_name: str
        project_id: Optional[str] = None
        project_key: Optional[str] = None
        top_k: int = Field(default=5, ge=1)
        auto_load_index: bool = True

        @field_validator("index_name")
        @classmethod
        def _validate_index(cls, value: str) -> str:
            if not value:
                raise ValueError("index_name must be provided")
            return value

    def __init__(
        self,
        *,
        config: Config,
        params: Optional[RetrievalService.Params] = None,
        client: Optional[MossClient] = None,
        name: Optional[str] = None,
    ):
        super().__init__(params=params, name=name)
        self._config = config
        self._client = client or MossClient(
            project_id=config.project_id,
            project_key=config.project_key,
        )

    async def retrieve_documents(
        self, query: str, *, limit: int
    ) -> Sequence[RetrievedDocument]:
        top_k = min(self._config.top_k, limit)
        try:
            result = await self._client.query(
                self._config.index_name,
                query,
                top_k=top_k,
                auto_load=self._config.auto_load_index,
            )

            if self.metrics_enabled:
                time_taken = getattr(result, "time_taken_ms", None)
                if time_taken is None and isinstance(result, dict):
                    time_taken = result.get("time_taken_ms")

                if time_taken is not None:
                    logger.info(f"{self}: Retrieval latency: {time_taken}ms")
                    await self.push_frame(
                        MetricsFrame(
                            data=[
                                TTFBMetricsData(
                                    processor=self.name,
                                    value=time_taken / 1000.0,
                                ),
                                # Report as processing time as well for visibility in some dashboards
                                ProcessingMetricsData(
                                    processor=self.name,
                                    value=time_taken / 1000.0,
                                )
                            ]
                        )
                    )
                else:
                    logger.warning(f"{self}: 'time_taken_ms' missing or None in result.")
        except Exception as e:
            logger.error(f"{self}: Moss retrieval failed: {e}")
            return []

        docs = getattr(result, "docs", []) or []
        documents = []
        
        for item in docs:
            # Handle both dict-like and object-like items
            is_dict = isinstance(item, dict)
            get_val = lambda k: item.get(k) if is_dict else getattr(item, k, None)
            
            text = get_val("text") or get_val("content") or get_val("chunk_text")
            if not text or not isinstance(text, str):
                continue

            metadata = get_val("metadata") or {}
            if not isinstance(metadata, dict):
                 metadata = {} # ensure dict
                 
            # Lift score/source into metadata if present at top level
            if (score := get_val("score")) is not None:
                metadata["score"] = score
            if source := get_val("source"):
                metadata["source"] = source
            
            doc_id = get_val("id") or get_val("doc_id")

            documents.append(
                RetrievedDocument(
                    id=str(doc_id) if doc_id else None,
                    text=text.strip(),
                    metadata=metadata
                )
            )
            
        return documents[:limit]
