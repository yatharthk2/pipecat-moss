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

from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from pydantic import BaseModel, Field

from pipecat.frames.frames import ErrorFrame, Frame, LLMContextFrame, LLMMessagesFrame, MetricsFrame
from pipecat.metrics.metrics import ProcessingMetricsData
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

__all__ = ["RetrievedDocument", "MossRetrievalService"]


class RetrievedDocument(BaseModel):
    """Container for normalized retrieval results.

    Parameters:
        id: Unique identifier for the document.
        text: The content text of the document.
        metadata: Additional metadata associated with the document.
    """

    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MossRetrievalService(FrameProcessor):
    """Retrieval service backed by InferEdge Moss vector indexes.

    Intercepts LLM context frames to augment them with relevant documents
    retrieved from a Moss vector index based on the user's latest query.
    """

    def __init__(
        self,
        *,
        index_name: str,
        project_id: Optional[str] = None,
        project_key: Optional[str] = None,
        top_k: int = 5,
        auto_load_index: bool = True,
        system_prompt: str = "Here is additional context retrieved from memory:\n\n",
        add_as_system_message: bool = True,
        deduplicate_queries: bool = True,
        max_documents: int = 5,
        max_document_chars: Optional[int] = 2000,
        client: Optional[MossClient] = None,
        name: Optional[str] = None,
    ):
        """Initialize the Moss retrieval service.

        Args:
            index_name: Name of the Moss index to query.
            project_id: Moss project ID (optional if provided via env var).
            project_key: Moss project key (optional if provided via env var).
            top_k: Number of results to retrieve from the index. Defaults to 5.
            auto_load_index: Whether to automatically load the index. Defaults to True.
            system_prompt: Prefix text for the injected context.
            add_as_system_message: If True, adds context as a system message.
                If False, appends to the user message. Defaults to True.
            deduplicate_queries: Whether to skip retrieval if the query matches
                the previous one. Defaults to True.
            max_documents: Maximum number of documents to include in context.
                Defaults to 5.
            max_document_chars: Maximum characters per document. None for no limit.
                Defaults to 2000.
            client: Existing MossClient instance. If None, creates a new one.
            name: Optional name for the processor.

        Raises:
            ValueError: If index_name is not provided.
        """
        super().__init__(name=name)
        if not index_name:
            raise ValueError("index_name must be provided")

        self._index_name = index_name
        self._top_k = max(1, top_k)
        self._auto_load_index = auto_load_index
        self._system_prompt = system_prompt
        self._add_as_system_message = add_as_system_message
        self._deduplicate_queries = deduplicate_queries
        self._max_documents = max(1, max_documents)
        self._max_document_chars = max_document_chars

        self._client = client or MossClient(
            project_id=project_id,
            project_key=project_key,
        )
        self._last_query: Optional[str] = None

    def can_generate_metrics(self) -> bool:
        """Check if the processor can generate metrics.

        Returns:
            True, as this processor generates retrieval latency metrics.
        """
        return True

    async def retrieve_documents(
        self, query: str, *, limit: int
    ) -> Sequence[RetrievedDocument]:
        """Retrieve and normalize documents for a given query.

        Args:
            query: The search query string.
            limit: Maximum number of documents to return.

        Returns:
            A sequence of RetrievedDocument objects.
        """
        top_k = min(self._top_k, limit)
        try:
            result = await self._client.query(
                self._index_name,
                query,
                top_k=top_k,
                auto_load=self._auto_load_index,
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
                                ProcessingMetricsData(
                                    processor=self.name,
                                    value=time_taken / 1000.0,
                                )
                            ]
                        )
                    )

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
                    metadata = {}

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
                        metadata=metadata,
                    )
                )

            return documents[:limit]

        except Exception as e:
            logger.error(f"{self}: Moss retrieval failed: {e}")
            return []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames to extract queries and augment LLM context with retrieved documents.

        Args:
            frame: The frame to process.
            direction: The direction of the frame flow.
        """
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
                    latest_user_message, limit=self._max_documents
                )
                self._set_last_query(latest_user_message)

                if documents:
                    content = self._format_documents(documents)
                    role = "system" if self._add_as_system_message else "user"
                    context.add_message({"role": role, "content": content})

            await self._emit_context(frame, messages, context)

        except Exception as exc:
            logger.exception(f"{self}: error while running retrieval: {exc}")
            await self.push_error(ErrorFrame(error=f"{self} retrieval error: {exc}"))

    def _set_last_query(self, query: str):
        """Update the last processed query to support deduplication.

        Args:
            query: The query string to store.
        """
        if self._deduplicate_queries:
            self._last_query = query

    def _should_skip_query(self, query: str) -> bool:
        """Determine if the query should be skipped based on deduplication rules.

        Args:
            query: The candidate query string.

        Returns:
            True if the query should be skipped, False otherwise.
        """
        return bool(self._deduplicate_queries and self._last_query == query)

    async def _emit_context(
        self,
        original_frame: Frame,
        original_messages: Optional[List[Dict[str, Any]]],
        context: LLMContext | OpenAILLMContext,
    ):
        """Emit the updated context or messages frame.

        Args:
            original_frame: The original frame received.
            original_messages: The original messages list if present.
            context: The updated LLM context object.
        """
        if original_messages is not None:
            await self.push_frame(LLMMessagesFrame(context.get_messages()))
        elif isinstance(original_frame, (LLMContextFrame, OpenAILLMContextFrame)):
            await self.push_frame(type(original_frame)(context=context))  # type: ignore[arg-type]
        else:
            await self.push_frame(original_frame)

    @staticmethod
    def _get_latest_user_text(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
        """Extract the text content from the latest user message.

        Args:
            messages: A sequence of message dictionaries.

        Returns:
            The text content of the last user message, or None if not found.
        """
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
        """Format retrieved documents into a single context string.

        Args:
            documents: Sequence of retrieved documents.

        Returns:
            A formatted string containing the system prompt and document contents.
        """
        lines = [self._system_prompt.rstrip(), ""]
        for idx, document in enumerate(documents, start=1):
            suffix = ""
            if document.metadata:
                extras = []
                if source := (
                    document.metadata.get("source") or document.metadata.get("origin")
                ):
                    extras.append(f"source={source}")
                if (score := document.metadata.get("score")) is not None:
                    extras.append(f"score={score}")
                if extras:
                    suffix = f" ({', '.join(extras)})"

            text = document.text
            limit = self._max_document_chars
            if limit and len(text) > limit:
                text = f"{text[:limit].rstrip()}…"

            lines.append(f"{idx}. {text}{suffix}")
        return "\n".join(line for line in lines if line).strip()
