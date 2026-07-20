"""
Core domain types for the search agent harness.

Domain models live here. API conversion is in ``adapters/openai.py``;
JSON persistence is in ``serde.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    WEB = "web"
    ACADEMIC = "academic"
    NEWS = "news"
    LOCAL = "local"


@dataclass(frozen=True, slots=True)
class Citation:
    """A single source in the session evidence pool."""

    url: str
    title: str
    snippet: str
    source_type: SourceType = SourceType.WEB
    accessed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    relevance_score: float = 0.0
    cited: bool = False
    id: str = field(default_factory=lambda: str(uuid4()))
    source_tool: str | None = None
    fetch_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "source_type": self.source_type.value,
            "accessed_at": self.accessed_at.isoformat(),
            "relevance_score": self.relevance_score,
            "cited": self.cited,
            "source_tool": self.source_tool,
            "fetch_id": self.fetch_id,
        }


# ---------------------------------------------------------------------------
# Content blocks (discriminated union)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TextBlock:
    type: Literal["text"] = "text"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ReasoningBlock:
    type: Literal["reasoning"] = "reasoning"
    text: str = ""


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    type: Literal["tool_use"] = "tool_use"
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    type: Literal["tool_result"] = "tool_result"
    content: str = ""
    is_error: bool = False


ContentBlock = TextBlock | ReasoningBlock | ToolUseBlock | ToolResultBlock


def block_text(block: ContentBlock) -> str:
    match block:
        case TextBlock(text=text) | ReasoningBlock(text=text):
            return text
        case ToolResultBlock(content=content):
            return content
        case ToolUseBlock():
            return ""


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class MessageMeta:
    tool_call_id: str | None = None
    tool_name: str | None = None
    tag: str | None = None
    compacted: bool = False
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    role: Literal["user", "assistant", "system", "tool"]
    content: str | tuple[ContentBlock, ...]
    metadata: MessageMeta = field(default_factory=MessageMeta)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def text_content(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return " ".join(
            block_text(block)
            for block in self.content
            if isinstance(block, (TextBlock, ToolResultBlock))
        )

    @property
    def blocks(self) -> tuple[ContentBlock, ...]:
        if isinstance(self.content, str):
            return (TextBlock(text=self.content),)
        return self.content


def message_tag(message: Message) -> str | None:
    """Return logical message tag (`plan_nudge`, etc.)."""
    if message.metadata.tag:
        return message.metadata.tag
    return message.metadata.extra.get("_tag")


# ---------------------------------------------------------------------------
# Tool results
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolResultMeta:
    pending_question: dict | None = None
    tool_name: str | None = None
    latency_ms: int | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class ToolResult:
    data: str
    citations: list[Citation] = field(default_factory=list)
    truncated: bool = False
    cached_path: str | None = None
    is_error: bool = False
    metadata: ToolResultMeta = field(default_factory=ToolResultMeta)

    def __post_init__(self) -> None:
        if isinstance(self.metadata, dict):
            raw = self.metadata
            self.metadata = ToolResultMeta(
                pending_question=raw.get("pending_question"),
                tool_name=raw.get("tool_name"),
                latency_ms=raw.get("latency_ms"),
                extra={
                    key: value
                    for key, value in raw.items()
                    if key not in {"pending_question", "tool_name", "latency_ms"}
                },
            )


def tool_pending_question(result: ToolResult) -> dict | None:
    return result.metadata.pending_question


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens

    def merged(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class StepMetrics:
    latency_ms: int = 0
    model_id: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)

    def to_dict(self) -> dict:
        return {
            "latency_ms": self.latency_ms,
            "model_id": self.model_id,
            "usage": self.usage.to_dict(),
        }


# ---------------------------------------------------------------------------
# Turn / step abstraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HookEvaluation:
    hook_name: str
    passed: bool
    feedback: str = ""


@dataclass(frozen=True, slots=True)
class AgentTurn:
    """One LLM iteration: assistant output, tool results, and hook outcomes."""

    turn_id: int
    run_id: str
    assistant: Message | None = None
    tool_results: tuple[Message, ...] = ()
    hook_evaluations: tuple[HookEvaluation, ...] = ()
    metrics: StepMetrics | None = None

    @property
    def hooks_passed(self) -> bool | None:
        if not self.hook_evaluations:
            return None
        return all(item.passed for item in self.hook_evaluations)

    def to_dict(self) -> dict:
        return {
            "turn_id": self.turn_id,
            "run_id": self.run_id,
            "assistant": message_to_dict(self.assistant) if self.assistant else None,
            "tool_results": [message_to_dict(msg) for msg in self.tool_results],
            "hook_evaluations": [
                {"hook_name": h.hook_name, "passed": h.passed, "feedback": h.feedback}
                for h in self.hook_evaluations
            ],
            "hooks_passed": self.hooks_passed,
            "metrics": self.metrics.to_dict() if self.metrics else None,
        }


# ---------------------------------------------------------------------------
# Stream events (typed payloads)
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    TEXT_DELTA = "text_delta"
    REASONING_DELTA = "reasoning_delta"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    CITATION = "citation"
    STATUS = "status"
    ERROR = "error"
    DONE = "done"
    PLAN_UPDATE = "plan_update"
    USER_QUESTION = "user_question"
    USAGE = "usage"
    TURN_COMPLETE = "turn_complete"


@dataclass(frozen=True, slots=True)
class TextDeltaPayload:
    text: str


@dataclass(frozen=True, slots=True)
class ReasoningDeltaPayload:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUsePayload:
    tool_use_id: str
    tool_name: str
    tool_input: dict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultPayload:
    tool_use_id: str
    tool_name: str
    content: str = ""
    result: str = ""
    result_chars: int = 0
    preview: bool = True
    is_error: bool = False
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class CitationPayload:
    citation: Citation


@dataclass(frozen=True, slots=True)
class StatusPayload:
    message: str


@dataclass(frozen=True, slots=True)
class ErrorPayload:
    message: str


@dataclass(frozen=True, slots=True)
class DonePayload:
    citations: tuple[dict, ...] = ()
    turn_count: int = 0
    compaction_count: int = 0
    session_summary: dict = field(default_factory=dict)
    final_messages: tuple[dict, ...] = ()
    answer: str = ""
    citation_count: int = 0


@dataclass(frozen=True, slots=True)
class PlanUpdatePayload:
    plan: dict


@dataclass(frozen=True, slots=True)
class UserQuestionPayload:
    tool_use_id: str = ""
    question: str = ""
    options: tuple[dict, ...] = ()


@dataclass(frozen=True, slots=True)
class UsagePayload:
    usage: TokenUsage
    model_id: str | None = None
    turn_id: int | None = None


@dataclass(frozen=True, slots=True)
class TurnCompletePayload:
    turn: AgentTurn


StreamEventPayload = (
    TextDeltaPayload
    | ReasoningDeltaPayload
    | ToolUsePayload
    | ToolResultPayload
    | CitationPayload
    | StatusPayload
    | ErrorPayload
    | DonePayload
    | PlanUpdatePayload
    | UserQuestionPayload
    | UsagePayload
    | TurnCompletePayload
)


@dataclass(frozen=True, slots=True)
class StreamEvent:
    type: EventType
    payload: StreamEventPayload
    run_id: str | None = None
    turn_id: int | None = None

    @property
    def data(self) -> dict:
        """Backward-compatible dict view of the typed payload."""
        return payload_to_dict(self.payload)

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "run_id": self.run_id,
            "turn_id": self.turn_id,
            "data": self.data,
        }


def payload_to_dict(payload: StreamEventPayload) -> dict:
    match payload:
        case TextDeltaPayload(text=text):
            return {"text": text}
        case ReasoningDeltaPayload(text=text):
            return {"text": text}
        case ToolUsePayload(tool_use_id=tool_use_id, tool_name=tool_name, tool_input=tool_input):
            return {
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            }
        case ToolResultPayload() as result:
            data = {
                "tool_use_id": result.tool_use_id,
                "tool_name": result.tool_name,
                "is_error": result.is_error,
                "truncated": result.truncated,
                "preview": result.preview,
            }
            if result.result:
                data["result"] = result.result
            if result.content:
                data["content"] = result.content
            if result.result_chars:
                data["result_chars"] = result.result_chars
            return data
        case CitationPayload(citation=citation):
            return citation.to_dict()
        case StatusPayload(message=message):
            return {"message": message}
        case ErrorPayload(message=message):
            return {"message": message}
        case DonePayload() as done:
            return {
                "citations": list(done.citations),
                "turn_count": done.turn_count,
                "compaction_count": done.compaction_count,
                "session_summary": done.session_summary,
                "final_messages": list(done.final_messages),
                "answer": done.answer,
                "citation_count": done.citation_count,
            }
        case PlanUpdatePayload(plan=plan):
            return {"plan": plan}
        case UserQuestionPayload() as question:
            return {
                "tool_use_id": question.tool_use_id,
                "question": question.question,
                "options": list(question.options),
            }
        case UsagePayload(usage=usage, model_id=model_id, turn_id=turn_id):
            data = usage.to_dict()
            if model_id is not None:
                data["model_id"] = model_id
            if turn_id is not None:
                data["turn_id"] = turn_id
            return data
        case TurnCompletePayload(turn=turn):
            return turn.to_dict()


def payload_from_dict(event_type: EventType, data: dict) -> StreamEventPayload:
    match event_type:
        case EventType.TEXT_DELTA:
            return TextDeltaPayload(text=data.get("text", ""))
        case EventType.REASONING_DELTA:
            return ReasoningDeltaPayload(text=data.get("text", ""))
        case EventType.TOOL_USE:
            return ToolUsePayload(
                tool_use_id=data.get("tool_use_id", ""),
                tool_name=data.get("tool_name", ""),
                tool_input=dict(data.get("tool_input", {})),
            )
        case EventType.TOOL_RESULT:
            return ToolResultPayload(
                tool_use_id=data.get("tool_use_id", ""),
                tool_name=data.get("tool_name", ""),
                content=data.get("content", ""),
                result=data.get("result", ""),
                result_chars=int(data.get("result_chars", 0)),
                preview=bool(data.get("preview", True)),
                is_error=bool(data.get("is_error", False)),
                truncated=bool(data.get("truncated", False)),
            )
        case EventType.CITATION:
            return CitationPayload(
                citation=Citation(
                    id=data.get("id") or str(uuid4()),
                    url=data.get("url", ""),
                    title=data.get("title", ""),
                    snippet=data.get("snippet", ""),
                    source_type=SourceType(data.get("source_type", SourceType.WEB.value)),
                    relevance_score=float(data.get("relevance_score", 0.0)),
                    cited=bool(data.get("cited", False)),
                    source_tool=data.get("source_tool"),
                    fetch_id=data.get("fetch_id"),
                )
            )
        case EventType.STATUS:
            return StatusPayload(message=data.get("message", ""))
        case EventType.ERROR:
            return ErrorPayload(message=data.get("message", ""))
        case EventType.DONE:
            return DonePayload(
                citations=tuple(data.get("citations", [])),
                turn_count=int(data.get("turn_count", 0)),
                compaction_count=int(data.get("compaction_count", 0)),
                session_summary=dict(data.get("session_summary", {})),
                final_messages=tuple(data.get("final_messages", [])),
                answer=data.get("answer", ""),
                citation_count=int(data.get("citation_count", 0)),
            )
        case EventType.PLAN_UPDATE:
            return PlanUpdatePayload(plan=dict(data.get("plan", data)))
        case EventType.USER_QUESTION:
            return UserQuestionPayload(
                tool_use_id=data.get("tool_use_id", ""),
                question=data.get("question", ""),
                options=tuple(data.get("options", [])),
            )
        case EventType.USAGE:
            return UsagePayload(
                usage=TokenUsage(
                    input_tokens=int(data.get("input_tokens", 0)),
                    output_tokens=int(data.get("output_tokens", 0)),
                    reasoning_tokens=int(data.get("reasoning_tokens", 0)),
                ),
                model_id=data.get("model_id"),
                turn_id=data.get("turn_id"),
            )
        case EventType.TURN_COMPLETE:
            return TurnCompletePayload(turn=data["turn"])  # type: ignore[arg-type]
        case _:
            return StatusPayload(message=str(data))


def stream_event(
    event_type: EventType,
    data: dict | None = None,
    *,
    run_id: str | None = None,
    turn_id: int | None = None,
) -> StreamEvent:
    """Construct a typed stream event from a legacy ``data`` dict."""
    return StreamEvent(
        type=event_type,
        payload=payload_from_dict(event_type, data or {}),
        run_id=run_id,
        turn_id=turn_id,
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    message: str = ""


# ---------------------------------------------------------------------------
# Research plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ResearchTask:
    id: str
    title: str
    details: str = ""
    status: Literal["pending", "in_progress", "completed"] = "pending"
    findings: str = ""


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    tasks: tuple[ResearchTask, ...] = ()

    def get_task(self, task_id: str) -> ResearchTask | None:
        return next((task for task in self.tasks if task.id == task_id), None)

    @property
    def completed_count(self) -> int:
        return sum(1 for task in self.tasks if task.status == "completed")

    @property
    def is_complete(self) -> bool:
        return bool(self.tasks) and all(task.status == "completed" for task in self.tasks)

    def summary(self) -> str:
        lines: list[str] = []
        for task in self.tasks:
            icon = {"pending": "○", "in_progress": "◉", "completed": "●"}[task.status]
            lines.append(f"{icon} [{task.id}] {task.title} — {task.status}")
            if task.findings:
                lines.append(f"  → {task.findings[:150]}")
        lines.append(f"\nProgress: {self.completed_count}/{len(self.tasks)} completed")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "details": task.details,
                    "status": task.status,
                    "findings": task.findings,
                }
                for task in self.tasks
            ],
            "completed_count": self.completed_count,
            "total_count": len(self.tasks),
            "is_complete": self.is_complete,
        }


# ---------------------------------------------------------------------------
# Loop state (immutable updates)
# ---------------------------------------------------------------------------

@dataclass
class LoopStateRef:
    """Mutable holder for frozen ``LoopState`` (tools update via this ref)."""

    state: LoopState


@dataclass(frozen=True, slots=True)
class LoopState:
    run_id: str
    messages: tuple[Message, ...] = ()
    turns: tuple[AgentTurn, ...] = ()
    turn_count: int = 0
    citations: tuple[Citation, ...] = ()
    compaction_count: int = 0
    search_count: int = 0
    fetch_count: int = 0
    research_plan: ResearchPlan | None = None
    total_usage: TokenUsage = field(default_factory=TokenUsage)

    @classmethod
    def initial(
        cls,
        run_id: str | None = None,
        *,
        history: tuple[Message, ...] | list[Message] = (),
    ) -> LoopState:
        return cls(
            run_id=run_id or str(uuid4()),
            messages=tuple(history),
        )

    @property
    def last_assistant_message(self) -> str | None:
        for message in reversed(self.messages):
            if message.role == "assistant":
                return message.text_content
        return None

    def with_message(self, message: Message) -> LoopState:
        return replace(self, messages=self.messages + (message,))

    def with_messages(self, messages: list[Message] | tuple[Message, ...]) -> LoopState:
        return replace(self, messages=self.messages + tuple(messages))

    def replace_messages(self, messages: list[Message] | tuple[Message, ...]) -> LoopState:
        return replace(self, messages=tuple(messages))

    def with_citations(self, *new_citations: Citation) -> LoopState:
        if not new_citations:
            return self
        return replace(self, citations=self.citations + new_citations)

    def with_turn(self, turn: AgentTurn) -> LoopState:
        usage = self.total_usage
        if turn.metrics is not None:
            usage = usage.merged(turn.metrics.usage)
        return replace(
            self,
            turns=self.turns + (turn,),
            turn_count=max(self.turn_count, turn.turn_id),
            total_usage=usage,
        )

    def with_research_plan(self, plan: ResearchPlan | None) -> LoopState:
        return replace(self, research_plan=plan)

    def increment_turn(self) -> LoopState:
        return replace(self, turn_count=self.turn_count + 1)

    def increment_search_count(self, delta: int = 1) -> LoopState:
        return replace(self, search_count=self.search_count + delta)

    def increment_fetch_count(self, delta: int = 1) -> LoopState:
        return replace(self, fetch_count=self.fetch_count + delta)

    def increment_compaction_count(self, delta: int = 1) -> LoopState:
        return replace(self, compaction_count=self.compaction_count + delta)

    def record_usage(self, usage: TokenUsage) -> LoopState:
        return replace(self, total_usage=self.total_usage.merged(usage))


# ---------------------------------------------------------------------------
# Session snapshot (versioned persistence)
# ---------------------------------------------------------------------------

SNAPSHOT_VERSION = "1"


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    version: Literal["1"]
    run_id: str
    state: LoopState
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_state(cls, state: LoopState) -> SessionSnapshot:
        now = datetime.now(timezone.utc)
        return cls(
            version=SNAPSHOT_VERSION,
            run_id=state.run_id,
            state=state,
            created_at=now,
            updated_at=now,
        )

    def with_state(self, state: LoopState) -> SessionSnapshot:
        return replace(self, state=state, updated_at=datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def message_to_dict(message: Message | None) -> dict | None:
    if message is None:
        return None
    return {
        "role": message.role,
        "content": (
            message.content
            if isinstance(message.content, str)
            else [content_block_to_dict(block) for block in message.content]
        ),
        "metadata": {
            "tool_call_id": message.metadata.tool_call_id,
            "tool_name": message.metadata.tool_name,
            "tag": message.metadata.tag,
            "compacted": message.metadata.compacted,
            "extra": message.metadata.extra,
        },
        "timestamp": message.timestamp.isoformat(),
    }


def content_block_to_dict(block: ContentBlock) -> dict:
    match block:
        case TextBlock(text=text):
            return {"type": "text", "text": text}
        case ReasoningBlock(text=text):
            return {"type": "reasoning", "text": text}
        case ToolUseBlock(tool_use_id=tool_use_id, tool_name=tool_name, tool_input=tool_input):
            return {
                "type": "tool_use",
                "tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "tool_input": tool_input,
            }
        case ToolResultBlock(content=content, is_error=is_error):
            return {
                "type": "tool_result",
                "content": content,
                "is_error": is_error,
            }
