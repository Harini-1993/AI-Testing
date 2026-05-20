import asyncio
from dataclasses import dataclass
from typing import Literal

from config import get_config
from app.comparison.differ import ComparisonResult, SectionDiff


LlmProvider = Literal["anthropic", "openai"]
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_OPENAI_MODEL = "gpt-4o"
SECTION_PROMPT = """You are a document analyst. Compare these two versions of a document section
and explain the key changes in 2-3 plain sentences. Be specific about what was
added, removed, or changed. Do not repeat the full text.

Section: {section_title}

Version A:
{text_a}

Version B:
{text_b}

Changes summary:"""


@dataclass
class EnrichedComparisonResult(ComparisonResult):
    """Comparison result with LLM-generated summaries."""

    section_summaries: dict[str, str]
    overall_summary: str


class DiffSummariser:
    """Generate plain-English summaries for structured document diffs."""

    def __init__(
        self,
        provider: LlmProvider | None = None,
        model: str | None = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
    ) -> None:
        self.provider = _normalise_provider(provider or get_config(validate=False).llm_provider)
        self.model = model or _default_model(self.provider)
        self.max_retries = max_retries
        self.base_delay = base_delay

    async def summarise_async(
        self,
        comparison: ComparisonResult,
    ) -> EnrichedComparisonResult:
        """Summarise modified sections and the full comparison asynchronously."""
        modified_sections = [
            section for section in comparison.sections if section.status == "modified"
        ]
        section_tasks = [
            self._summarise_section(section) for section in modified_sections
        ]
        section_outputs = await asyncio.gather(*section_tasks) if section_tasks else []
        section_summaries = {
            _summary_key(section): summary
            for section, summary in zip(modified_sections, section_outputs, strict=True)
        }
        overall_summary = await self._summarise_overall(comparison, section_summaries)

        return EnrichedComparisonResult(
            version_a_label=comparison.version_a_label,
            version_b_label=comparison.version_b_label,
            sections=comparison.sections,
            summary_stats=comparison.summary_stats,
            section_summaries=section_summaries,
            overall_summary=overall_summary,
        )

    def summarise(self, comparison: ComparisonResult) -> EnrichedComparisonResult:
        """Synchronously summarise a comparison result.

        Use summarise_async when calling from code that already owns an event loop.
        """
        _raise_if_running_loop("Use summarise_async from async code.")
        return asyncio.run(self.summarise_async(comparison))

    async def _summarise_section(self, section: SectionDiff) -> str:
        prompt = SECTION_PROMPT.format(
            section_title=section.section_title or "Untitled section",
            text_a=section.text_a,
            text_b=section.text_b,
        )
        return await self._call_llm_with_retry(prompt)

    async def _summarise_overall(
        self,
        comparison: ComparisonResult,
        section_summaries: dict[str, str],
    ) -> str:
        section_summary_text = "\n".join(
            f"- {section_title}: {summary}"
            for section_title, summary in section_summaries.items()
        )
        prompt = (
            "You are a document analyst. Summarise the overall comparison between "
            f"{comparison.version_a_label} and {comparison.version_b_label} in 3-5 "
            "plain sentences. Focus on the most important additions, removals, and "
            "modifications.\n\n"
            f"Summary stats: {comparison.summary_stats}\n\n"
            f"Modified section summaries:\n{section_summary_text or 'No modified sections.'}\n\n"
            "Overall summary:"
        )
        return await self._call_llm_with_retry(prompt)

    async def _call_llm_with_retry(self, prompt: str) -> str:
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                return await asyncio.to_thread(self._call_llm_sync, prompt)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= self.max_retries:
                    raise

                last_error = exc
                delay = self.base_delay * (2**attempt)
                await asyncio.sleep(delay)

        raise RuntimeError("LLM request failed after retries") from last_error

    def _call_llm_sync(self, prompt: str) -> str:
        if self.provider == "anthropic":
            return _call_anthropic(prompt, self.model)

        return _call_openai(prompt, self.model)


def summarise_diff(
    diff_text: str,
    provider: LlmProvider = "anthropic",
    model: str | None = None,
) -> str:
    """Summarise raw diff text for legacy callers."""
    summariser = DiffSummariser(provider=provider, model=model)
    prompt = (
        "Summarise the following document diff in clear, plain language:\n\n"
        f"{diff_text}"
    )
    _raise_if_running_loop("Use DiffSummariser._call_llm_with_retry from async code.")
    return asyncio.run(summariser._call_llm_with_retry(prompt))


def _call_anthropic(prompt: str, model: str) -> str:
    from anthropic import Anthropic

    client = Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def _call_openai(prompt: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(model=model, input=prompt)
    return response.output_text


def _normalise_provider(provider: str) -> LlmProvider:
    provider_name = provider.lower().strip()
    if provider_name in {"anthropic", "claude"}:
        return "anthropic"

    if provider_name == "openai":
        return "openai"

    raise ValueError(f"Unsupported LLM provider: {provider}")


def _default_model(provider: LlmProvider) -> str:
    if provider == "anthropic":
        return DEFAULT_ANTHROPIC_MODEL

    return DEFAULT_OPENAI_MODEL


def _is_rate_limit_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "ratelimit" in name or "rate limit" in message or "429" in message


def _summary_key(section: SectionDiff) -> str:
    return section.section_title or "Untitled section"


def _raise_if_running_loop(message: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return

    raise RuntimeError(message)
