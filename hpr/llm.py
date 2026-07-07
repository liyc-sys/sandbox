"""LLM backends for the router, interpreter, and hindsight regeneration.

All artifact-construction models are called through one OpenAI-compatible
interface so the router / regenerator stay frozen prompt-based components.
API keys are read from environment variables and never stored.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional


class LLMBackend:
    """Minimal chat-completion interface."""

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        raise NotImplementedError


class OpenAICompatBackend(LLMBackend):
    """OpenAI-compatible endpoint (vLLM, oneapi, gateway, ...)."""

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "HPR_API_KEY",
        max_retries: int = 2,
        request_timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.base_url = base_url or os.environ.get("HPR_BASE_URL", "")
        self.api_key_env = api_key_env
        self.max_retries = max_retries
        self.request_timeout = request_timeout

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        try:
            import httpx
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("online mode requires the openai and httpx packages") from e

        api_key = os.environ.get(self.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError("missing API key env var: " + self.api_key_env)
        client = OpenAI(
            api_key=api_key,
            base_url=self.base_url or None,
            http_client=httpx.Client(trust_env=False, timeout=self.request_timeout),
        )

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    n=n,
                )
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                resp = client.chat.completions.create(**kwargs)
                texts = [
                    (choice.message.content or "").strip() for choice in resp.choices
                ]
                texts = [t for t in texts if t]
                if texts:
                    return texts
                raise RuntimeError("empty completion")
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                    continue
                raise RuntimeError(
                    "LLM call failed after retries: %s" % last_error
                ) from e
        raise RuntimeError("LLM call failed: %s" % last_error)


class MockBackend(LLMBackend):
    """Deterministic offline backend for tests and dry runs.

    Router prompts get a JSON label derived from simple keyword rules;
    regeneration prompts get a templated revised response. Deterministic in
    the prompt content so repeated runs produce identical artifacts.
    """

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: Optional[int] = None,
    ) -> List[str]:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""
        if "feedback router" in system.lower():
            return [self._route(user)] * n
        if "hindsight preference training" in system.lower():
            return [self._regenerate(user, i) for i in range(n)]
        digest = hashlib.sha1(user.encode("utf-8")).hexdigest()[:8]
        return ["mock-completion-" + digest] * n

    @staticmethod
    def _route(user_prompt: str) -> str:
        text = user_prompt.lower()
        feedback = text.split("next state / feedback text:", 1)[-1]
        if re.search(r"tests? (fail|pass)|task (succeed|fail)|issue resolved|final", feedback):
            label, conf = "delayed_trajectory_outcome", 0.9
        elif re.search(r"error|invalid|malformed|traceback|exit code", feedback):
            label, conf = "tool_api_outcome", 0.9
        elif re.search(r"\bignored\b|\byou missed\b|\bi meant\b|\bthat's wrong\b|\binstead\b|\bshould have\b", feedback):
            label, conf = "local_correction", 0.85
        elif re.search(r"\bprefer\b|\btoo long\b|\bshorter\b|\bcheaper\b|\bfirst\b", feedback):
            label, conf = "local_preference", 0.8
        else:
            label, conf = "neutral", 0.75
        hint = ""
        if label in ("local_preference", "local_correction"):
            hint = "Revise the response to satisfy the user's stated feedback."
        return json.dumps(
            {
                "feedback_type": label,
                "label": "undesirable" if label != "neutral" else "neutral",
                "critique": "mock critique",
                "hint": hint,
                "confidence": conf,
                "rationale": "mock keyword rule",
            }
        )

    @staticmethod
    def _regenerate(user_prompt: str, index: int) -> str:
        digest = hashlib.sha1(user_prompt.encode("utf-8")).hexdigest()[:8]
        return (
            "Revised response candidate %d (%s): follows the user's local "
            "feedback while keeping the valid task content." % (index, digest)
        )
