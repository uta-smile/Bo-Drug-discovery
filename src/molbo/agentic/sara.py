"""The SARA endpoint as a drop-in :class:`molbo.llm.Backend`.

SARA (Surrogate-Augmented Autoresearch Agent, arXiv 2608.00316, implemented at
github.com/uta-smile/AgenticBo) is configured through ``SARA_BASE_URL``,
``SARA_MODEL`` and ``SARA_API_KEY`` and speaks the OpenAI chat-completions
shape. Implementing the existing ``Backend`` protocol rather than a parallel
client buys the thing this integration most needs first:

**The model confound.** Every LLM result in this project is Qwen3.5-4B, which
misidentified the series' chemotype in every prompt format tried. SARA's paper
runs against a frontier endpoint. Compare the two as published and you cannot
separate *agentic architecture* from *model scale* -- the single most likely
confound in the whole comparison.

Because this class satisfies the same protocol as
:class:`~molbo.llm.LlamaCppBackend`, the existing three arms can be re-run
against SARA's model with no code change beyond a flag::

    from molbo.agentic.sara import SaraBackend
    proposer = LLMConstraintProposer(backend=SaraBackend.from_env(), cache=cache)

That experiment needs no new arms and no agentic machinery, and it must run
before any agentic result is interpretable. If a frontier model beats the
non-LLM controls where Qwen-4B did not, the project's headline finding was
about model size rather than coupling design -- which is a correction worth
publishing on its own.

Determinism
-----------
Unknown, and reported as such. ``is_deterministic`` is true only when the
endpoint accepted ``temperature=0`` *and* a seed, which most hosted endpoints
do not guarantee even then. The response cache is what actually makes a rerun
reproducible, as it is for :class:`~molbo.llm.AnthropicBackend`.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from ..llm import Backend, LLMUnavailable

SARA_BASE_URL_ENV = "SARA_BASE_URL"
SARA_MODEL_ENV = "SARA_MODEL"
SARA_API_KEY_ENV = "SARA_API_KEY"


@dataclass
class SaraBackend(Backend):
    """An OpenAI-compatible chat-completions endpoint, schema-constrained.

    Parameters mirror :class:`~molbo.llm.LlamaCppBackend` so the two are
    interchangeable in every arm.

    ``strict_schema`` asks the endpoint to enforce the JSON schema server-side.
    Not every deployment supports it; when the endpoint rejects the field this
    class retries once without it and parses the content itself, recording
    ``schema_enforced`` in the metadata so a run cannot silently claim a
    guarantee it did not get.
    """

    base_url: str
    model: str
    api_key: str | None = None
    temperature: float = 0.0
    seed: int | None = 0
    max_tokens: int = 1500
    timeout: int = 600
    strict_schema: bool = True

    @classmethod
    def from_env(cls, **overrides: object) -> "SaraBackend":
        """Build from ``SARA_BASE_URL`` / ``SARA_MODEL`` / ``SARA_API_KEY``.

        Raises :class:`~molbo.llm.LLMUnavailable` with the variable names rather
        than a ``KeyError``, so a sweep that is missing configuration fails with
        something a user can act on.
        """
        base = os.environ.get(SARA_BASE_URL_ENV)
        model = os.environ.get(SARA_MODEL_ENV)
        missing = [
            name
            for name, val in ((SARA_BASE_URL_ENV, base), (SARA_MODEL_ENV, model))
            if not val
        ]
        if missing:
            raise LLMUnavailable(
                f"SARA endpoint not configured: set {' and '.join(missing)}. "
                f"See github.com/uta-smile/AgenticBo for the endpoint contract."
            )
        kwargs: dict = {
            "base_url": str(base).rstrip("/"),
            "model": str(model),
            "api_key": os.environ.get(SARA_API_KEY_ENV),
        }
        kwargs.update(overrides)  # type: ignore[arg-type]
        return cls(**kwargs)  # type: ignore[arg-type]

    def key(self) -> str:
        """Cache identity. Includes sampling settings, per the project rule that
        a representation or sampling change must invalidate cached values rather
        than blend two configurations into one table."""
        return f"sara:{self.model}:t={self.temperature}:seed={self.seed}"

    @property
    def is_deterministic(self) -> bool:
        return self.temperature == 0.0 and self.seed is not None

    def health(self) -> tuple[bool, str]:
        import urllib.error
        import urllib.request

        req = urllib.request.Request(f"{self.base_url}/v1/models", headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200, r.read().decode()[:200]
        except Exception as exc:  # pragma: no cover - environment dependent
            return False, str(exc)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _body(self, system: str, prompt: str, schema: dict, *, with_schema: bool) -> dict:
        body: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.seed is not None:
            body["seed"] = self.seed
        if with_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "decision", "strict": True, "schema": schema},
            }
        return body

    def complete(self, system: str, prompt: str, schema: dict) -> tuple[dict, dict]:
        import urllib.error
        import urllib.request

        attempts = [True, False] if self.strict_schema else [False]
        last_error: Exception | None = None

        for with_schema in attempts:
            req = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(self._body(system, prompt, schema, with_schema=with_schema)).encode(),
                headers=self._headers(),
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode())
            except urllib.error.HTTPError as exc:
                # A 400 on the first attempt usually means the deployment does
                # not implement `response_format`. Anything else is real.
                last_error = exc
                if with_schema and exc.code in (400, 404, 422):
                    continue
                raise LLMUnavailable(
                    f"SARA endpoint {self.base_url} returned {exc.code}: "
                    f"{exc.read().decode()[:300]}"
                ) from exc
            except urllib.error.URLError as exc:
                raise LLMUnavailable(
                    f"could not reach the SARA endpoint at {self.base_url}: {exc}. "
                    f"Check {SARA_BASE_URL_ENV} and network access."
                ) from exc

            choice = payload["choices"][0]
            text = choice["message"].get("content")
            if not text or not text.strip():
                raise LLMUnavailable(
                    f"empty response from {self.model} "
                    f"(finish_reason={choice.get('finish_reason')})"
                )
            return json.loads(_strip_fence(text)), {
                "usage": payload.get("usage", {}),
                "finish_reason": choice.get("finish_reason"),
                "schema_enforced": with_schema,
            }

        raise LLMUnavailable(f"SARA endpoint rejected every attempt: {last_error}")


def _strip_fence(text: str) -> str:
    """Drop a ```json fence if the endpoint wrapped the object in one.

    Only needed on the no-schema retry path -- a server enforcing the schema
    returns a bare object. Cheap, and it turns a common endpoint quirk into a
    non-issue rather than a parse failure recorded as a model error.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    s = s.split("\n", 1)[1] if "\n" in s else s
    if s.rstrip().endswith("```"):
        s = s.rstrip()[:-3]
    return s.strip()
