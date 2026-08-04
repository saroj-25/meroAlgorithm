
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Sequence

from ..utils.logging_utils import get_logger
from .prompts import SYSTEM_PROMPT, build_user_prompt, describe_register

log = get_logger("models.generator")


class BaseGenerator:
    name = "base"

    def generate(self, query: str, chunks: Sequence[dict],
                 language_profile: Dict[str, float]) -> str:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Offline extractive generator
# --------------------------------------------------------------------------- #
_SECTION_RE = re.compile(r"^##\s*(.+)$", re.MULTILINE)

NEPALI_SCAFFOLD = {
    "intuition": "**Intuition (sajilo bhasama):**",
    "formal": "**Formal statement (exam ma lekhne):**",
    "apply": "**Kasari apply garne:**",
    "complexity": "**Complexity:**",
    "pitfall": "**Dhyan dinu parne galti:**",
    "sources": "**Source chunks:**",
    "checkback": "**Ek prashna tapailai:**",
    "nothing": ("Yo question ko lagi knowledge base ma pugne jasto content bhetiena. "
                "Lecture notes ko sambandhit topic heri, ani prashna ali specific "
                "banaera sodhnus."),
}

ENGLISH_SCAFFOLD = {
    "intuition": "**Intuition:**",
    "formal": "**Formal statement:**",
    "apply": "**How to apply it:**",
    "complexity": "**Complexity:**",
    "pitfall": "**Common mistake:**",
    "sources": "**Source chunks:**",
    "checkback": "**One question back to you:**",
    "nothing": ("I could not find supporting material for this question in the "
                "knowledge base. Check the related lecture notes, or ask a more "
                "specific question."),
}

# Short Nepali connective sentences appended to each section when the reply is in
# a Nepali register.  Without them the reply is Nepali headings around English
# sentences, which measurably fails the register-match metric (see docs/findings.md).
NEPALI_FOLLOWUPS = {
    "intuition": "Yo idea pahila aaphno bhasama bhannus, ani matra code tira janus.",
    "formal": "Yo line exam ma jasto ko tyastai lekhda marks aauncha, tara kina yesto ho "
              "bhanne pani bujhnu parcha.",
    "apply": "Yi step haru copy ma lekhera ek choti sano input ma dry run garnus.",
    "complexity": "Complexity bhanda pahila kun case ho — best, average ki worst — tyo "
                  "spasta bhannus.",
    "pitfall": "Yo galti lab ma dherai jana le garchan, tyasaile code lekhi sakepachi "
               "yehi kura pahila check garnus.",
}

CHECKBACKS_EN = [
    "Can you state the invariant of the loop you would write for this?",
    "Which case (best, average or worst) does the bound you just read refer to, and why?",
    "What happens to this method when the input is empty or has one element?",
    "Can you name one input for which this approach would be the wrong choice?",
]

CHECKBACKS_NP = [
    "Yo ma loop ko invariant k huncha, aafai bhannus ta?",
    "Yo complexity kun case ko ho — best, average ki worst? Kina?",
    "Input khali bhayo bhane yo method le k garcha?",
    "Kun input ma yo approach galat hunthyo, ek ota example dinus ta?",
]


def _extract_section(text: str, heading: str) -> str:
    """Pull one '## Heading' section out of a knowledge-base chunk."""
    pattern = re.compile(rf"^##\s*{re.escape(heading)}\s*$(.*?)(?=^##\s|\Z)",
                         re.MULTILINE | re.DOTALL)
    match = pattern.search(text)
    return " ".join(match.group(1).split()) if match else ""


def _first_sentences(text: str, n: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", " ".join(text.split()))
    return " ".join(sentences[:n]).strip()


class TemplateGenerator(BaseGenerator):
    """Extractive, grounded, register-preserving generator (no LLM required)."""

    name = "template"

    def __init__(self, socratic: bool = True, cite: bool = True):
        self.socratic = socratic
        self.cite = cite

    def generate(self, query: str, chunks: Sequence[dict],
                 language_profile: Dict[str, float]) -> str:
        register = describe_register(language_profile)
        nepali_mode = "Nepali" in register
        scaffold = NEPALI_SCAFFOLD if nepali_mode else ENGLISH_SCAFFOLD

        if not chunks:
            return scaffold["nothing"]

        # When replying in a Nepali register, prefer the instructor-authored
        # code-mixed chunks as the source of the explanation: they are already
        # written in the learner's register, so the reply preserves it too.
        if nepali_mode:
            chunks = sorted(chunks, key=lambda c: 0 if c.get("language") == "cm" else 1)

        # Collect material from the retrieved chunks only.
        pieces = {"Intuition": [], "Precise statement": [], "Cost": [],
                  "How to apply it": [], "Frequent mistake": []}
        cited: List[str] = []
        for chunk in chunks:
            got_any = False
            for heading in pieces:
                section = _extract_section(chunk["text"], heading)
                if section:
                    pieces[heading].append((section, chunk["chunk_id"]))
                    got_any = True
            if not got_any:  # code-mixed mini-explanations have no '##' headings
                pieces["Intuition"].append(
                    (_first_sentences(chunk["text"].split("\n\n", 1)[-1], 3),
                     chunk["chunk_id"]))
            cited.append(chunk["chunk_id"])

        def render(heading: str, sentences: int = 2) -> str:
            if not pieces[heading]:
                return ""
            text, chunk_id = pieces[heading][0]
            body = _first_sentences(text, sentences)
            return f"{body} [{chunk_id}]" if self.cite else body

        topic = chunks[0]["topic"]
        concept = chunks[0].get("concept_name", "")
        lines: List[str] = []

        if nepali_mode:
            lines.append(f"Tapaile **{concept}** ({topic}) ko barema sodhnu bhayo. "
                         f"Tala ko kura hamro course material bata nai liyeko ho, "
                         f"ani hare point ko sath ma source chunk pani dieko cha, "
                         f"jasle garda tapai aafai verify garna saknu huncha:")
        else:
            lines.append(f"Your question is about **{concept}** ({topic}). "
                         f"Here is what the course material says.")
        lines.append("")

        for key, heading, n in [("intuition", "Intuition", 2),
                                ("formal", "Precise statement", 2),
                                ("apply", "How to apply it", 3),
                                ("complexity", "Cost", 2),
                                ("pitfall", "Frequent mistake", 2)]:
            body = render(heading, n)
            if body:
                lines.append(f"{scaffold[key]} {body}")
                if nepali_mode and key in NEPALI_FOLLOWUPS:
                    lines.append(NEPALI_FOLLOWUPS[key])
                lines.append("")

        if self.cite:
            lines.append(f"{scaffold['sources']} " + ", ".join(f"[{c}]" for c in cited))
            lines.append("")

        if self.socratic:
            pool = CHECKBACKS_NP if nepali_mode else CHECKBACKS_EN
            idx = abs(hash(query)) % len(pool)
            lines.append(f"{scaffold['checkback']} {pool[idx]}")

        return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# LLM backends
# --------------------------------------------------------------------------- #
class OllamaGenerator(BaseGenerator):
    """Local Llama-3.1-8B-Instruct via Ollama (``ollama serve``)."""

    name = "ollama"

    def __init__(self, model_name: str, url: str = "http://localhost:11434",
                 temperature: float = 0.3, top_p: float = 0.9, max_tokens: int = 800,
                 timeout: int = 120, fallback_on_error: bool = True):
        self.model_name, self.url = model_name, url.rstrip("/")
        self.temperature, self.top_p = temperature, top_p
        self.max_tokens, self.timeout = max_tokens, timeout
        self.fallback_on_error = fallback_on_error
        self._fallback: BaseGenerator | None = None

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def ping(self) -> bool:
        """Is the Ollama server reachable at all?"""
        try:
            urllib.request.urlopen(f"{self.url}/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def warm_up(self) -> bool:
        """Load the model into memory before the first real query.

        Ollama loads weights lazily, so the first /api/generate call pays the
        model-load cost on top of generation - which is what makes a timeout on
        query 1 of an evaluation look like a hang.  This sends a one-token
        request to get that cost out of the way.
        """
        try:
            self._post("/api/generate",
                       {"model": self.model_name, "prompt": "hi", "stream": False,
                        "options": {"num_predict": 1}})
            return True
        except Exception as exc:
            log.warning("Model warm-up failed (%s); the first query may be slow", exc)
            return False

    def available_models(self) -> List[str]:
        """Model tags the local server actually has pulled."""
        try:
            with urllib.request.urlopen(f"{self.url}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return []

    def resolve_model(self) -> str | None:
        """Return a usable model tag, or None if nothing suitable is pulled.

        A running server does NOT imply the configured model exists - asking for
        a missing model returns HTTP 404 from /api/generate.  If the exact tag is
        absent we look for another tag of the same family (``llama3.1:8b`` when
        ``llama3.1:8b-instruct-q4_K_M`` was requested) before giving up.
        """
        models = self.available_models()
        if not models:
            return None
        if self.model_name in models:
            return self.model_name
        # Same family, different quantisation/tag.
        family = self.model_name.split(":")[0]
        candidates = [m for m in models if m.split(":")[0] == family]
        if candidates:
            chosen = sorted(candidates)[0]
            log.warning("Ollama does not have '%s'; using '%s' instead "
                        "(pull the exact tag with: ollama pull %s)",
                        self.model_name, chosen, self.model_name)
            return chosen
        log.warning("Ollama is running but has none of the '%s' family. "
                    "Available: %s", family, ", ".join(models) or "(none)")
        return None

    def generate(self, query: str, chunks: Sequence[dict],
                 language_profile: Dict[str, float]) -> str:
        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": build_user_prompt(query, chunks, language_profile),
            "stream": False,
            "options": {"temperature": self.temperature, "top_p": self.top_p,
                        "num_predict": self.max_tokens},
        }
        try:
            return self._post("/api/generate", payload).get("response", "").strip()
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")[:300]
            except Exception:
                pass
            if exc.code == 404:
                message = (f"Ollama has no model '{self.model_name}'. "
                           f"Fix with: ollama pull {self.model_name}   "
                           f"(or set generation.model_name to one of: "
                           f"{', '.join(self.available_models()) or 'none pulled'})")
            else:
                message = f"Ollama returned HTTP {exc.code}: {detail}"
            return self._degrade(message, query, chunks, language_profile)
        except (TimeoutError, OSError) as exc:
            return self._degrade(
                f"Ollama did not respond within {self.timeout}s ({exc}). On CPU an 8B "
                f"model needs roughly 2-3 minutes per answer. Either raise "
                f"generation.timeout_s, lower generation.max_output_tokens, or use a "
                f"smaller model (ollama pull llama3.2:3b).",
                query, chunks, language_profile)
        except Exception as exc:
            return self._degrade(f"Ollama request failed: {exc}",
                                 query, chunks, language_profile)

    def _degrade(self, message: str, query: str, chunks: Sequence[dict],
                 language_profile: Dict[str, float]) -> str:
        """Never let one bad request kill a 300-query evaluation.

        With ``fallback_on_error`` the offline extractive generator takes over
        for the rest of the run; the switch is logged loudly, once.  Set
        ``fallback_on_error=False`` if you would rather the run fail fast.
        """
        if not self.fallback_on_error:
            raise RuntimeError(message)
        if self._fallback is None:
            log.error("%s\n  -> falling back to the offline extractive generator "
                      "for the remainder of this run.", message)
            self._fallback = TemplateGenerator()
            self.name = "template (ollama unavailable)"
        return self._fallback.generate(query, chunks, language_profile)


class OpenAIGenerator(BaseGenerator):
    """GPT-4o-mini fallback generator (Table 2)."""

    name = "openai"

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.3,
                 top_p: float = 0.9, max_tokens: int = 800):
        from openai import OpenAI

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self._client = OpenAI()
        self.model_name = model_name
        self.temperature, self.top_p, self.max_tokens = temperature, top_p, max_tokens

    def generate(self, query: str, chunks: Sequence[dict],
                 language_profile: Dict[str, float]) -> str:
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",
                       "content": build_user_prompt(query, chunks, language_profile)}],
        )
        return (response.choices[0].message.content or "").strip()


def build_generator(cfg) -> BaseGenerator:
    """Pick a generator backend, degrading to the offline template generator."""
    backend = cfg.get("generation.backend", "auto")
    offline = cfg.get("project.mode", "auto") == "offline"
    socratic = cfg.get("generation.socratic_checkback", True)
    cite = cfg.get("generation.require_citations", True)

    if backend in ("auto", "ollama") and not offline:
        gen = OllamaGenerator(cfg.get("generation.model_name"),
                              url=cfg.get("generation.ollama_url"),
                              temperature=cfg.get("generation.temperature", 0.3),
                              top_p=cfg.get("generation.top_p", 0.9),
                              max_tokens=cfg.get("generation.max_output_tokens", 800),
                              timeout=cfg.get("generation.timeout_s", 600),
                              fallback_on_error=cfg.get("generation.fallback_on_error", True))
        if gen.ping():
            # A reachable server is not enough: the model must actually be pulled,
            # otherwise every /api/generate call returns HTTP 404.
            resolved = gen.resolve_model()
            if resolved:
                gen.model_name = resolved
                log.info("Using Ollama generator: %s (timeout %ss)",
                         gen.model_name, gen.timeout)
                log.info("Warming up the model (first load can take a minute)...")
                gen.warm_up()
                return gen
            hint = (f"Ollama is running but '{cfg.get('generation.model_name')}' is not "
                    f"pulled. Run:  ollama pull {cfg.get('generation.model_name')}")
            if backend == "ollama":
                raise RuntimeError(hint)
            log.warning("%s -- trying the next backend.", hint)
        else:
            if backend == "ollama":
                raise RuntimeError("Ollama is not reachable at " + gen.url +
                                   " (start it with: ollama serve)")
            log.info("Ollama not reachable; trying the next backend")

    if backend in ("auto", "openai") and not offline:
        try:
            gen = OpenAIGenerator(cfg.get("generation.openai_model", "gpt-4o-mini"),
                                  temperature=cfg.get("generation.temperature", 0.3),
                                  top_p=cfg.get("generation.top_p", 0.9),
                                  max_tokens=cfg.get("generation.max_output_tokens", 800))
            log.info("Using OpenAI generator: %s", gen.model_name)
            return gen
        except Exception as exc:
            if backend == "openai":
                raise
            log.info("OpenAI backend unavailable (%s)", exc)

    log.info("Using the offline extractive template generator")
    return TemplateGenerator(socratic=socratic, cite=cite)