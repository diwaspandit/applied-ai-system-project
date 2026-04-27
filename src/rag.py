import csv
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Protocol, Tuple

try:
    from .recommender import recommend_songs
except ImportError:
    from recommender import recommend_songs


LOGGER = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class KnowledgeFact:
    """One grounded fact that can be retrieved for a song recommendation."""

    fact_id: str
    song_id: Optional[int]
    title: str
    artist: str
    topic: str
    fact: str
    source: str


@dataclass(frozen=True)
class RetrievedContext:
    """Facts selected for one recommended song."""

    song_id: int
    title: str
    facts: List[KnowledgeFact]
    retrieval_score: float

    def to_prompt_lines(self) -> List[str]:
        return [
            f"- {fact.fact} Source: {fact.source}."
            for fact in self.facts
        ]


@dataclass(frozen=True)
class ValidatedProfile:
    """A normalized user profile plus guardrail notes."""

    profile: Dict
    guardrail_notes: List[str]


@dataclass(frozen=True)
class GeneratedExplanation:
    """Parsed and guarded model output."""

    answer: str
    confidence: float
    citations: List[str]
    guardrail_notes: List[str]


@dataclass(frozen=True)
class RecommendationResult:
    """End-to-end recommendation result with ranking and RAG explanation."""

    song: Dict
    score: float
    score_explanation: str
    context: RetrievedContext
    generated: GeneratedExplanation
    generator_name: str


class TextGenerator(Protocol):
    """Boundary for real or fake text generation providers."""

    name: str

    def generate(self, prompt: str) -> str:
        """Return raw JSON text for a grounded recommendation explanation."""


class GeminiGenerationError(RuntimeError):
    """Raised when the Gemini provider cannot return usable text."""


def _tokens(value: str) -> set[str]:
    return set(_TOKEN_RE.findall(value.lower()))


def load_knowledge_facts(csv_path: str) -> List[KnowledgeFact]:
    """Load local music knowledge facts from CSV."""
    facts: List[KnowledgeFact] = []
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            song_id = row.get("song_id", "").strip()
            facts.append(
                KnowledgeFact(
                    fact_id=row["fact_id"],
                    song_id=int(song_id) if song_id else None,
                    title=row["title"],
                    artist=row["artist"],
                    topic=row["topic"],
                    fact=row["fact"],
                    source=row["source"],
                )
            )
    return facts


def validate_user_profile(user_prefs: Dict) -> ValidatedProfile:
    """Normalize profile input and collect guardrail notes."""
    required = {
        "favorite_genre": str,
        "favorite_mood": str,
        "target_energy": (int, float),
        "likes_acoustic": bool,
    }
    missing = [key for key in required if key not in user_prefs]
    if missing:
        raise ValueError(f"Missing user preference fields: {', '.join(missing)}")

    notes: List[str] = []
    favorite_genre = str(user_prefs["favorite_genre"]).strip().lower()
    favorite_mood = str(user_prefs["favorite_mood"]).strip().lower()
    if not favorite_genre or not favorite_mood:
        raise ValueError("favorite_genre and favorite_mood must be non-empty")

    try:
        target_energy = float(user_prefs["target_energy"])
    except (TypeError, ValueError) as exc:
        raise ValueError("target_energy must be a number") from exc

    if target_energy < 0.0 or target_energy > 1.0:
        notes.append("target_energy was outside 0.0-1.0 and was clipped")
        target_energy = max(0.0, min(1.0, target_energy))

    likes_acoustic = user_prefs["likes_acoustic"]
    if not isinstance(likes_acoustic, bool):
        raise ValueError("likes_acoustic must be a boolean")

    return ValidatedProfile(
        profile={
            "favorite_genre": favorite_genre,
            "favorite_mood": favorite_mood,
            "target_energy": target_energy,
            "likes_acoustic": likes_acoustic,
        },
        guardrail_notes=notes,
    )


class KnowledgeRetriever:
    """Simple local retriever for fictional catalog facts."""

    def __init__(self, facts: Iterable[KnowledgeFact]):
        self._facts = list(facts)

    def retrieve(self, song: Dict, user_prefs: Dict, limit: int = 3) -> RetrievedContext:
        """Retrieve the strongest facts for a song and profile."""
        query_tokens = _tokens(
            " ".join(
                [
                    song["title"],
                    song["artist"],
                    song["genre"],
                    song["mood"],
                    user_prefs["favorite_genre"],
                    user_prefs["favorite_mood"],
                    "acoustic" if user_prefs["likes_acoustic"] else "less acoustic",
                ]
            )
        )

        scored: List[Tuple[float, KnowledgeFact]] = []
        for fact in self._facts:
            fact_tokens = _tokens(
                " ".join([fact.title, fact.artist, fact.topic, fact.fact])
            )
            score = float(len(query_tokens & fact_tokens))
            if fact.song_id == song["id"]:
                score += 8.0
            if fact.artist == song["artist"]:
                score += 2.0
            if fact.title == song["title"]:
                score += 2.0
            if score > 0:
                scored.append((score, fact))

        scored.sort(key=lambda item: (item[0], item[1].song_id == song["id"]), reverse=True)
        selected = [fact for _, fact in scored[:limit]]
        retrieval_score = scored[0][0] if scored else 0.0
        return RetrievedContext(
            song_id=song["id"],
            title=song["title"],
            facts=selected,
            retrieval_score=retrieval_score,
        )


def build_recommendation_prompt(
    user_prefs: Dict,
    song: Dict,
    score: float,
    score_explanation: str,
    context: RetrievedContext,
) -> str:
    """Build a constrained JSON prompt for Gemini."""
    context_lines = "\n".join(context.to_prompt_lines()) or "- No local facts retrieved."
    profile = (
        f"genre={user_prefs['favorite_genre']}, mood={user_prefs['favorite_mood']}, "
        f"target_energy={user_prefs['target_energy']:.2f}, "
        f"likes_acoustic={user_prefs['likes_acoustic']}"
    )
    return f"""You are VibeFinder, a careful music recommendation assistant.
Use only the local catalog context below. Do not claim to search the live web.
Return valid JSON with these keys: answer, confidence, citations, guardrail_notes.

User profile: {profile}
Recommended song: {song['title']} by {song['artist']}
Song metadata: genre={song['genre']}, mood={song['mood']}, energy={song['energy']}, acousticness={song['acousticness']}
Score: {score:.2f}
Scoring explanation: {score_explanation}
Retrieved local context:
{context_lines}

Write a concise answer explaining why this recommendation fits. Set confidence from 0.0 to 1.0 based on context strength. Include citation strings copied from the local context source names.
"""


class GeminiTextGenerator:
    """Gemini API implementation of the text generation boundary."""

    name = "gemini"

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        client=None,
    ):
        self.model = model
        self._client = client
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if self._client is None:
            if not self._api_key:
                raise GeminiGenerationError("GEMINI_API_KEY is not set")
            try:
                from google import genai
            except ImportError as exc:
                raise GeminiGenerationError("google-genai is not installed") from exc
            self._client = genai.Client(api_key=self._api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"},
            )
        except Exception as exc:
            raise GeminiGenerationError(f"Gemini generation failed: {exc}") from exc

        text = getattr(response, "text", "")
        if not text or not text.strip():
            raise GeminiGenerationError("Gemini returned an empty response")
        return text


class FallbackTextGenerator:
    """Deterministic generator used when Gemini is unavailable or invalid."""

    name = "local-fallback"

    def __init__(self, reason: str = "Gemini unavailable; used deterministic fallback."):
        self.reason = reason

    def generate(self, prompt: str) -> str:
        song_match = re.search(r"Recommended song: (.+)", prompt)
        score_match = re.search(r"Scoring explanation: (.+)", prompt)
        fact_lines = [line[2:] for line in prompt.splitlines() if line.startswith("- ")]
        song_text = song_match.group(1) if song_match else "this song"
        score_text = score_match.group(1) if score_match else "the scoring signals matched"
        facts_text = " ".join(fact_lines[:2]) if fact_lines else "No local facts were available."
        return json.dumps(
            {
                "answer": (
                    f"{song_text} fits because {score_text}. "
                    f"Grounding from the local catalog: {facts_text}"
                ),
                "confidence": 0.62 if fact_lines else 0.35,
                "citations": ["local fictional catalog notes"] if fact_lines else [],
                "guardrail_notes": [self.reason],
            }
        )


def build_default_generator() -> TextGenerator:
    """Prefer Gemini when configured, otherwise keep the app reproducible."""
    try:
        return GeminiTextGenerator()
    except GeminiGenerationError as exc:
        LOGGER.warning("Using fallback generator: %s", exc)
        return FallbackTextGenerator(str(exc))


def parse_generated_explanation(raw_text: str) -> GeneratedExplanation:
    """Parse and validate model JSON into a safe explanation object."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError("Generator returned malformed JSON") from exc

    answer = str(payload.get("answer", "")).strip()
    if not answer:
        raise ValueError("Generator returned an empty answer")

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    citations = payload.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    citations = [str(citation).strip() for citation in citations if str(citation).strip()]

    notes = payload.get("guardrail_notes", [])
    if not isinstance(notes, list):
        notes = [str(notes)]
    guardrail_notes = [str(note).strip() for note in notes if str(note).strip()]

    if "live web" in answer.lower() or "searched the web" in answer.lower():
        guardrail_notes.append("Removed unsupported live-web claim from output")
        answer = re.sub(r"(?i)\b(i )?searched the web\b", "used the local catalog", answer)
        answer = re.sub(r"(?i)\blive web\b", "local catalog", answer)

    return GeneratedExplanation(
        answer=answer,
        confidence=confidence,
        citations=citations,
        guardrail_notes=guardrail_notes,
    )


class RecommendationAssistant:
    """Coordinates ranking, retrieval, generation, validation, and logging."""

    def __init__(
        self,
        songs: List[Dict],
        retriever: KnowledgeRetriever,
        generator: TextGenerator,
        fallback_generator: Optional[TextGenerator] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._songs = songs
        self._retriever = retriever
        self._generator = generator
        self._fallback_generator = fallback_generator or FallbackTextGenerator()
        self._logger = logger or LOGGER

    def recommend(self, user_prefs: Dict, k: int = 3) -> List[RecommendationResult]:
        validated = validate_user_profile(user_prefs)
        ranked = recommend_songs(validated.profile, self._songs, k=k)
        results: List[RecommendationResult] = []

        for song, score, score_explanation in ranked:
            context = self._retriever.retrieve(song, validated.profile)
            prompt = build_recommendation_prompt(
                validated.profile,
                song,
                score,
                score_explanation,
                context,
            )
            generated, generator_name = self._generate_with_guardrails(prompt, context)
            notes = [*validated.guardrail_notes, *generated.guardrail_notes]
            if not context.facts:
                notes.append("No retrieved context was available for this song")
            generated = GeneratedExplanation(
                answer=generated.answer,
                confidence=generated.confidence,
                citations=generated.citations,
                guardrail_notes=notes,
            )
            results.append(
                RecommendationResult(
                    song=song,
                    score=score,
                    score_explanation=score_explanation,
                    context=context,
                    generated=generated,
                    generator_name=generator_name,
                )
            )

        return results

    def _generate_with_guardrails(
        self,
        prompt: str,
        context: RetrievedContext,
    ) -> Tuple[GeneratedExplanation, str]:
        try:
            raw_text = self._generator.generate(prompt)
            generated = parse_generated_explanation(raw_text)
            return self._ensure_citations(generated, context), self._generator.name
        except (GeminiGenerationError, ValueError) as exc:
            self._logger.warning("Primary generator failed; using fallback: %s", exc)
            fallback_raw = self._fallback_generator.generate(prompt)
            generated = parse_generated_explanation(fallback_raw)
            notes = [*generated.guardrail_notes, str(exc)]
            return (
                GeneratedExplanation(
                    answer=generated.answer,
                    confidence=generated.confidence,
                    citations=generated.citations,
                    guardrail_notes=notes,
                ),
                self._fallback_generator.name,
            )

    @staticmethod
    def _ensure_citations(
        generated: GeneratedExplanation,
        context: RetrievedContext,
    ) -> GeneratedExplanation:
        if generated.citations or not context.facts:
            return generated
        return GeneratedExplanation(
            answer=generated.answer,
            confidence=generated.confidence,
            citations=sorted({fact.source for fact in context.facts}),
            guardrail_notes=[
                *generated.guardrail_notes,
                "Citations were inferred from retrieved local context",
            ],
        )
