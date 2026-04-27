"""Command line runner for the Gemini RAG Music Recommender."""

try:
    from .rag import (
        KnowledgeRetriever,
        RecommendationResult,
        build_default_generator,
        load_knowledge_facts,
    )
    from .recommender import load_songs
except ImportError:
    from rag import (
        KnowledgeRetriever,
        RecommendationResult,
        build_default_generator,
        load_knowledge_facts,
    )
    from recommender import load_songs

from pathlib import Path
import logging


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SONGS_PATH = PROJECT_ROOT / "data" / "songs.csv"
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "music_knowledge.csv"


USER_PREFERENCE_PROFILES = {
    "Chill Lofi": {
        "favorite_genre": "lofi",
        "favorite_mood": "chill",
        "target_energy": 0.38,
        "likes_acoustic": True,
    },
    "High-Energy Pop": {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.85,
        "likes_acoustic": False,
    },
    "Deep Intense Rock": {
        "favorite_genre": "rock",
        "favorite_mood": "intense",
        "target_energy": 0.90,
        "likes_acoustic": False,
    },
}


def _format_result(result: RecommendationResult) -> str:
    citations = ", ".join(result.generated.citations) or "no citations"
    guardrails = (
        "; ".join(result.generated.guardrail_notes)
        if result.generated.guardrail_notes
        else "none"
    )
    return (
        f"{result.song['title']} by {result.song['artist']} - Score: {result.score:.2f}\n"
        f"Ranking signals: {result.score_explanation}\n"
        f"RAG explanation: {result.generated.answer}\n"
        f"Confidence: {result.generated.confidence:.2f} | Generator: {result.generator_name}\n"
        f"Citations: {citations}\n"
        f"Guardrails: {guardrails}\n"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    songs = load_songs(str(SONGS_PATH))
    facts = load_knowledge_facts(str(KNOWLEDGE_PATH))
    generator = build_default_generator()
    retriever = KnowledgeRetriever(facts)

    print(f"Loaded songs: {len(songs)}")
    print(f"Loaded knowledge facts: {len(facts)}")

    selected_profile = "Chill Lofi"
    user_prefs = USER_PREFERENCE_PROFILES[selected_profile]
    print(f"Using profile: {selected_profile}")

    try:
        from .rag import RecommendationAssistant
    except ImportError:
        from rag import RecommendationAssistant

    assistant = RecommendationAssistant(
        songs=songs,
        retriever=retriever,
        generator=generator,
    )
    recommendations = assistant.recommend(user_prefs, k=3)

    print("\nTop grounded recommendations:\n")
    for result in recommendations:
        print(_format_result(result))


if __name__ == "__main__":
    main()
