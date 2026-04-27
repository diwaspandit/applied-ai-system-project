import json

from src.evaluation import EvaluationCase, evaluate_assistant
from src.rag import KnowledgeFact, KnowledgeRetriever, RecommendationAssistant


def make_fact():
    return KnowledgeFact(
        fact_id="fact-1",
        song_id=1,
        title="Sunrise City",
        artist="Neon Echo",
        topic="listening_context",
        fact="Sunrise City is bright and upbeat.",
        source="local fictional catalog notes",
    )


def make_song():
    return {
        "id": 1,
        "title": "Sunrise City",
        "artist": "Neon Echo",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.82,
        "tempo_bpm": 118,
        "valence": 0.84,
        "danceability": 0.79,
        "acousticness": 0.18,
    }


def make_profile():
    return {
        "favorite_genre": "pop",
        "favorite_mood": "happy",
        "target_energy": 0.8,
        "likes_acoustic": False,
    }


class StaticGenerator:
    name = "static"

    def generate(self, prompt):
        return json.dumps(
            {
                "answer": "Sunrise City fits the upbeat pop profile.",
                "confidence": 0.80,
                "citations": ["local fictional catalog notes"],
                "guardrail_notes": [],
            }
        )


def test_evaluate_assistant_returns_pass_fail_summary():
    assistant = RecommendationAssistant(
        songs=[make_song()],
        retriever=KnowledgeRetriever([make_fact()]),
        generator=StaticGenerator(),
    )

    summary = evaluate_assistant(
        assistant,
        cases=[
            EvaluationCase(
                name="Pop case",
                profile=make_profile(),
                expected_top_genres=("pop",),
                minimum_confidence=0.5,
            )
        ],
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.outcomes[0].top_song == "Sunrise City"
