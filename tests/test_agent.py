import json

from src.agent import AgenticMusicAgent
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
                "answer": "Sunrise City fits because it is grounded in local facts.",
                "confidence": 0.82,
                "citations": ["local fictional catalog notes"],
                "guardrail_notes": [],
            }
        )


class LowConfidenceGenerator:
    name = "low-confidence"

    def generate(self, prompt):
        return json.dumps(
            {
                "answer": "Sunrise City may fit.",
                "confidence": 0.10,
                "citations": ["local fictional catalog notes"],
                "guardrail_notes": [],
            }
        )


def make_agent(generator=StaticGenerator(), minimum_confidence=0.30):
    assistant = RecommendationAssistant(
        songs=[make_song()],
        retriever=KnowledgeRetriever([make_fact()]),
        generator=generator,
    )
    return AgenticMusicAgent(
        assistant=assistant,
        minimum_confidence=minimum_confidence,
    )


def test_agent_records_observable_planning_tool_and_check_steps():
    agent = make_agent()

    run = agent.run(make_profile(), k=1)

    assert run.recommendations[0].song["title"] == "Sunrise City"
    assert run.passed_self_check is True
    assert [step.name for step in run.steps] == [
        "Plan workflow",
        "Validate profile",
        "Call RAG recommender",
        "Self-check recommendations",
        "Finalize answer",
    ]
    assert any(step.tool == "recommendation_assistant" for step in run.steps)
    assert any("PASS Sunrise City" in step.observation for step in run.steps)


def test_agent_stops_on_invalid_profile_with_failed_trace():
    agent = make_agent()

    run = agent.run({"favorite_genre": "pop"}, k=1)

    assert run.recommendations == []
    assert run.passed_self_check is False
    assert run.steps[-1].status == "failed"
    assert "Missing user preference fields" in run.steps[-1].observation


def test_agent_marks_self_check_warning_and_returns_top_candidate():
    agent = make_agent(generator=LowConfidenceGenerator(), minimum_confidence=0.50)

    run = agent.run(make_profile(), k=1)

    assert run.recommendations[0].song["title"] == "Sunrise City"
    assert run.passed_self_check is False
    assert any(step.name == "Fallback decision" for step in run.steps)
    assert any("confidence" in step.observation for step in run.steps)
