import json

import pytest

from src.rag import (
    FallbackTextGenerator,
    GeminiTextGenerator,
    KnowledgeFact,
    KnowledgeRetriever,
    RecommendationAssistant,
    build_recommendation_prompt,
    load_knowledge_facts,
    parse_generated_explanation,
    validate_user_profile,
)


def make_fact(song_id=1, fact="Sunrise City is bright and upbeat."):
    return KnowledgeFact(
        fact_id="fact-1",
        song_id=song_id,
        title="Sunrise City",
        artist="Neon Echo",
        topic="listening_context",
        fact=fact,
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


def test_load_knowledge_facts_reads_csv_rows(tmp_path):
    csv_path = tmp_path / "facts.csv"
    csv_path.write_text(
        "fact_id,song_id,title,artist,topic,fact,source\n"
        "1,1,Sunrise City,Neon Echo,context,A bright pop track.,local notes\n",
        encoding="utf-8",
    )

    facts = load_knowledge_facts(str(csv_path))

    assert facts == [
        KnowledgeFact(
            fact_id="1",
            song_id=1,
            title="Sunrise City",
            artist="Neon Echo",
            topic="context",
            fact="A bright pop track.",
            source="local notes",
        )
    ]


def test_knowledge_retriever_prioritizes_matching_song_facts():
    facts = [
        make_fact(song_id=1, fact="Sunrise City is bright and upbeat."),
        make_fact(song_id=2, fact="A chill lofi track for quiet focus."),
    ]
    retriever = KnowledgeRetriever(facts)

    context = retriever.retrieve(make_song(), make_profile(), limit=1)

    assert context.song_id == 1
    assert context.facts[0].song_id == 1
    assert context.retrieval_score > 0


def test_build_recommendation_prompt_includes_grounding_and_json_contract():
    context = KnowledgeRetriever([make_fact()]).retrieve(make_song(), make_profile())

    prompt = build_recommendation_prompt(
        make_profile(),
        make_song(),
        score=11.0,
        score_explanation="mood match (+3.0)",
        context=context,
    )

    assert "Return valid JSON" in prompt
    assert "Do not claim to search the live web" in prompt
    assert "Sunrise City is bright and upbeat" in prompt
    assert "mood match (+3.0)" in prompt


def test_gemini_text_generator_calls_client_boundary():
    class Response:
        text = json.dumps(
            {
                "answer": "Grounded response",
                "confidence": 0.8,
                "citations": ["local fictional catalog notes"],
                "guardrail_notes": [],
            }
        )

    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, **kwargs):
            self.calls.append(kwargs)
            return Response()

    class FakeClient:
        def __init__(self):
            self.models = FakeModels()

    client = FakeClient()
    generator = GeminiTextGenerator(model="gemini-test", client=client)

    raw = generator.generate("hello")

    assert json.loads(raw)["answer"] == "Grounded response"
    assert client.models.calls[0]["model"] == "gemini-test"
    assert client.models.calls[0]["contents"] == "hello"
    assert client.models.calls[0]["config"]["response_mime_type"] == "application/json"


def test_fallback_generator_returns_parseable_grounded_json():
    prompt = (
        "Recommended song: Sunrise City by Neon Echo\n"
        "Scoring explanation: mood match (+3.0)\n"
        "- Sunrise City is bright and upbeat. Source: local notes.\n"
    )

    generated = parse_generated_explanation(FallbackTextGenerator().generate(prompt))

    assert "Sunrise City by Neon Echo fits" in generated.answer
    assert generated.confidence > 0
    assert generated.citations


def test_validate_user_profile_clips_out_of_range_energy():
    validated = validate_user_profile(
        {
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 1.5,
            "likes_acoustic": False,
        }
    )

    assert validated.profile["target_energy"] == 1.0
    assert "clipped" in validated.guardrail_notes[0]


def test_validate_user_profile_rejects_missing_fields():
    with pytest.raises(ValueError, match="Missing user preference fields"):
        validate_user_profile({"favorite_genre": "pop"})


def test_parse_generated_explanation_guards_live_web_claims():
    raw = json.dumps(
        {
            "answer": "I searched the web and checked the live web for this track.",
            "confidence": 1.5,
            "citations": "not-a-list",
            "guardrail_notes": [],
        }
    )

    generated = parse_generated_explanation(raw)

    assert "searched the web" not in generated.answer.lower()
    assert generated.confidence == 1.0
    assert generated.citations == []
    assert any("live-web claim" in note for note in generated.guardrail_notes)


def test_recommendation_assistant_uses_fallback_for_malformed_json():
    class BadGenerator:
        name = "bad"

        def generate(self, prompt):
            return "not json"

    assistant = RecommendationAssistant(
        songs=[make_song()],
        retriever=KnowledgeRetriever([make_fact()]),
        generator=BadGenerator(),
    )

    results = assistant.recommend(make_profile(), k=1)

    assert len(results) == 1
    assert results[0].generator_name == "local-fallback"
    assert results[0].generated.answer
    assert any("malformed JSON" in note for note in results[0].generated.guardrail_notes)


def test_recommendation_assistant_generates_end_to_end_result():
    class StaticGenerator:
        name = "static"

        def generate(self, prompt):
            return json.dumps(
                {
                    "answer": "Sunrise City fits the upbeat pop profile.",
                    "confidence": 0.84,
                    "citations": ["local fictional catalog notes"],
                    "guardrail_notes": [],
                }
            )

    assistant = RecommendationAssistant(
        songs=[make_song()],
        retriever=KnowledgeRetriever([make_fact()]),
        generator=StaticGenerator(),
    )

    results = assistant.recommend(make_profile(), k=1)

    assert results[0].song["title"] == "Sunrise City"
    assert results[0].generated.confidence == 0.84
    assert results[0].context.facts
