from pathlib import Path
import logging

import streamlit as st

from src.agent import AgenticMusicAgent, AgentRun
from src.rag import (
    KnowledgeRetriever,
    RecommendationAssistant,
    build_default_generator,
    load_knowledge_facts,
)
from src.recommender import load_songs
from src.ui_helpers import (
    build_user_profile,
    catalog_options,
    confidence_label,
    status_label,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SONGS_PATH = PROJECT_ROOT / "data" / "songs.csv"
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "music_knowledge.csv"

PRESET_PROFILES = {
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


@st.cache_data
def load_catalog_data():
    songs = load_songs(str(SONGS_PATH))
    facts = load_knowledge_facts(str(KNOWLEDGE_PATH))
    return songs, facts


def run_agent(user_profile, songs, facts, k: int, minimum_confidence: float) -> AgentRun:
    assistant = RecommendationAssistant(
        songs=songs,
        retriever=KnowledgeRetriever(facts),
        generator=build_default_generator(),
    )
    return AgenticMusicAgent(
        assistant=assistant,
        minimum_confidence=minimum_confidence,
    ).run(user_profile, k=k)


def render_recommendations(agent_run: AgentRun) -> None:
    if not agent_run.recommendations:
        st.warning(agent_run.summary)
        return

    for index, result in enumerate(agent_run.recommendations, start=1):
        with st.container(border=True):
            left, right = st.columns([3, 1])
            with left:
                st.subheader(f"{index}. {result.song['title']}")
                st.caption(
                    f"{result.song['artist']} | {result.song['genre']} | {result.song['mood']}"
                )
            with right:
                st.metric("Score", f"{result.score:.2f}")
                st.metric("Confidence", confidence_label(result.generated.confidence))

            st.progress(result.generated.confidence)
            st.write(result.generated.answer)

            details = st.columns(3)
            details[0].markdown(f"**Signals**  \n{result.score_explanation}")
            details[1].markdown(
                "**Citations**  \n"
                + (", ".join(result.generated.citations) or "No citations")
            )
            details[2].markdown(
                "**Guardrails**  \n"
                + ("; ".join(result.generated.guardrail_notes) or "None")
            )


def render_agent_trace(agent_run: AgentRun) -> None:
    st.info(agent_run.summary)
    for step in agent_run.steps:
        with st.container(border=True):
            status = status_label(step.status)
            st.markdown(f"**{step.number}. {step.name}** | `{status}`")
            st.caption(step.tool)
            st.write(step.action)
            st.write(step.observation)


def render_catalog(songs, facts) -> None:
    song_rows = [
        {
            "title": song["title"],
            "artist": song["artist"],
            "genre": song["genre"],
            "mood": song["mood"],
            "energy": song["energy"],
            "acousticness": song["acousticness"],
        }
        for song in songs
    ]
    fact_rows = [
        {
            "title": fact.title,
            "artist": fact.artist,
            "topic": fact.topic,
            "fact": fact.fact,
            "source": fact.source,
        }
        for fact in facts
    ]
    st.dataframe(song_rows, use_container_width=True, hide_index=True)
    st.dataframe(fact_rows, use_container_width=True, hide_index=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    st.set_page_config(
        page_title="VibeFinder",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; }
        div[data-testid="stMetricValue"] { font-size: 1.4rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    songs, facts = load_catalog_data()
    options = catalog_options(songs)

    with st.sidebar:
        st.title("VibeFinder")
        preset_name = st.selectbox("Preset", [*PRESET_PROFILES.keys(), "Custom"])
        preset = PRESET_PROFILES.get(preset_name, PRESET_PROFILES["Chill Lofi"])

        genre_index = (
            options.genres.index(preset["favorite_genre"])
            if preset["favorite_genre"] in options.genres
            else 0
        )
        mood_index = (
            options.moods.index(preset["favorite_mood"])
            if preset["favorite_mood"] in options.moods
            else 0
        )

        favorite_genre = st.selectbox("Genre", options.genres, index=genre_index)
        favorite_mood = st.selectbox("Mood", options.moods, index=mood_index)
        target_energy = st.slider(
            "Target energy",
            min_value=0.0,
            max_value=1.0,
            value=float(preset["target_energy"]),
            step=0.01,
        )
        likes_acoustic = st.toggle(
            "Acoustic preference",
            value=bool(preset["likes_acoustic"]),
        )
        k = st.slider("Recommendations", min_value=1, max_value=5, value=3, step=1)
        minimum_confidence = st.slider(
            "Self-check threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.30,
            step=0.05,
        )
        run_requested = st.button("Run Agent", type="primary", use_container_width=True)

    user_profile = build_user_profile(
        favorite_genre=favorite_genre,
        favorite_mood=favorite_mood,
        target_energy=target_energy,
        likes_acoustic=likes_acoustic,
    )
    if run_requested or "agent_run" not in st.session_state:
        st.session_state.agent_run = run_agent(
            user_profile=user_profile,
            songs=songs,
            facts=facts,
            k=k,
            minimum_confidence=minimum_confidence,
        )

    agent_run = st.session_state.agent_run

    st.title("VibeFinder")
    summary_cols = st.columns(4)
    summary_cols[0].metric("Songs", len(songs))
    summary_cols[1].metric("Facts", len(facts))
    summary_cols[2].metric("Accepted", len(agent_run.recommendations))
    summary_cols[3].metric(
        "Self-check",
        "Passed" if agent_run.passed_self_check else "Review",
    )

    rec_tab, trace_tab, catalog_tab = st.tabs(
        ["Recommendations", "Agent Trace", "Catalog"]
    )
    with rec_tab:
        render_recommendations(agent_run)
    with trace_tab:
        render_agent_trace(agent_run)
    with catalog_tab:
        render_catalog(songs, facts)


if __name__ == "__main__":
    main()
