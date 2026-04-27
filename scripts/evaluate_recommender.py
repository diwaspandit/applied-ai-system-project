from pathlib import Path
import logging
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import evaluate_assistant
from src.rag import (
    KnowledgeRetriever,
    RecommendationAssistant,
    build_default_generator,
    load_knowledge_facts,
)
from src.recommender import load_songs


def build_assistant() -> RecommendationAssistant:
    songs = load_songs(str(PROJECT_ROOT / "data" / "songs.csv"))
    facts = load_knowledge_facts(str(PROJECT_ROOT / "data" / "music_knowledge.csv"))
    return RecommendationAssistant(
        songs=songs,
        retriever=KnowledgeRetriever(facts),
        generator=build_default_generator(),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    summary = evaluate_assistant(build_assistant())

    print("Reliability evaluation")
    print(f"Passed {summary.passed} out of {summary.total} cases")
    for outcome in summary.outcomes:
        status = "PASS" if outcome.passed else "FAIL"
        print(
            f"\n[{status}] {outcome.case_name}: "
            f"top='{outcome.top_song}', confidence={outcome.confidence:.2f}"
        )
        for check in outcome.checks:
            print(f"- {check}")


if __name__ == "__main__":
    main()
