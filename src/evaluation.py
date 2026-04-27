from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

try:
    from .rag import RecommendationAssistant, RecommendationResult
except ImportError:
    from rag import RecommendationAssistant, RecommendationResult


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    profile: Dict
    expected_top_genres: Sequence[str]
    minimum_confidence: float = 0.30


@dataclass(frozen=True)
class EvaluationOutcome:
    case_name: str
    passed: bool
    top_song: str
    confidence: float
    checks: List[str]


@dataclass(frozen=True)
class EvaluationSummary:
    outcomes: List[EvaluationOutcome]

    @property
    def passed(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.passed)

    @property
    def total(self) -> int:
        return len(self.outcomes)


DEFAULT_EVALUATION_CASES = [
    EvaluationCase(
        name="Chill lofi acoustic",
        profile={
            "favorite_genre": "lofi",
            "favorite_mood": "chill",
            "target_energy": 0.38,
            "likes_acoustic": True,
        },
        expected_top_genres=("lofi", "ambient"),
    ),
    EvaluationCase(
        name="High-energy pop",
        profile={
            "favorite_genre": "pop",
            "favorite_mood": "happy",
            "target_energy": 0.85,
            "likes_acoustic": False,
        },
        expected_top_genres=("pop", "indie pop"),
    ),
    EvaluationCase(
        name="Deep intense rock",
        profile={
            "favorite_genre": "rock",
            "favorite_mood": "intense",
            "target_energy": 0.90,
            "likes_acoustic": False,
        },
        expected_top_genres=("rock", "pop", "metal"),
    ),
]


def evaluate_assistant(
    assistant: RecommendationAssistant,
    cases: Iterable[EvaluationCase] = DEFAULT_EVALUATION_CASES,
    k: int = 3,
) -> EvaluationSummary:
    outcomes = [_evaluate_case(assistant, case, k=k) for case in cases]
    return EvaluationSummary(outcomes=outcomes)


def _evaluate_case(
    assistant: RecommendationAssistant,
    case: EvaluationCase,
    k: int,
) -> EvaluationOutcome:
    checks: List[str] = []
    try:
        results = assistant.recommend(case.profile, k=k)
    except Exception as exc:
        return EvaluationOutcome(
            case_name=case.name,
            passed=False,
            top_song="none",
            confidence=0.0,
            checks=[f"failed to run: {exc}"],
        )

    if not results:
        return EvaluationOutcome(
            case_name=case.name,
            passed=False,
            top_song="none",
            confidence=0.0,
            checks=["no recommendations returned"],
        )

    top_result = results[0]
    checks.extend(_result_checks(top_result, case))
    passed = all(check.startswith("PASS") for check in checks)
    return EvaluationOutcome(
        case_name=case.name,
        passed=passed,
        top_song=top_result.song["title"],
        confidence=top_result.generated.confidence,
        checks=checks,
    )


def _result_checks(result: RecommendationResult, case: EvaluationCase) -> List[str]:
    checks: List[str] = []
    top_genre = result.song["genre"]
    if top_genre in case.expected_top_genres:
        checks.append(f"PASS top genre '{top_genre}' matched expectation")
    else:
        checks.append(f"FAIL top genre '{top_genre}' was unexpected")

    if result.generated.answer.strip():
        checks.append("PASS generated explanation was non-empty")
    else:
        checks.append("FAIL generated explanation was empty")

    if result.generated.confidence >= case.minimum_confidence:
        checks.append(
            f"PASS confidence {result.generated.confidence:.2f} met threshold"
        )
    else:
        checks.append(
            f"FAIL confidence {result.generated.confidence:.2f} missed threshold"
        )

    if result.context.facts and result.generated.citations:
        checks.append("PASS retrieved context and citations were present")
    else:
        checks.append("FAIL retrieved context or citations were missing")

    return checks
