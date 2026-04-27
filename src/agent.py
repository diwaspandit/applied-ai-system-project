from dataclasses import dataclass
from typing import Dict, List

try:
    from .rag import RecommendationAssistant, RecommendationResult, validate_user_profile
except ImportError:
    from rag import RecommendationAssistant, RecommendationResult, validate_user_profile


@dataclass(frozen=True)
class AgentStep:
    """One observable planning, tool, or checking step from the agent."""

    number: int
    name: str
    tool: str
    action: str
    observation: str
    status: str


@dataclass(frozen=True)
class AgentRun:
    """The agent's trace plus its accepted recommendation results."""

    steps: List[AgentStep]
    recommendations: List[RecommendationResult]
    passed_self_check: bool
    summary: str


class AgenticMusicAgent:
    """Plans, calls recommendation tools, and self-checks grounded outputs."""

    def __init__(self, assistant: RecommendationAssistant, minimum_confidence: float = 0.30):
        self._assistant = assistant
        self._minimum_confidence = minimum_confidence

    def run(self, user_prefs: Dict, k: int = 3) -> AgentRun:
        steps: List[AgentStep] = []
        steps.append(
            self._step(
                steps,
                name="Plan workflow",
                tool="agent_planner",
                action=(
                    "Plan: validate profile, rank catalog, retrieve local facts, "
                    "generate grounded explanations, then self-check outputs."
                ),
                observation="Prepared a five-step recommendation workflow.",
                status="passed",
            )
        )

        try:
            validated = validate_user_profile(user_prefs)
        except ValueError as exc:
            steps.append(
                self._step(
                    steps,
                    name="Validate profile",
                    tool="profile_validator",
                    action="Check required fields and normalize values.",
                    observation=str(exc),
                    status="failed",
                )
            )
            return AgentRun(
                steps=steps,
                recommendations=[],
                passed_self_check=False,
                summary="Agent stopped because the user profile was invalid.",
            )

        validation_note = (
            "; ".join(validated.guardrail_notes)
            if validated.guardrail_notes
            else "Profile was valid with no corrections."
        )
        steps.append(
            self._step(
                steps,
                name="Validate profile",
                tool="profile_validator",
                action="Check required fields and normalize values.",
                observation=validation_note,
                status="passed",
            )
        )

        recommendations = self._assistant.recommend(validated.profile, k=k)
        top_titles = ", ".join(result.song["title"] for result in recommendations) or "none"
        steps.append(
            self._step(
                steps,
                name="Call RAG recommender",
                tool="recommendation_assistant",
                action="Rank songs, retrieve facts, and generate explanations.",
                observation=f"Generated {len(recommendations)} candidates: {top_titles}.",
                status="passed" if recommendations else "failed",
            )
        )

        accepted, check_messages = self._self_check(recommendations)
        passed_self_check = len(accepted) == len(recommendations) and bool(accepted)
        status = "passed" if passed_self_check else "warning"
        steps.append(
            self._step(
                steps,
                name="Self-check recommendations",
                tool="grounding_checker",
                action=(
                    "Verify each output has retrieved context, citations, non-empty "
                    "explanation text, acceptable confidence, and no live-web claims."
                ),
                observation="; ".join(check_messages),
                status=status,
            )
        )

        if not accepted and recommendations:
            accepted = recommendations[:1]
            steps.append(
                self._step(
                    steps,
                    name="Fallback decision",
                    tool="agent_decider",
                    action="Keep the top scored candidate when no result passes every check.",
                    observation="Returned the top candidate with warning status.",
                    status="warning",
                )
            )

        summary = (
            f"Agent accepted {len(accepted)} of {len(recommendations)} generated candidates."
        )
        steps.append(
            self._step(
                steps,
                name="Finalize answer",
                tool="agent_decider",
                action="Return accepted recommendations and the observable trace.",
                observation=summary,
                status="passed" if accepted else "failed",
            )
        )

        return AgentRun(
            steps=steps,
            recommendations=accepted,
            passed_self_check=passed_self_check,
            summary=summary,
        )

    def _self_check(
        self,
        recommendations: List[RecommendationResult],
    ) -> tuple[List[RecommendationResult], List[str]]:
        accepted: List[RecommendationResult] = []
        messages: List[str] = []
        for result in recommendations:
            checks = {
                "context": bool(result.context.facts),
                "citations": bool(result.generated.citations),
                "answer": bool(result.generated.answer.strip()),
                "confidence": result.generated.confidence >= self._minimum_confidence,
                "no_live_web_claim": "live web" not in result.generated.answer.lower(),
            }
            if all(checks.values()):
                accepted.append(result)
                messages.append(f"PASS {result.song['title']} passed grounding checks")
            else:
                failed = ", ".join(name for name, passed in checks.items() if not passed)
                messages.append(f"WARN {result.song['title']} failed: {failed}")
        if not recommendations:
            messages.append("WARN no recommendation candidates were generated")
        return accepted, messages

    @staticmethod
    def _step(
        steps: List[AgentStep],
        name: str,
        tool: str,
        action: str,
        observation: str,
        status: str,
    ) -> AgentStep:
        return AgentStep(
            number=len(steps) + 1,
            name=name,
            tool=tool,
            action=action,
            observation=observation,
            status=status,
        )
