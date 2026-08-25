from __future__ import annotations

import unittest

from app.cmiygl.mapService.mapModel import RouteStep
from app.cmiygl.mapService.tool_registry import (
    RankedPlaceCandidate,
    RankedRecoveredLocationCandidate,
    RoutePlan,
)
from app.cmiygl.realtime.navigation_agent import NavigationAgent
from app.cmiygl.realtime.orchestrator import CmiyglSessionOrchestrator
from app.cmiygl.realtime.session import build_session_record


class FakeMapTools:
    def __init__(self) -> None:
        self.last_recover_last_known_location = None

    def geocode(self, query, *, limit: int = 5):
        text = getattr(query, "query", "") or ""
        if "city mall" in text.lower():
            return [
                RankedPlaceCandidate(
                    provider="fake",
                    lat=6.601,
                    lng=3.351,
                    display_name="City Mall, Lagos",
                    name="City Mall",
                    confidence=0.96,
                    rank_score=0.96,
                )
            ]
        return [
            RankedPlaceCandidate(
                provider="fake",
                lat=6.55,
                lng=3.38,
                display_name=text or "Unknown",
                name=text or "Unknown",
                confidence=0.94,
                rank_score=0.94,
            )
        ]

    def reverse_geocode(self, query):
        return RankedPlaceCandidate(
            provider="fake",
            lat=query.lat,
            lng=query.lng,
            display_name="Known Current Location",
            name="Known Current Location",
            confidence=1.0,
            rank_score=1.0,
        )

    def recover_location(self, *, landmarks, last_known_location=None, search_radius: int = 1000):
        self.last_recover_last_known_location = last_known_location
        return [
            RankedRecoveredLocationCandidate(
                lat=6.5401,
                lng=3.3712,
                confidence=0.97,
                rank_score=0.97,
                matched_landmarks=list(landmarks),
                matched_landmark_count=len(landmarks),
                evidence_names=list(landmarks),
                evidence_count=len(landmarks),
                approx_address="St Peter Church Bus Stop, Lagos",
            )
        ]

    def route(self, query):
        return RoutePlan(
            provider="fake",
            distance_meters=1200,
            duration_seconds=900,
            step_count=2,
            first_instruction="Head north on Station Road",
            steps=[
                RouteStep(
                    instruction="Head north on Station Road",
                    distance_meters=400,
                    duration_seconds=240,
                ),
                RouteStep(
                    instruction="Turn right at City Mall",
                    distance_meters=800,
                    duration_seconds=660,
                ),
            ],
        )


class NavigationAgentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.map_tools = FakeMapTools()
        self.agent = NavigationAgent(self.map_tools)
        self.record = build_session_record("test-session")

    def test_destination_confirmation_then_landmark_route_flow(self) -> None:
        first = self.agent.handle_turn(self.record, "Take me to City Mall")
        self.assertIn("where you want to go", first.assistant_text.lower())
        self.assertEqual(self.record.pending_clarification.kind.value, "destination")

        second = self.agent.handle_turn(self.record, "yes")
        self.assertIn("tell me where you are", second.assistant_text.lower())
        self.assertIsNotNone(self.record.navigation.selected_destination)

        third = self.agent.handle_turn(self.record, "I can see St Peter Church and Main Bus Stop")
        self.assertEqual(third.state.value, "navigating")
        self.assertIn("head north on station road", third.assistant_text.lower())
        self.assertIsNotNone(self.record.navigation.active_route)
        self.assertEqual(self.record.navigation.current_route_step_index, 0)

    def test_landmark_recovery_runs_without_last_known_location(self) -> None:
        result = self.agent.handle_turn(self.record, "I can see St Peter Church and Main Bus Stop")
        self.assertEqual(self.map_tools.last_recover_last_known_location, None)
        self.assertIn("where you want to go", result.assistant_text.lower())

    def test_repeat_and_next_step_use_cached_route(self) -> None:
        self.agent.handle_turn(self.record, "Take me to City Mall")
        self.agent.handle_turn(self.record, "yes")
        self.agent.handle_turn(self.record, "I can see St Peter Church and Main Bus Stop")

        repeated = self.agent.handle_turn(self.record, "repeat that")
        self.assertIn("head north on station road", repeated.assistant_text.lower())

        next_step = self.agent.handle_turn(self.record, "next step")
        self.assertIn("turn right at city mall", next_step.assistant_text.lower())
        self.assertEqual(self.record.navigation.current_route_step_index, 1)

    def test_orchestrator_exposes_navigation_turns(self) -> None:
        orchestrator = CmiyglSessionOrchestrator(
            "orchestrator-session",
            navigation_agent=NavigationAgent(self.map_tools),
        )
        result = orchestrator.handle_text_input("Take me to City Mall")
        self.assertEqual(orchestrator.last_turn_result, result)
        self.assertGreater(len(orchestrator.outbound_events), 0)


if __name__ == "__main__":
    unittest.main()
