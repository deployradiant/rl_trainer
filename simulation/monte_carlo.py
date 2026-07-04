"""Seeded Level 0 match-outcome Monte Carlo simulation."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

from data.match_state import MatchState, TeamState


@dataclass(frozen=True)
class OutcomeSimulationConfig:
    rollouts: int = 10000
    seed: int = 7
    horizon_minutes: float | None = None
    base_total_goals_per_90: float = 2.6
    red_card_goal_multiplier: float = 0.78
    score_state_push: float = 0.08

    def validate(self) -> None:
        if self.rollouts <= 0:
            raise ValueError("rollouts must be positive")
        if self.base_total_goals_per_90 <= 0:
            raise ValueError("base_total_goals_per_90 must be positive")
        if self.red_card_goal_multiplier <= 0:
            raise ValueError("red_card_goal_multiplier must be positive")
        if self.horizon_minutes is not None and self.horizon_minutes < 0:
            raise ValueError("horizon_minutes must be non-negative")


def simulate_outcomes(
    state: MatchState,
    config: OutcomeSimulationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run a simple outcome Monte Carlo from the current score and clock."""
    state.validate()
    config.validate()

    rng = random.Random(config.seed)
    remaining = state.remaining_minutes(config.horizon_minutes)
    home_rate, away_rate = _expected_goals_per_90(state, config)
    home_lambda = home_rate * remaining / 90.0
    away_lambda = away_rate * remaining / 90.0

    result_counts: Counter[str] = Counter()
    scoreline_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []

    for rollout_id in range(config.rollouts):
        home_added = _sample_poisson(rng, home_lambda)
        away_added = _sample_poisson(rng, away_lambda)
        home_goals = state.home.goals + home_added
        away_goals = state.away.goals + away_added
        result = _result_label(home_goals, away_goals)
        scoreline = f"{home_goals}-{away_goals}"
        result_counts[result] += 1
        scoreline_counts[scoreline] += 1
        if len(samples) < 50:
            samples.append(
                {
                    "rollout_id": rollout_id,
                    "home_goals": home_goals,
                    "away_goals": away_goals,
                    "home_added_goals": home_added,
                    "away_added_goals": away_added,
                    "result": result,
                }
            )

    summary = {
        "metadata": state.metadata.__dict__,
        "config": asdict(config),
        "teams": {
            "home": state.home.__dict__,
            "away": state.away.__dict__,
        },
        "clock": state.clock.__dict__,
        "remaining_minutes": remaining,
        "expected_additional_goals": {
            "home": home_lambda,
            "away": away_lambda,
        },
        "result_probabilities": _probabilities(result_counts, config.rollouts),
        "top_scorelines": [
            {"scoreline": scoreline, "probability": count / config.rollouts, "count": count}
            for scoreline, count in scoreline_counts.most_common(12)
        ],
        "rollouts": config.rollouts,
    }
    return summary, samples


def _expected_goals_per_90(
    state: MatchState,
    config: OutcomeSimulationConfig,
) -> tuple[float, float]:
    base = config.base_total_goals_per_90 / 2.0
    home_strength = _card_adjusted_strength(state.home, config)
    away_strength = _card_adjusted_strength(state.away, config)
    average_strength = (home_strength + away_strength) / 2.0
    home_rate = base * home_strength / average_strength
    away_rate = base * away_strength / average_strength

    score_diff = state.home.goals - state.away.goals
    if score_diff > 0:
        away_rate *= 1.0 + config.score_state_push
        home_rate *= 1.0 - config.score_state_push / 2.0
    elif score_diff < 0:
        home_rate *= 1.0 + config.score_state_push
        away_rate *= 1.0 - config.score_state_push / 2.0

    return max(0.01, home_rate), max(0.01, away_rate)


def _card_adjusted_strength(team: TeamState, config: OutcomeSimulationConfig) -> float:
    return team.strength * math.pow(config.red_card_goal_multiplier, team.red_cards)


def _sample_poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _result_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home_win"
    if away_goals > home_goals:
        return "away_win"
    return "draw"


def _probabilities(counts: Counter[str], total: int) -> dict[str, float]:
    return {
        "home_win": counts["home_win"] / total,
        "draw": counts["draw"] / total,
        "away_win": counts["away_win"] / total,
    }
