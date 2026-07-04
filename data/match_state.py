"""Canonical match-state schema for simulation inputs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeedMetadata:
    source: str
    competition: str
    match_id: str
    fetch_time_epoch_utc: int
    snapshot_epoch_utc: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FeedMetadata":
        return cls(
            source=str(payload["source"]),
            competition=str(payload["competition"]),
            match_id=str(payload["match_id"]),
            fetch_time_epoch_utc=int(payload["fetch_time_epoch_utc"]),
            snapshot_epoch_utc=int(payload["snapshot_epoch_utc"]),
        )


@dataclass(frozen=True)
class TeamState:
    team_id: str
    name: str
    goals: int = 0
    red_cards: int = 0
    strength: float = 1.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TeamState":
        return cls(
            team_id=str(payload["team_id"]),
            name=str(payload["name"]),
            goals=int(payload.get("goals", 0)),
            red_cards=int(payload.get("red_cards", 0)),
            strength=float(payload.get("strength", 1.0)),
        )


@dataclass(frozen=True)
class MatchClock:
    period: str
    minute: int
    second: int = 0
    stoppage_minute: int = 0
    regulation_minutes: int = 90

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MatchClock":
        return cls(
            period=str(payload.get("period", "regulation")),
            minute=int(payload["minute"]),
            second=int(payload.get("second", 0)),
            stoppage_minute=int(payload.get("stoppage_minute", 0)),
            regulation_minutes=int(payload.get("regulation_minutes", 90)),
        )

    @property
    def elapsed_minutes(self) -> float:
        return self.minute + self.second / 60.0


@dataclass(frozen=True)
class MatchState:
    metadata: FeedMetadata
    home: TeamState
    away: TeamState
    clock: MatchClock
    neutral_venue: bool = True
    possession_team_id: str | None = None
    phase: str = "unknown"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MatchState":
        state = cls(
            metadata=FeedMetadata.from_dict(payload["metadata"]),
            home=TeamState.from_dict(payload["home"]),
            away=TeamState.from_dict(payload["away"]),
            clock=MatchClock.from_dict(payload["clock"]),
            neutral_venue=bool(payload.get("neutral_venue", True)),
            possession_team_id=payload.get("possession_team_id"),
            phase=str(payload.get("phase", "unknown")),
        )
        state.validate()
        return state

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.metadata.fetch_time_epoch_utc < 0:
            raise ValueError("fetch_time_epoch_utc must be non-negative")
        if self.metadata.snapshot_epoch_utc < 0:
            raise ValueError("snapshot_epoch_utc must be non-negative")
        if self.clock.minute < 0 or self.clock.second < 0:
            raise ValueError("clock minute and second must be non-negative")
        if self.clock.second >= 60:
            raise ValueError("clock second must be < 60")
        if self.clock.regulation_minutes <= 0:
            raise ValueError("regulation_minutes must be positive")
        for team in [self.home, self.away]:
            if team.goals < 0:
                raise ValueError(f"{team.name} goals must be non-negative")
            if team.red_cards < 0:
                raise ValueError(f"{team.name} red_cards must be non-negative")
            if team.strength <= 0:
                raise ValueError(f"{team.name} strength must be positive")
        if self.possession_team_id not in {None, self.home.team_id, self.away.team_id}:
            raise ValueError("possession_team_id must match home, away, or null")

    def remaining_minutes(self, horizon_minutes: float | None = None) -> float:
        remaining = max(0.0, self.clock.regulation_minutes - self.clock.elapsed_minutes)
        if horizon_minutes is None:
            return remaining
        return min(remaining, max(0.0, horizon_minutes))


def load_match_state(path: str | Path) -> MatchState:
    return MatchState.from_dict(json.loads(Path(path).read_text()))


def save_match_state(state: MatchState, path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(state.to_dict(), indent=2, sort_keys=True))
