"""Shared probe result contracts for physics validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProbeResult:
    """Structured evidence emitted by one physical validation probe."""

    name: str
    passed: bool
    tier_evidence: int
    objects: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    diagnosis: str = ""
    repair: str = ""
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "tier_evidence": self.tier_evidence,
            "objects": list(self.objects),
            "metrics": self.metrics,
            "diagnosis": self.diagnosis,
            "repair": self.repair,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ProbePlan:
    """Probe names used to inspect one declared subgoal."""

    subgoal_kind: str
    probes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "subgoal_kind": self.subgoal_kind,
            "probes": list(self.probes),
        }
