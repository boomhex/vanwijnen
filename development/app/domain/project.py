from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """Pure representation of a project folder."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def comparison_path(self) -> Path:
        return self.path / 'comparison.json'
