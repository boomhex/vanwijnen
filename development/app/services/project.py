from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from services.offer import Offer

if TYPE_CHECKING:
    from services.folder_handler import FolderHandler


class Project:
    """Represents a project with offers and a comparison."""

    def __init__(self, path: Path, folder_handler: 'FolderHandler') -> None:
        self.path = path
        self.folder_handler = folder_handler

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def comparison_path(self) -> Path:
        return self.folder_handler.project_comparison_path(self.path)

    def offers(self) -> list[Offer]:
        return [
            self.folder_handler.offer_from_path(offer_dir)
            for offer_dir in self.folder_handler.project_offer_paths(self.path)
        ]

    def offer_from_filename(self, filename: str) -> Offer:
        for offer_path in self.folder_handler.project_offer_paths(self.path):
            if offer_path.name == filename:
                return self.folder_handler.offer_from_path(offer_path)
        return None # No offer with filename found

    def files(self) -> list[Path]:
        return [offer.document for offer in self.offers() if offer.document.exists()]

    def load_comparison(self) -> dict[str, Any]:
        return self.folder_handler.load_comparison(self.path)

    def save_comparison(self, comparison: dict[str, Any]) -> None:
        self.folder_handler.save_comparison(self.path, comparison)

    def __repr__(self) -> str:
        return f'Project({self.name})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Project):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)
