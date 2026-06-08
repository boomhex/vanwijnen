from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from domain.offer import Offer
from domain.project import Project as DomainProject

if TYPE_CHECKING:
    from services.folder_handler import FolderHandler


class Project(DomainProject):
    """Represents a project with offers and a comparison."""

    def __init__(self, path, folder_handler: 'FolderHandler') -> None:
        super().__init__(path)
        self.folder_handler = folder_handler

    def offers(self) -> list[Offer]:
        return [
            self.folder_handler.offer_from_path(offer_dir)
            for offer_dir in self.folder_handler.project_offer_paths(self.path)
        ]

    def offer_from_filename(self, filename: str) -> Offer | None:
        for offer_path in self.folder_handler.project_offer_paths(self.path):
            if offer_path.name == filename:
                return self.folder_handler.offer_from_path(offer_path)
        return None

    def files(self) -> list[Path]:
        return [offer.document for offer in self.offers() if offer.document.exists()]

    def load_comparison(self) -> dict[str, Any]:
        return self.folder_handler.load_comparison(self.path)

    def save_comparison(self, comparison: dict[str, Any]) -> None:
        self.folder_handler.save_comparison(self.path, comparison)

    def rename(self, new_name: str | None) -> 'Project':
        return self.folder_handler.rename_project(self, new_name)

    def __repr__(self) -> str:
        return f'Project({self.name})'
