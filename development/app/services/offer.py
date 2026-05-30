from pathlib import Path
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from urllib.parse import quote

if TYPE_CHECKING:
    from services.folder_handler import FolderHandler


@dataclass
class Posten:
    omschrijving: str = ''
    aantal: str = ''
    eenheid: str = ''
    eenheidsprijs: str = ''
    totaalbedrag: str = ''

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Posten':
        return cls(
            omschrijving=data.get('Omschrijving', '') or '',
            aantal=data.get('Aantal', '') or '',
            eenheid=data.get('Eenheid', '') or '',
            eenheidsprijs=data.get('Eenheidsprijs', '') or '',
            totaalbedrag=data.get('Totaalbedrag', '') or '',
        )

    def to_dict(self) -> dict[str, str]:
        return {
            'Omschrijving': self.omschrijving,
            'Aantal': self.aantal,
            'Eenheid': self.eenheid,
            'Eenheidsprijs': self.eenheidsprijs,
            'Totaalbedrag': self.totaalbedrag,
        }


class Offer:
    """Represents an offer with its PDF and extracted data."""

    def __init__(self, offer_dir: Path, folder_handler: 'FolderHandler') -> None:
        """
        Initialize an Offer.
        
        Args:
            offer_dir: Path to the offer folder (storage/<project>/<offer_name>)
            folder_handler: FolderHandler instance for file operations
        """
        self.path = offer_dir
        self.folder_handler = folder_handler
        self.name = offer_dir.name
        self.project_name = folder_handler.project_name_for_file(self.document)

    @property
    def document(self) -> Path:
        """Get the PDF document path."""
        return self.folder_handler.offer_document_path(self.path)

    @property
    def extract_path(self) -> Path:
        """Get the extract.json path."""
        return self.folder_handler.offer_extract_path(self.path)

    @property
    def raw_path(self) -> Path:
        """Get the raw.txt path."""
        return self.folder_handler.offer_raw_path(self.path)

    def load_data(self) -> dict[str, Any] | None:
        """Load the extracted offer data."""
        return self.folder_handler.load_result(self.document)

    def save_data(self, data: dict[str, Any]) -> None:
        """Save the extracted offer data."""
        self.folder_handler.save_result(self.document, data)

    @staticmethod
    def _posten_from_result(result: dict[str, Any]) -> list[Posten]:
        posten = result.get('Posten', [])
        if not isinstance(posten, list):
            return []
        return [Posten.from_dict(post) for post in posten if isinstance(post, dict)]

    @staticmethod
    def _posten_to_result(posten: list[Posten]) -> list[dict[str, str]]:
        return [post.to_dict() for post in posten]

    def update_summary_value(self, result: dict[str, Any], field: str, value: str) -> None:
        """Update a top-level summary field and persist the result."""
        result[field] = value
        self.save_data(result)

    def add_summary_field(self, result: dict[str, Any], field: str, value: str | None) -> None:
        """Add a top-level summary field and persist the result."""
        result[field] = value or ''
        self.save_data(result)

    def add_post_row(self, result: dict[str, Any]) -> Posten:
        """Append an empty post row and persist the result."""
        row = Posten()
        posten = self._posten_from_result(result)
        posten.append(row)
        result['Posten'] = self._posten_to_result(posten)
        self.save_data(result)
        return row

    def update_post_row(self, result: dict[str, Any], row_id: int | None, field: str | None, value: str) -> None:
        """Update a post row cell and persist the result."""
        fields = {'Omschrijving', 'Aantal', 'Eenheid', 'Eenheidsprijs', 'Totaalbedrag'}
        if field not in fields or row_id is None:
            return

        posten = self._posten_from_result(result)
        if row_id < 0 or row_id >= len(posten):
            return

        setattr(posten[row_id], self._field_name(field), value)
        result['Posten'] = self._posten_to_result(posten)
        self.save_data(result)

    def delete_post_row(self, result: dict[str, Any], row_id: int | None) -> None:
        """Delete a post row and persist the result."""
        if row_id is None:
            return

        posten = self._posten_from_result(result)
        if row_id < 0 or row_id >= len(posten):
            return

        posten.pop(row_id)
        result['Posten'] = self._posten_to_result(posten)
        self.save_data(result)

    def posten_list(self, result: dict[str, Any]) -> list[Posten]:
        """Get the offer posts as dataclass objects."""
        return self._posten_from_result(result)

    def save_posten_list(self, result: dict[str, Any], posten: list[Posten]) -> None:
        """Persist the given post list to the result."""
        result['Posten'] = self._posten_to_result(posten)
        self.save_data(result)

    @staticmethod
    def _field_name(field: str) -> str:
        mapping = {
            'Omschrijving': 'omschrijving',
            'Aantal': 'aantal',
            'Eenheid': 'eenheid',
            'Eenheidsprijs': 'eenheidsprijs',
            'Totaalbedrag': 'totaalbedrag',
        }
        return mapping[field]

    def save_raw_text(self, text: str) -> None:
        """Save the raw PDF extraction text."""
        self.folder_handler.save_raw_pdf_text(self.document, text)

    @property
    def comparison_key(self) -> str:
        """Return the stable key used in comparison JSON for this offer."""
        return f'{self.name}.pdf'

    def pdf_url(self, storage_dir: Path) -> str:
        """Get the URL for viewing the PDF."""
        relative_path = self.document.relative_to(storage_dir).as_posix()
        return f'/storage/{quote(relative_path, safe="/")}'

    def rename(self, new_name: str | None) -> 'Offer':
        """Rename the offer (returns updated Offer instance)."""
        new_path = self.folder_handler.rename_file(self.document, new_name).parent
        return Offer(new_path, self.folder_handler)

    def move_to_project(self, target_project: str | None) -> 'Offer':
        """Move offer to a different project (returns updated Offer instance)."""
        new_path = self.folder_handler.move_file(self.document, target_project).parent
        return Offer(new_path, self.folder_handler)

    def delete(self) -> None:
        """Delete the entire offer folder."""
        self.folder_handler.delete_file(self.document)

    def __repr__(self) -> str:
        return f"Offer({self.name})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Offer):
            return NotImplemented
        return self.path == other.path

    def __hash__(self) -> int:
        return hash(self.path)

