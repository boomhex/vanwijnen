from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass
class Posten:
    omschrijving: str = ''
    categorie: str = ''
    aantal: str = ''
    eenheid: str = ''
    eenheidsprijs: str = ''
    totaalbedrag: str = ''

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Posten':
        return cls(
            omschrijving=data.get('Omschrijving', '') or '',
            categorie=data.get('Categorie', '') or '',
            aantal=data.get('Aantal', '') or '',
            eenheid=data.get('Eenheid', '') or '',
            eenheidsprijs=data.get('Eenheidsprijs', '') or '',
            totaalbedrag=data.get('Totaalbedrag', '') or '',
        )

    def to_dict(self) -> dict[str, str]:
        return {
            'Omschrijving': self.omschrijving,
            'Categorie': self.categorie,
            'Aantal': self.aantal,
            'Eenheid': self.eenheid,
            'Eenheidsprijs': self.eenheidsprijs,
            'Totaalbedrag': self.totaalbedrag,
        }


@dataclass(frozen=True)
class Offer:
    """Pure representation of an offer folder."""

    path: Path

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def project_name(self) -> str:
        return self.path.parent.name

    @property
    def document(self) -> Path:
        return self.path / 'document.pdf'

    @property
    def extract_path(self) -> Path:
        return self.path / 'extract.json'

    @property
    def raw_path(self) -> Path:
        return self.path / 'raw.txt'

    @property
    def llm_response_path(self) -> Path:
        return self.path / 'llm_response.txt'

    @property
    def comparison_key(self) -> str:
        return f'{self.name}.pdf'

    def pdf_url(self, storage_dir: Path) -> str:
        relative_path = self.document.relative_to(storage_dir).as_posix()
        return f'/storage/{quote(relative_path, safe="/")}'
