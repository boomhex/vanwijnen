from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote


@dataclass
class Posten:
    omschrijving: str = ''
    beschrijving: str = ''
    categorie: str = ''
    aantal: str = ''
    eenheid: str = ''
    eenheidsprijs: str = ''
    totaalbedrag: str = ''
    extra: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Posten':
        known_fields = {
            'Omschrijving',
            'Beschrijving',
            'Categorie',
            'Aantal',
            'Eenheid',
            'Eenheidsprijs',
            'Totaalbedrag',
        }
        return cls(
            omschrijving=data.get('Omschrijving', '') or '',
            beschrijving=data.get('Beschrijving', '') or '',
            categorie=data.get('Categorie', '') or '',
            aantal=data.get('Aantal', '') or '',
            eenheid=data.get('Eenheid', '') or '',
            eenheidsprijs=data.get('Eenheidsprijs', '') or '',
            totaalbedrag=data.get('Totaalbedrag', '') or '',
            extra={key: value for key, value in data.items() if key not in known_fields},
        )

    def to_dict(self) -> dict[str, Any]:
        post = dict(self.extra or {})
        post.update({
            'Omschrijving': self.omschrijving,
            'Beschrijving': self.beschrijving,
            'Categorie': self.categorie,
            'Aantal': self.aantal,
            'Eenheid': self.eenheid,
            'Eenheidsprijs': self.eenheidsprijs,
            'Totaalbedrag': self.totaalbedrag,
        })
        return post


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
    def status_path(self) -> Path:
        return self.path / 'status.json'

    @property
    def comparison_key(self) -> str:
        return f'{self.name}.pdf'

    def pdf_url(self, storage_dir: Path) -> str:
        relative_path = self.document.relative_to(storage_dir).as_posix()
        return f'/storage/{quote(relative_path, safe="/")}'
