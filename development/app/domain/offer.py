from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from domain.money import UNKNOWN


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


def _field_value(post: Posten, field: str) -> str:
    if field == 'Categorie':
        return post.categorie
    return str((post.extra or {}).get(field, '') or '')


def _is_reliable_group_field(posten: list[Posten], field: str) -> bool:
    """Whether `field` looks like a genuine, repeated section label.

    Some documents reuse a field like Regelnummer or Code as a section
    header spanning several distinct posts (e.g. "V01" for every line item
    under a "V01" heading); others use it as a unique per-post identifier
    (e.g. a bestekcode like "44.31.10-a"), or leave it mostly unset. Only
    the first case is worth grouping by. This mirrors the same "repeated
    label vs. unique identifier" signal already used to guard against
    incorrectly merging distinct posts that happen to share a code — see
    services.extract_offer.merge_post_chunks and
    matching.match_lookup.build_post_code_index.
    """
    if not posten:
        return False

    counts: dict[str, int] = {}
    for post in posten:
        value = _field_value(post, field).strip()
        if not value or value.upper() == UNKNOWN:
            continue
        counts[value] = counts.get(value, 0) + 1

    repeated_values = [value for value, count in counts.items() if count >= 2]
    if len(repeated_values) < 2:
        return False

    grouped_posts = sum(counts[value] for value in repeated_values)
    return grouped_posts / len(posten) >= 0.5


def grouping_fields(posten: list[Posten]) -> list[str]:
    """Tabulator groupBy fields for a list of posts, or [] for a flat table.

    Categorie is used as the outer grouping level when the offer has real
    chapter structure. Regelnummer or Code (whichever looks reliable, see
    _is_reliable_group_field) is added as an inner level when the offer
    reuses it as a section label. An offer where these fields don't show
    that pattern — a unique code per post, or mostly ONBEKEND — gets an
    empty list, so its table renders exactly as it does today.
    """
    fields = []
    if _is_reliable_group_field(posten, 'Categorie'):
        fields.append('Categorie')

    for field in ('Regelnummer', 'Code'):
        if _is_reliable_group_field(posten, field):
            fields.append(field)
            break

    return fields


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
