from pathlib import Path
from dataclasses import dataclass, field
from services.offer import Offer


@dataclass
class MainPageState:
    opened_offer: Offer | None = None
    current_view: str = 'offer'
    comparison_project: Path | None = None
    upload_project: str | None = None
    extract_requested_offers: set[Offer] = field(default_factory=set)
