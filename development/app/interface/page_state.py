from dataclasses import dataclass, field
from domain.offer import Offer
from services.project import Project


@dataclass
class MainPageState:
    opened_offer: Offer | None = None
    current_view: str = 'offer'
    comparison_project: Project | None = None
    upload_project: str | None = None
    extract_requested_offers: set[Offer] = field(default_factory=set)
