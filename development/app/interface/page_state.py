from dataclasses import dataclass, field
from enum import Enum

from domain.offer import Offer
from services.project import Project


class View(str, Enum):
    OFFER = 'offer'
    COMPARISON = 'comparison'


@dataclass
class MainPageState:
    opened_offer: Offer | None = None
    current_view: View = View.OFFER
    comparison_project: Project | None = None
    upload_project: str | None = None
    extract_requested_offers: set[Offer] = field(default_factory=set)
