from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

from domain.offer import Offer
from services.project import Project


class View(str, Enum):
    OFFER = 'offer'
    COMPARISON = 'comparison'


@dataclass
class PendingUndo:
    label: str
    restore: Callable[[], None]


@dataclass
class MainPageState:
    opened_offer: Offer | None = None
    current_view: View = View.OFFER
    comparison_project: Project | None = None
    selected_offer: Offer | None = None
    selected_project: Project | None = None
    upload_project: str | None = None
    extract_requested_offers: set[Offer] = field(default_factory=set)
    expanded_project_names: set[str] = field(default_factory=set)
    tree_search: str = ''
    selected_offers: set[Offer] = field(default_factory=set)
    selection_mode: bool = False
    pending_undo: PendingUndo | None = None
