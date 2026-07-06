import asyncio
from pathlib import Path
from collections.abc import Callable
from decimal import Decimal
import json


from nicegui import run, ui

from services.comparison_matcher import ComparisonMatcher
from services.folder_handler import FolderHandler
from interface.page_state import MainPageState, View
from .subpage import SubPage
from .comparison_page import ComparisonPage
from .offer_page import OfferPage


class RightSide(SubPage):
    def __init__(self, *, state: MainPageState, folder_handler: FolderHandler, projects_dir: Path) -> None:
        super().__init__(state, folder_handler, projects_dir, None)

        self.comparison_page = ComparisonPage(
            state, folder_handler, projects_dir, self.schedule_refresh_safe
        )
        self.offer_page = OfferPage(
            state, folder_handler, projects_dir, self.schedule_refresh_safe
        )

    def render(self) -> None:
        self.container = ui.column().classes('w-full overflow-hidden')
        with self.container:
            self.show()

    def refresh(self) -> None:
        if not self.container_is_live():
            return

        self.container.clear()
        with self.container:
            self.show()

    def schedule_refresh(self) -> None:
        if not self.container_is_live():
            return
        asyncio.ensure_future(self._deferred_refresh())

    async def _deferred_refresh(self) -> None:
        await asyncio.sleep(0.05)
        self.refresh()

    def schedule_refresh_safe(self) -> None:
        self.schedule_refresh()

    def container_is_live(self) -> bool:
        if self.container is None:
            return False
        if getattr(self.container, 'is_deleted', False):
            return False

        client = getattr(self.container, 'client', None)
        if client is None:
            return False

        return not getattr(client, '_deleted', False)

    def show(self) -> None:
        if self.state.current_view == View.COMPARISON:
            self.comparison_page.render()
            return

        if self.state.current_view == View.OFFER:
            self.offer_page.render()
            return
