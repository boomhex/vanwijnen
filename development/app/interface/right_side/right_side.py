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
        self.render_undo_banner()

        if self.state.current_view == View.COMPARISON:
            self.comparison_page.render()
            return

        if self.state.current_view == View.OFFER:
            self.offer_page.render()
            return

    def render_undo_banner(self) -> None:
        pending = self.state.pending_undo
        if pending is None:
            return

        with ui.row().classes(
            'items-center gap-2 w-full bg-gray-100 border border-gray-300 rounded px-3 py-2 mt-2'
        ):
            ui.icon('info').classes('text-gray-600')
            ui.label(pending.label).classes('text-sm grow')
            ui.button('Undo', on_click=self.undo_pending).props('flat dense no-caps size=sm color=primary')

        ui.timer(8.0, self.expire_pending_undo, once=True)

    def undo_pending(self) -> None:
        pending = self.state.pending_undo
        if pending is None:
            return

        self.state.pending_undo = None
        pending.restore()
        self.refresh()

    def expire_pending_undo(self) -> None:
        if self.state.pending_undo is None:
            return

        self.state.pending_undo = None
        self.refresh_safe()

    def refresh_safe(self) -> None:
        try:
            self.refresh()
        except RuntimeError:
            pass
