import asyncio
from pathlib import Path
from typing import Callable

from nicegui import events, ui

from application.offer_service import OfferService
from application.project_service import ProjectService
from application.extraction_job_service import ExtractionJobService
from services.folder_handler import FolderHandler, UNASSIGNED_PROJECT
from services.offer import Offer
from services.project import Project
from interface.page_state import MainPageState
from utils.app_logging import log_action

from . import dialogs
from .actions import DrawerActions, DrawerActionResult
from .header import DrawerHeader
from .toolbar import DrawerToolbar
from .tree import DrawerTree
from .utils import element_is_live


class LeftDrawer:
    def __init__(
        self,
        *,
        state: MainPageState,
        folder_handler: FolderHandler,
        extraction_job_service: ExtractionJobService,
        projects_dir: Path,
        refresh_right_side: Callable[[], None],
    ) -> None:
        self.state = state
        self.folder_handler = folder_handler
        self.extraction_job_service = extraction_job_service
        self.offer_service = OfferService(folder_handler)
        self.project_service = ProjectService(folder_handler)
        self.actions = DrawerActions(
            state=state,
            folder_handler=folder_handler,
            offer_service=self.offer_service,
            project_service=self.project_service,
            extraction_job_service=extraction_job_service,
        )
        self.refresh_right_side = refresh_right_side
        self.file_list_container = None
        self.last_extraction_snapshot = None
        self._polling_timer = None

    def render(self) -> None:
        self._cancel_polling_timer()
        self.file_list_container = ui.column().classes('w-full')
        with self.file_list_container:
            self.file_list()
        self.last_extraction_snapshot = self.extraction_status_snapshot()
        self._polling_timer = ui.timer(2.0, self.poll_extraction_statuses)

    def _cancel_polling_timer(self) -> None:
        if self._polling_timer is not None:
            try:
                self._polling_timer.cancel()
            except Exception:
                pass
            self._polling_timer = None

    async def handle_upload(self, event: events.UploadEventArguments) -> None:
        try:
            await self.folder_handler.add_uploaded_file(event, self.state.upload_project)
        except (FileExistsError, OSError, ValueError) as error:
            log_action('upload_failed', file=event.file.name, project=self.state.upload_project, error=str(error))
            ui.notify(str(error), color='negative')
            return

        log_action('upload', file=event.file.name, project=self.state.upload_project)
        self.schedule_refresh()

    def schedule_refresh(self) -> None:
        if not self.file_list_container_is_live():
            return
        asyncio.ensure_future(self._deferred_refresh_safe())

    async def _deferred_refresh_safe(self) -> None:
        await asyncio.sleep(0.05)
        self.refresh_safe()

    def refresh(self) -> None:
        if not self.file_list_container_is_live():
            return

        self.file_list_container.clear()
        with self.file_list_container:
            self.file_list()

    def refresh_safe(self) -> None:
        try:
            self.refresh()
        except RuntimeError:
            pass

    def poll_extraction_statuses(self) -> None:
        if not self.file_list_container_is_live():
            return

        snapshot = self.extraction_status_snapshot()
        if snapshot == self.last_extraction_snapshot:
            return

        # Refresh on any change so this client also picks up uploads, deletes,
        # renames and finished extractions triggered by other users.
        self.last_extraction_snapshot = snapshot
        self.refresh_safe()

        # Only nudge the right side when the open offer is the one extracting,
        # so unrelated background jobs don't interrupt an in-progress edit
        # elsewhere with a full re-render.
        if self._opened_offer_is_extracting():
            self.schedule_right_side_refresh()

    def _opened_offer_is_extracting(self) -> bool:
        offer = self.state.opened_offer
        if offer is None:
            return False
        return self.actions.offer_item_status(offer).extract_requested

    def extraction_status_snapshot(self) -> tuple[tuple[str, bool, str | None, str | None, str | None, bool], ...]:
        snapshot = []
        for project in self.project_service.list_projects():
            for offer in project.offers():
                status = self.actions.load_extraction_status(offer) or {}
                snapshot.append((
                    str(offer.path),
                    self.extraction_job_service.is_running(offer),
                    status.get('status'),
                    status.get('step'),
                    status.get('updated_at'),
                    offer.extract_path.exists(),
                ))

        return tuple(snapshot)

    def file_list_container_is_live(self) -> bool:
        return element_is_live(self.file_list_container)

    def file_list(self) -> None:
        self.drawer_header().render()
        ui.upload(on_upload=self.handle_upload, auto_upload=True).classes('w-full')
        self.drawer_toolbar().render()
        projects = self.project_service.list_projects()
        self.drawer_tree().render(projects)

    def create_project(self, project_name: str | None) -> None:
        self.handle_action_result(self.actions.create_project(project_name))

    def drawer_header(self) -> DrawerHeader:
        return DrawerHeader(
            state=self.state,
            project_service=self.project_service,
            create_project=self.create_project,
        )

    def drawer_toolbar(self) -> DrawerToolbar:
        return DrawerToolbar(
            state=self.state,
            actions=self.actions,
            open_offer=self.open_file,
            open_project=self.open_project_comparison,
            extract_offer=self.request_extract,
            reset_offer=self.clear_extraction_button,
            rename_offer=self.rename_file,
            rename_project=self.rename_project,
            move_offer=self.move_button,
            delete_offer=self.delete_file,
            delete_project=self.delete_project,
            toggle_selection_mode=self.toggle_selection_mode,
            bulk_extract=self.bulk_extract,
            bulk_move=self.bulk_move,
            bulk_delete=self.bulk_delete,
        )

    def drawer_tree(self) -> DrawerTree:
        return DrawerTree(
            state=self.state,
            actions=self.actions,
            project_service=self.project_service,
            open_offer=self.open_file,
            select_project=self.set_project_expanded,
            search_changed=self.set_tree_search,
            toggle_offer_selected=self.set_offer_selected,
            cancel_extraction=self.cancel_extraction,
        )

    def toggle_selection_mode(self) -> None:
        self.state.selection_mode = not self.state.selection_mode
        if not self.state.selection_mode:
            self.state.selected_offers.clear()
        self.refresh_safe()

    def set_offer_selected(self, offer: Offer, selected: bool) -> None:
        if selected:
            self.state.selected_offers.add(offer)
        else:
            self.state.selected_offers.discard(offer)
        self.refresh_safe()

    def bulk_extract(self) -> None:
        offers = list(self.state.selected_offers)
        if not offers:
            return
        self.handle_action_result(self.actions.bulk_extract_offers(offers))
        self.state.selected_offers.clear()

    def bulk_delete(self) -> None:
        offers = list(self.state.selected_offers)
        if not offers:
            return
        dialogs.bulk_delete_dialog(offers, on_confirm=lambda offers=offers: self._confirm_bulk_delete(offers))

    def _confirm_bulk_delete(self, offers: list[Offer]) -> None:
        self.handle_action_result(self.actions.bulk_delete_offers(offers))
        self.state.selected_offers.clear()

    def bulk_move(self) -> None:
        offers = list(self.state.selected_offers)
        if not offers:
            return

        project_options = [project.name for project in self.project_service.list_projects()]
        if UNASSIGNED_PROJECT not in project_options:
            project_options.insert(0, UNASSIGNED_PROJECT)

        dialogs.bulk_move_dialog(
            offers,
            project_options=project_options,
            on_save=lambda target_project, offers=offers: self._confirm_bulk_move(offers, target_project),
        )

    def _confirm_bulk_move(self, offers: list[Offer], target_project: str | None) -> bool:
        result = self.actions.bulk_move_offers(offers, target_project)
        self.handle_action_result(result)
        self.state.selected_offers.clear()
        return result.success

    def set_tree_search(self, value: str) -> None:
        self.state.tree_search = value
        self.refresh_safe()

    def set_project_expanded(self, project: Project, expanded: bool) -> None:
        self.handle_action_result(self.actions.select_project(project))
        project_name = project.name
        if expanded:
            self.state.expanded_project_names.add(project_name)
            return

        self.state.expanded_project_names.discard(project_name)

    def open_file(self, offer: Offer) -> None:
        self.handle_action_result(self.actions.open_offer(offer))

    def open_project_comparison(self, project: Project) -> None:
        self.handle_action_result(self.actions.open_project_comparison(project))

    def delete_file(self, offer: Offer) -> None:
        self.handle_action_result(self.actions.delete_offer(offer))

    def clear_extraction_button(self, offer: Offer) -> None:
        dialogs.reset_extraction_button(
            offer,
            on_confirm=lambda selected_offer=offer: self.clear_extraction(selected_offer),
        )

    def clear_extraction(self, offer: Offer) -> None:
        self.handle_action_result(self.actions.clear_extraction(offer))

    def rename_file(self, offer: Offer, new_name: str | None) -> bool:
        return self.handle_action_result(self.actions.rename_offer(offer, new_name))

    def rename_project(self, project: Project, new_name: str | None) -> bool:
        return self.handle_action_result(self.actions.rename_project(project, new_name))

    def delete_project(self, project: Project) -> bool:
        return self.handle_action_result(self.actions.delete_project(project))

    def move_file(self, offer: Offer, target_project: str | None) -> bool:
        return self.handle_action_result(self.actions.move_offer(offer, target_project))

    def handle_action_result(self, result: DrawerActionResult) -> bool:
        if result.message:
            ui.notify(result.message)
        if result.refresh_right_side:
            self.schedule_right_side_refresh()
        if result.refresh_drawer:
            self.schedule_refresh()
        return result.success

    def schedule_right_side_refresh(self) -> None:
        asyncio.ensure_future(self._deferred_right_side_refresh())

    async def _deferred_right_side_refresh(self) -> None:
        await asyncio.sleep(0.05)
        try:
            self.refresh_right_side()
        except RuntimeError:
            pass

    def move_button(self, offer: Offer) -> None:
        project_options = [project.name for project in self.project_service.list_projects()]
        if UNASSIGNED_PROJECT not in project_options:
            project_options.insert(0, UNASSIGNED_PROJECT)
        current_project = offer.project_name

        dialogs.move_offer_dialog(
            offer,
            project_options=project_options,
            current_project=current_project,
            on_save=lambda target_project, selected_offer=offer: self.move_file(selected_offer, target_project),
        )

    def request_extract(self, offer: Offer) -> None:
        self.handle_action_result(self.actions.request_extract(offer))

    def cancel_extraction(self, offer: Offer) -> None:
        self.handle_action_result(self.actions.cancel_extraction(offer))

    def delete_selected(self) -> None:
        """Trigger the toolbar's delete action for the current selection (used by the Delete key shortcut)."""
        self.drawer_toolbar().delete_selected()
